# src/train/eval.py
import os
import sys
import json
import logging as log
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss,
    confusion_matrix
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.paths import PROCESSED_DIR, ARTIFACTS_DIR, MODELS_DIR, RAW_CSV_PATH
from src.features.transformers import FeatureConfig

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_clip_probs(y_prob: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=float)
    return np.clip(y_prob, eps, 1.0 - eps)


def safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Evita problemas por y_prob=0/1 y compatibilidad con versiones de sklearn
    donde 'eps' ya no existe en log_loss.
    """
    y_prob2 = safe_clip_probs(y_prob, eps=1e-15)
    return float(log_loss(y_true, y_prob2, labels=[0, 1]))


def get_best_iteration(model):
    # Booster: best_iteration
    bi = getattr(model, "best_iteration", None)
    if isinstance(bi, (int, np.integer)) and bi > 0:
        return int(bi)

    # sklearn wrapper: best_iteration_
    bi2 = getattr(model, "best_iteration_", None)
    if isinstance(bi2, (int, np.integer)) and bi2 > 0:
        return int(bi2)

    return None


def predict_proba_1(model, X: pd.DataFrame, best_iter=None) -> np.ndarray:
    """
    Devuelve probabilidad de clase 1 para Booster o LGBMClassifier.
    """
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X, num_iteration=best_iter)[:, 1]
        except TypeError:
            # algunos wrappers no aceptan num_iteration en predict_proba
            proba = model.predict_proba(X)[:, 1]
    else:
        # Booster
        proba = model.predict(X, num_iteration=best_iter)

    return safe_clip_probs(np.asarray(proba, dtype=float), eps=1e-15)


def confusion_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return {
        "thr": float(thr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "accuracy": float(acc),
    }


# -----------------------------
# Plots
# -----------------------------
def save_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> dict:
    ensure_dir(out_dir)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    roc_path = out_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    pr_path = out_dir / "pr_curve.png"
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()

    log.info(f"📈 ROC guardada: {roc_path}")
    log.info(f"📈 PR guardada: {pr_path}")
    return {"roc_curve_png": str(roc_path), "pr_curve_png": str(pr_path)}


def plot_feature_importance(model, feature_names, out_dir: Path, top_n: int = 25) -> dict:
    """
    Genera feature_importance.png + feature_importance.csv.
    Soporta Booster (feature_importance) y wrapper (feature_importances_).
    """
    ensure_dir(out_dir)
    out_png = out_dir / "feature_importance.png"
    out_csv = out_dir / "feature_importance.csv"

    # intentamos extraer importancias tipo 'gain' si es Booster
    importance = None
    importance_type_used = None

    # Booster API
    if hasattr(model, "feature_importance"):
        try:
            importance = model.feature_importance(importance_type="gain")
            importance_type_used = "gain"
        except Exception:
            try:
                importance = model.feature_importance(importance_type="split")
                importance_type_used = "split"
            except Exception:
                importance = None

        # feature names desde el modelo si existen
        try:
            fn_model = model.feature_name()
            if fn_model and len(fn_model) == len(importance):
                feature_names = fn_model
        except Exception:
            pass

    # sklearn wrapper API
    if importance is None and hasattr(model, "feature_importances_"):
        try:
            importance = np.asarray(model.feature_importances_, dtype=float)
            importance_type_used = "feature_importances_"
        except Exception:
            importance = None

    if importance is None:
        log.warning("⚠️ No pude extraer feature importance del modelo. Se omite feature_importance.png.")
        return {"feature_importance_png": None, "feature_importance_csv": None, "importance_type": None}

    importance = np.asarray(importance, dtype=float)
    feature_names = list(feature_names)

    if len(feature_names) != len(importance):
        # fallback: si no coincide, usa nombres genéricos
        log.warning("⚠️ Longitud de feature_names no coincide con importancias. Usando nombres genéricos.")
        feature_names = [f"f_{i}" for i in range(len(importance))]

    df_imp = pd.DataFrame({"feature": feature_names, "importance": importance})
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)

    # guardamos csv completo
    df_imp.to_csv(out_csv, index=False, encoding="utf-8")

    # plot top_n
    top = df_imp.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("importance")
    plt.ylabel("feature")
    title = f"Top {min(top_n, len(df_imp))} Features (type={importance_type_used})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    log.info(f"🧠 Feature importance guardada: {out_png}")
    return {
        "feature_importance_png": str(out_png),
        "feature_importance_csv": str(out_csv),
        "importance_type": importance_type_used,
        "top_features": df_imp.head(30).to_dict(orient="records"),
    }


# -----------------------------
# Triage thresholds
# -----------------------------
def triage_stats(y_true: np.ndarray, y_prob: np.ndarray, T_low: float, T_high: float) -> dict:
    """
    3 buckets:
      - approve: y_prob < T_low
      - review:  T_low <= y_prob < T_high
      - decline: y_prob >= T_high
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    approve = y_prob < T_low
    decline = y_prob >= T_high
    review = (~approve) & (~decline)

    def _rate(mask):
        return float(mask.mean()) if len(mask) else 0.0

    approve_rate = _rate(approve)
    decline_rate = _rate(decline)
    review_rate = _rate(review)

    approve_fraud_rate = float(y_true[approve].mean()) if approve.any() else None
    decline_precision = None
    decline_recall = None

    if decline.any():
        yt = y_true[decline]
        tp = int(yt.sum())
        fp = int(len(yt) - tp)
        decline_precision = float(tp / (tp + fp)) if (tp + fp) else 0.0

        # recall global respecto a todos los positivos
        total_pos = int(y_true.sum())
        decline_recall = float(tp / total_pos) if total_pos else None

    return {
        "T_low": float(T_low),
        "T_high": float(T_high),
        "rates": {
            "approve_rate": approve_rate,
            "review_rate": review_rate,
            "decline_rate": decline_rate,
        },
        "quality": {
            "approve_fraud_rate": approve_fraud_rate,   # fraude que se “cuela” en aprobadas
            "decline_precision": decline_precision,     # qué tan “seguro” es el rechazo
            "decline_recall": decline_recall,           # cuánto fraude capturas SOLO con auto-decline
        },
        "counts": {
            "approve_n": int(approve.sum()),
            "review_n": int(review.sum()),
            "decline_n": int(decline.sum()),
        }
    }


def recommend_triage_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_approve_fraud_rate: float = 0.01,
    target_decline_precision: float = 0.95,
    min_auto_approve_rate: float = 0.20,
    min_auto_decline_rate: float = 0.01,
    grid_size: int = 199,
) -> tuple[float, float, dict]:
    """
    - T_low (approve): máximo umbral tal que el FRAUD_RATE en aprobadas <= target_approve_fraud_rate
    - T_high (decline): mínimo umbral tal que la PRECISION en rechazadas >= target_decline_precision

    Además, exige coberturas mínimas (min_auto_approve_rate / min_auto_decline_rate).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    thresholds = np.linspace(0.001, 0.999, grid_size)

    # --- T_low (approve) ---
    approve_candidates = []
    for t in thresholds:
        approve = y_prob < t
        ar = float(approve.mean())
        if ar < min_auto_approve_rate:
            continue
        if not approve.any():
            continue
        fraud_rate = float(y_true[approve].mean())
        if fraud_rate <= target_approve_fraud_rate:
            approve_candidates.append({"thr": float(t), "approve_rate": ar, "approve_fraud_rate": fraud_rate})

    if approve_candidates:
        # queremos el máximo t (más approve) manteniendo control de fraude en aprobadas
        best_low = max(approve_candidates, key=lambda x: x["thr"])
        T_low = float(best_low["thr"])
    else:
        # fallback conservador: percentil 20
        T_low = float(np.quantile(y_prob, 0.20))

    # --- T_high (decline) ---
    decline_candidates = []
    total_pos = int(y_true.sum())
    for t in thresholds:
        decline = y_prob >= t
        dr = float(decline.mean())
        if dr < min_auto_decline_rate:
            continue
        if not decline.any():
            continue
        yt = y_true[decline]
        tp = int(yt.sum())
        fp = int(len(yt) - tp)
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall_from_declines = float(tp / total_pos) if total_pos else None

        if precision >= target_decline_precision:
            decline_candidates.append({
                "thr": float(t),
                "decline_rate": dr,
                "decline_precision": precision,
                "decline_recall": recall_from_declines
            })

    if decline_candidates:
        # queremos el mínimo t (más coverage de decline) manteniendo alta precision
        best_high = min(decline_candidates, key=lambda x: x["thr"])
        T_high = float(best_high["thr"])
    else:
        # fallback estricto: percentil 95
        T_high = float(np.quantile(y_prob, 0.95))

    # Asegura orden y margen
    if T_low >= T_high:
        # ajuste de emergencia
        T_low = min(T_low, 0.35)
        T_high = max(T_high, 0.65)

    debug = {
        "targets": {
            "target_approve_fraud_rate": float(target_approve_fraud_rate),
            "target_decline_precision": float(target_decline_precision),
            "min_auto_approve_rate": float(min_auto_approve_rate),
            "min_auto_decline_rate": float(min_auto_decline_rate),
        },
        "candidates": {
            "approve_candidates_found": int(len(approve_candidates)),
            "decline_candidates_found": int(len(decline_candidates)),
        }
    }
    return T_low, T_high, debug


# -----------------------------
# Segment report + examples
# -----------------------------
def segment_report(df_raw: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray,
                   col: str, thr: float = 0.5, topk: int = 10, min_n: int = 200) -> list:
    if col not in df_raw.columns:
        return []

    s = df_raw[col].astype(str).fillna("__NA__")
    top_vals = s.value_counts().head(topk).index.tolist()
    s = s.where(s.isin(top_vals), "__OTHER__")

    y_pred = (y_prob >= thr).astype(int)
    out = []

    for v in pd.unique(s):
        mask = (s == v).values
        if mask.sum() < min_n:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        base_rate = float(np.mean(yt)) if len(yt) else None
        pred_rate = float(np.mean(yp)) if len(yp) else None

        recall = float(tp/(tp+fn)) if (tp+fn) else None
        fpr = float(fp/(fp+tn)) if (fp+tn) else None

        out.append({
            "value": v,
            "n": int(mask.sum()),
            "base_rate": base_rate,
            "pred_rate": pred_rate,
            "recall@thr": recall,
            "fpr@thr": fpr,
            "avg_score": float(np.mean(y_prob[mask])) if mask.sum() else None,
        })

    return sorted(out, key=lambda x: x["n"], reverse=True)


def load_raw_split(meta: dict, split_name: str) -> pd.DataFrame | None:
    """
    Intenta reconstruir el raw del split (por mes) para diagnóstico/segmentos.
    Si no puede, devuelve None.
    """
    try:
        df_raw = pd.read_csv(RAW_CSV_PATH)
    except Exception as e:
        log.warning(f"⚠️ No pude leer RAW_CSV_PATH para segmentos/ejemplos: {e}")
        return None

    # mismo filtro que pipeline (si aplica)
    if "Transaction type" in df_raw.columns:
        df_raw = df_raw[df_raw["Transaction type"].astype(str) == "AUTHORIZATION_AND_CAPTURE"].copy()

    cfg = FeatureConfig()
    time_col = cfg.time_col_primary if cfg.time_col_primary in df_raw.columns else cfg.time_col_fallback
    ts = pd.to_datetime(df_raw[time_col], errors="coerce")
    month = ts.dt.to_period("M").astype(str)

    month_key = "test_month" if split_name == "test" else "valid_month"
    target_month = meta.get(month_key)

    if target_month:
        df_raw_split = df_raw[month == target_month].copy()
    else:
        df_raw_split = df_raw.copy()

    df_raw_split = df_raw_split.reset_index(drop=True)
    return df_raw_split


def write_summary_files(summary_json: dict, out_dir: Path) -> dict:
    ensure_dir(out_dir)
    out_json = out_dir / "eval_summary.json"
    out_md = out_dir / "eval_summary.md"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    # Markdown “de batalla”
    lines = []
    lines.append(f"# Eval Summary ({summary_json.get('split_used')})")
    lines.append("")
    lines.append(f"- Timestamp: {summary_json.get('timestamp')}")
    lines.append(f"- N: {summary_json.get('n')} | base_rate: {summary_json.get('base_rate'):.4f}")
    lines.append("")
    m = summary_json.get("metrics", {})
    lines.append("## Global metrics")
    lines.append(f"- ROC-AUC: {m.get('roc_auc')}")
    lines.append(f"- AP (AUPRC): {m.get('average_precision')}")
    lines.append(f"- LogLoss: {m.get('logloss')}")
    lines.append(f"- Brier: {m.get('brier')}")
    lines.append("")
    lines.append("## Threshold @ 0.5")
    c05 = summary_json.get("threshold_0_5", {})
    lines.append(f"- precision: {c05.get('precision')}, recall: {c05.get('recall')}, fpr: {c05.get('fpr')}, fnr: {c05.get('fnr')}")
    lines.append("")
    lines.append("## Triage thresholds")
    t = summary_json.get("triage", {})
    lines.append(f"- T_low (approve): {t.get('T_low')}")
    lines.append(f"- T_high (decline): {t.get('T_high')}")
    q = t.get("quality", {})
    r = t.get("rates", {})
    lines.append(f"- approve_rate: {r.get('approve_rate')}, review_rate: {r.get('review_rate')}, decline_rate: {r.get('decline_rate')}")
    lines.append(f"- approve_fraud_rate: {q.get('approve_fraud_rate')}, decline_precision: {q.get('decline_precision')}, decline_recall: {q.get('decline_recall')}")
    lines.append("")
    lines.append("## Top features")
    topf = summary_json.get("top_features", [])[:15]
    for row in topf:
        lines.append(f"- {row['feature']}: {row['importance']}")
    lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"eval_summary_json": str(out_json), "eval_summary_md": str(out_md)}


# -----------------------------
# Main
# -----------------------------
def main():
    log.info("🧐 Evaluación (TEST si existe, si no VALID).")

    eval_dir = ARTIFACTS_DIR / "eval"
    ensure_dir(eval_dir)

    # 1) meta
    meta_path = ARTIFACTS_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    # 2) carga modelo
    model_path = (MODELS_DIR / "LGBM" / "lgbm_fraud_model_v1.pkl")
    if not model_path.exists():
        model_path = MODELS_DIR / "lgbm_fraud_model_v1.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No encuentro el modelo en: {model_path}")

    model = joblib.load(model_path)

    # 3) split
    X_path = PROCESSED_DIR / "X_test.parquet"
    y_path = PROCESSED_DIR / "y_test.parquet"
    split_name = "test"

    if not X_path.exists() or not y_path.exists():
        X_path = PROCESSED_DIR / "X_valid.parquet"
        y_path = PROCESSED_DIR / "y_valid.parquet"
        split_name = "valid"

    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)["y"].astype(int).values

    base_rate = float(np.mean(y)) if len(y) else 0.0
    log.info(f"📦 Split usado: {split_name} | X={X.shape} | base_rate={base_rate:.4f}")

    # 4) predict
    best_iter = get_best_iteration(model)
    y_prob = predict_proba_1(model, X, best_iter=best_iter)

    # 5) métricas globales
    roc_auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else None
    ap = average_precision_score(y, y_prob) if len(np.unique(y)) > 1 else None
    ll = safe_log_loss(y, y_prob)
    brier = float(brier_score_loss(y, y_prob))

    log.info(f"🏆 ROC-AUC: {roc_auc}")
    log.info(f"🏆 Average Precision (AP): {ap}")
    log.info(f"📉 LogLoss: {ll:.4f} | Brier: {brier:.4f}")

    # 6) curvas ROC/PR
    curves_paths = save_roc_pr_curves(y, y_prob, eval_dir)

    # 7) barrido de umbrales (para csv)
    thresholds = np.linspace(0.01, 0.99, 99)
    sweep = [confusion_at_threshold(y, y_prob, float(t)) for t in thresholds]
    sweep_df = pd.DataFrame(sweep)
    sweep_csv_path = eval_dir / "threshold_sweep.csv"
    sweep_df.to_csv(sweep_csv_path, index=False, encoding="utf-8")
    log.info(f"📌 Threshold sweep guardado: {sweep_csv_path}")

    # 8) triage thresholds (recomendación corregida)
    T_low, T_high, triage_debug = recommend_triage_thresholds(
        y, y_prob,
        target_approve_fraud_rate=0.01,   # ajusta según negocio (fraude tolerado en aprobadas)
        target_decline_precision=0.95,    # rechazos casi seguros
        min_auto_approve_rate=0.20,       # al menos 20% auto-approve (si se puede)
        min_auto_decline_rate=0.01        # al menos 1% auto-decline (si se puede)
    )
    triage = triage_stats(y, y_prob, T_low, T_high)
    log.info(f"🎯 Triage recomendado: T_low={T_low:.2f} | T_high={T_high:.2f}")

    # 9) threshold diagnóstico clásico
    thr_diag = 0.5
    diag = confusion_at_threshold(y, y_prob, thr_diag)

    # 10) raw split para segmentos y ejemplos (opcional)
    df_raw_split = load_raw_split(meta, split_name)
    segment_cols = [
        "Payment method", "Payment model", "Transaction origin",
        "Country", "Country BIN ISO", "BIN Bank", "Card type",
        "Franchise", "Accreditation model"
    ]

    segment_metrics = {}
    top_fp_records = []
    top_fn_records = []

    if df_raw_split is not None:
        # alinea por tamaño
        n = min(len(df_raw_split), len(y_prob))
        if len(df_raw_split) != len(y_prob):
            log.warning(f"⚠️ RAW split (n={len(df_raw_split)}) y y_prob (n={len(y_prob)}) difieren. "
                        f"Se alinea a n={n} (por orden).")

        df_raw_split = df_raw_split.iloc[:n].copy()
        y2 = y[:n]
        y_prob2 = y_prob[:n]

        # segmentos @ thr=0.5
        for col in segment_cols:
            segment_metrics[col] = segment_report(df_raw_split, y2, y_prob2, col, thr=thr_diag, topk=10, min_n=200)

        # top FP/FN @ thr=0.5
        y_pred2 = (y_prob2 >= thr_diag).astype(int)
        fp_idx = np.where((y_pred2 == 1) & (y2 == 0))[0]
        fn_idx = np.where((y_pred2 == 0) & (y2 == 1))[0]

        top_fp = fp_idx[np.argsort(-y_prob2[fp_idx])][:50] if len(fp_idx) else []
        top_fn = fn_idx[np.argsort(+y_prob2[fn_idx])][:50] if len(fn_idx) else []

        sample_cols = [c for c in segment_cols if c in df_raw_split.columns]
        if "Valor" in df_raw_split.columns:
            sample_cols += ["Valor"]

        if len(top_fp):
            top_fp_records = df_raw_split.iloc[list(top_fp)][sample_cols].to_dict(orient="records")
        if len(top_fn):
            top_fn_records = df_raw_split.iloc[list(top_fn)][sample_cols].to_dict(orient="records")

    # 11) feature importance
    feature_names = list(X.columns)
    fi = plot_feature_importance(model, feature_names, eval_dir, top_n=25)

    # 12) reporte detallado (como bitácora completa)
    report = {
        "split_used": split_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "paths": {
            "model_path": str(model_path),
            "X_path": str(X_path),
            "y_path": str(y_path),
            **curves_paths,
            "threshold_sweep_csv": str(sweep_csv_path),
            "feature_importance_png": fi.get("feature_importance_png"),
            "feature_importance_csv": fi.get("feature_importance_csv"),
        },
        "meta": {
            "valid_month": meta.get("valid_month"),
            "test_month": meta.get("test_month"),
            "feature_count": int(len(meta.get("feature_names", []))) if meta.get("feature_names") else int(X.shape[1]),
            "best_iteration": int(best_iter) if best_iter is not None else None
        },
        "metrics": {
            "roc_auc": roc_auc,
            "average_precision": ap,
            "logloss": ll,
            "brier": brier,
            "n": int(len(y)),
            "base_rate": base_rate
        },
        "threshold_0_5": diag,
        "triage": {
            **triage,
            "notes": {
                "T_low_meaning": "Auto-approve si score < T_low (controla fraude en aprobadas).",
                "T_high_meaning": "Auto-decline si score >= T_high (controla precision en rechazadas).",
            },
            "debug": triage_debug
        },
        "threshold_sweep_0_01_to_0_99": sweep,
        "segment_report_thr_0_5": segment_metrics,
        "top_false_positives_thr_0_5": top_fp_records,
        "top_false_negatives_thr_0_5": top_fn_records,
        "feature_importance_top30": fi.get("top_features", [])[:30],
        "recommendations": [
            "Si FP se concentran por BIN/Country: considera umbrales por segmento o calibración.",
            "Agrega velocity multi-ventana (5m/1h/6h/24h) para separar ráfagas vs comportamiento normal.",
            "Monitorea drift mensual y reentrena con ventana móvil (rolling) o por cohortes."
        ]
    }

    out_path = eval_dir / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info(f"✅ Reporte detallado guardado en: {out_path}")

    # 13) resumen técnico compacto (para vista general)
    summary_json = {
        "split_used": split_name,
        "timestamp": report["timestamp"],
        "n": int(len(y)),
        "base_rate": base_rate,
        "metrics": report["metrics"],
        "threshold_0_5": report["threshold_0_5"],
        "triage": report["triage"],
        "top_features": report.get("feature_importance_top30", [])[:15],
        "paths": report["paths"],
    }

    summary_paths = write_summary_files(summary_json, eval_dir)
    log.info(f"🧾 Resumen guardado: {summary_paths['eval_summary_json']}")
    log.info(f"🧾 Resumen MD guardado: {summary_paths['eval_summary_md']}")

    log.info("🏁 Listo. Ya tienes detalle + resumen + curvas + importance + sweep.")


if __name__ == "__main__":
    main()
