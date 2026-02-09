import os
import sys
import json
import logging as log

import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.paths import RAW_CSV_PATH, PROCESSED_DIR, ARTIFACTS_DIR
from src.features.transformers import FraudFeatureEngineer, FeatureConfig

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



def make_label(df: pd.DataFrame, cfg: FeatureConfig) -> pd.Series:
    if cfg.label_col not in df.columns:
        raise ValueError(f"No existe la columna de label: {cfg.label_col}")

    y = (df[cfg.label_col].astype(str) == cfg.positive_label_value).astype("int8")
    return y


def temporal_split_train_valid_test(df: pd.DataFrame, time_col: str):
    ts = pd.to_datetime(df[time_col], errors="coerce")
    if ts.isna().mean() > 0.2:
        raise ValueError(f"Demasiados nulos al parsear {time_col}. Revisa el formato.")

    month = ts.dt.to_period("M")
    months_sorted = sorted(month.dropna().unique())

    if len(months_sorted) < 3:
        raise ValueError(
            "Necesitas al menos 3 meses para Train/Valid/Test por mes. "
            "Si tienes 2, hacemos test por últimos N días."
        )

    test_month = months_sorted[-1]
    valid_month = months_sorted[-2]

    train_df = df[month < valid_month].copy()
    valid_df = df[month == valid_month].copy()
    test_df  = df[month == test_month].copy()

    return train_df, valid_df, test_df, str(valid_month), str(test_month)


def main():
    cfg = FeatureConfig()

    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"No encontré el CSV en {RAW_CSV_PATH}. Colócalo ahí o ajusta config/paths.py"
        )

    df = pd.read_csv(RAW_CSV_PATH)

    # (Opcional) Filtra a transacciones que representan compras reales
    if "Transaction type" in df.columns:
        df = df[df["Transaction type"].astype(str) == "AUTHORIZATION_AND_CAPTURE"].copy()

    # El label se genera desde Response code (pero NO se usa como feature)
    y = make_label(df, cfg)

    # time columns
    time_col = cfg.time_col_primary if cfg.time_col_primary in df.columns else cfg.time_col_fallback
    train_df, valid_df, test_df, valid_month, test_month = temporal_split_train_valid_test(df, time_col=time_col)

    y_train = y.loc[train_df.index].copy()
    y_valid = y.loc[valid_df.index].copy()
    y_test = y.loc[test_df.index].copy()

    # Feature Engineering (fit en train; transform en train/valid)
    fe = FraudFeatureEngineer(cfg)
    fe.fit(train_df, y_train)

    X_train = fe.transform(train_df)
    X_valid = fe.transform(valid_df)
    X_test = fe.transform(test_df)

    # save local
    X_train_path = PROCESSED_DIR / "X_train.parquet"
    y_train_path = PROCESSED_DIR / "y_train.parquet"
    X_valid_path = PROCESSED_DIR / "X_valid.parquet"
    y_valid_path = PROCESSED_DIR / "y_valid.parquet"
    X_test_path = PROCESSED_DIR / "X_test.parquet"
    y_test_path = PROCESSED_DIR / "y_test.parquet"

    X_train.to_parquet(X_train_path, index=False)
    pd.DataFrame({"y": y_train.values}).to_parquet(y_train_path, index=False)

    X_valid.to_parquet(X_valid_path, index=False)
    pd.DataFrame({"y": y_valid.values}).to_parquet(y_valid_path, index=False)

    X_test.to_parquet(X_test_path, index=False)
    pd.DataFrame({"y": y_test.values}).to_parquet(y_test_path, index=False)

    # Guarda transformer + metadata de features
    joblib.dump(fe, ARTIFACTS_DIR / "feature_engineer.pkl")

    meta = {
        "time_col_used": fe.time_col_used_,
        "feature_names": fe.feature_names_,
        "categorical_feature_names": fe.get_categorical_feature_names(),
        "n_train": int(X_train.shape[0]),
        "n_valid": int(X_valid.shape[0]),
        "n_test": int(X_test.shape[0]),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_valid": float(y_valid.mean()),
        "positive_rate_test": float(y_test.mean()),
        "valid_month": valid_month,
        "test_month": test_month,
    }
    with open(ARTIFACTS_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log.info("✅ Splits + features guardados.")
    log.info(f" - {X_train_path}")
    log.info(f" - {X_valid_path}")
    log.info(f" - {X_test_path}")
    log.info(f" - {ARTIFACTS_DIR / 'feature_engineer.pkl'}")
    log.info(f" - {ARTIFACTS_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()
