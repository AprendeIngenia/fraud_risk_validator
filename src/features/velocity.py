# src/features/velocity.py
import pandas as pd


def rolling_count_last_24h(
        df: pd.DataFrame,
        time_col: str,
        key_col: str,
        out_col: str = "cnt_24h",
        window: str = "24h",
) -> pd.Series:
    """
    Conteo de ocurrencias de key_col en la ventana [t-24h, t] sin fuga temporal.
    """
    ts = pd.to_datetime(df[time_col], errors="coerce")
    key = df[key_col].astype("string").fillna("__NA__").astype(str)

    # Serie de salida alineada al índice original
    out = pd.Series(0, index=df.index, dtype="int32", name=out_col)

    valid_mask = ts.notna()
    if valid_mask.sum() == 0:
        return out

    d = pd.DataFrame({
        "ts": ts[valid_mask],
        "key": key[valid_mask],
        "counter": 1,
        "original_idx": df.index[valid_mask]
    }).sort_values("ts")

    d = d.set_index("ts")

    counts = (
        d.groupby("key", sort=False)["counter"]
        .rolling(window)
        .count()
        .reset_index(level=0, drop=True)
    )

    counts = (counts - 1).clip(lower=0).astype("int32")

    # Mapeamos de vuelta al índice original del DataFrame
    out.loc[d["original_idx"].values] = counts.values
    return out