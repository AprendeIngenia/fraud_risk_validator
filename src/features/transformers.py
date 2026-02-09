import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.features.velocity import rolling_count_last_24h


FORBIDDEN_RESULT_COLUMNS = [
    "Status",
    "Response code",
    "Bank Response code",
    "Message response error",
    "Authorization code",
]


@dataclass
class FeatureConfig:
    time_col_primary: str = "Creation date"
    time_col_fallback: str = "Fecha"
    label_col: str = "Response code"
    positive_label_value: str = "ANTIFRAUD_REJECTED"

    # util columns
    id_like_cols: Tuple[str, ...] = ("Merchant Id", "Account Id", "BIN Bank", "Visible number")

    # drop columns
    drop_cols: Tuple[str, ...] = (
        "Order Id",         # normalmente único -> ruido/overfit
        "Reference",        # suele ser quasi-unique
        "Description",      # texto libre (mejor tratarlo aparte)
        "Fecha",            # si usamos Creation date como timestamp principal
        "Creation date",    # se reemplaza por features de tiempo
    )

    raw_id_cols_to_remove: Tuple[str, ...] = ("Account Id", "Visible number")


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer para:
    - Parsear fecha/hora
    - Construir features rápidas para producción (tabulares + tiempo + flags)
    - Encode categóricas en enteros con manejo de desconocidos
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

        # artifacts del fit
        self.categorical_cols_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.category_maps_: Dict[str, Dict[str, int]] = {}
        self.feature_names_: List[str] = []
        self.time_col_used_: Optional[str] = None

    def _parse_datetime(self, df: pd.DataFrame) -> pd.Series:
        cfg = self.config
        if cfg.time_col_primary in df.columns:
            ts = pd.to_datetime(df[cfg.time_col_primary], errors="coerce")
            if ts.notna().mean() > 0.95:
                self.time_col_used_ = cfg.time_col_primary
                return ts
        if cfg.time_col_fallback in df.columns:
            ts = pd.to_datetime(df[cfg.time_col_fallback], errors="coerce")
            self.time_col_used_ = cfg.time_col_fallback
            return ts

        raise ValueError("No se pudo parsear una columna de tiempo válida (Creation date / Fecha).")

    def _base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        X = df.copy()

        # Timestamp -> features de tiempo
        ts = self._parse_datetime(X)
        time_col_used = self.time_col_used_ if self.time_col_used_ else cfg.time_col_primary
        X["tx_hour"] = ts.dt.hour.fillna(-1).astype("int16")
        X["tx_dow"] = ts.dt.dayofweek.fillna(-1).astype("int16")
        X["tx_day"] = ts.dt.day.fillna(-1).astype("int16")
        X["tx_month"] = ts.dt.month.fillna(-1).astype("int16")
        X["tx_is_weekend"] = (X["tx_dow"].isin([5, 6])).astype("int8")

        # Monetary
        if "Valor" in X.columns:
            X["valor_log1p"] = np.log1p(pd.to_numeric(X["Valor"], errors="coerce").fillna(0)).astype("float32")

        if "Processing value" in X.columns:
            pv = pd.to_numeric(X["Processing value"], errors="coerce").fillna(0)
            X["processing_value_log1p"] = np.log1p(pv).astype("float32")

        # Mismatch flags
        if ("Country" in X.columns) and ("Country BIN ISO" in X.columns):
            X["country_mismatch"] = (X["Country"].astype(str) != X["Country BIN ISO"].astype(str)).astype("int8")

        if ("Transaction currency" in X.columns) and ("Processing currency" in X.columns):
            X["currency_mismatch"] = (
                X["Transaction currency"].astype(str) != X["Processing currency"].astype(str)
            ).astype("int8")

        if "Account Id" in X.columns:
            X["cnt_account_24h"] = rolling_count_last_24h(X, time_col_used, "Account Id", "cnt_account_24h")

        if "Visible number" in X.columns:
            X["cnt_card_24h"] = rolling_count_last_24h(X, time_col_used, "Visible number", "cnt_card_24h")

        # Tipo de columnas id-like a string para categorización consistente
        for c in cfg.id_like_cols:
            if c in X.columns:
                X[c] = X[c].astype(str)

        # Drop columnas prohibidas como features
        for c in FORBIDDEN_RESULT_COLUMNS:
            if c in X.columns:
                X.drop(columns=[c], inplace=True)

        # Drop columnas adicionales
        for c in cfg.drop_cols:
            if c in X.columns:
                X.drop(columns=[c], inplace=True)

        # Drop IDs
        for c in cfg.raw_id_cols_to_remove:
            if c in X.columns:
                X.drop(columns=[c], inplace=True)

        return X

    def fit(self, df: pd.DataFrame, y=None):
        X = self._base_features(df)

        # Identifica columnas categóricas vs numéricas
        # (nota: ids tratados como string -> categóricas)
        cat_cols = []
        num_cols = []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                num_cols.append(col)
            else:
                cat_cols.append(col)

        # Construye mapeos categoría->id (unknown => -1)
        self.category_maps_ = {}
        for c in cat_cols:
            # Manejo de nulos como string
            vals = X[c].fillna("__NA__").astype(str)
            cats = pd.Index(vals.unique()).sort_values()
            self.category_maps_[c] = {k: i for i, k in enumerate(cats.tolist())}

        self.categorical_cols_ = cat_cols
        self.numeric_cols_ = num_cols

        # Feature order final: numéricas + categóricas (encoded)
        self.feature_names_ = self.numeric_cols_ + self.categorical_cols_
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_:
            raise RuntimeError("Transformer no está fiteado. Llama fit() antes de transform().")

        X = self._base_features(df)

        # Asegura que existan columnas esperadas
        for col in self.numeric_cols_:
            if col not in X.columns:
                X[col] = 0.0
        for col in self.categorical_cols_:
            if col not in X.columns:
                X[col] = "__MISSING_COL__"

        # Encode categóricas
        for c in self.categorical_cols_:
            mapping = self.category_maps_.get(c, {})
            vals = X[c].fillna("__NA__").astype(str)
            X[c] = vals.map(mapping).fillna(-1).astype("int32")

        # Normaliza numéricas
        for c in self.numeric_cols_:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype("float32")

        # Orden final
        X = X[self.feature_names_].copy()
        return X

    def get_categorical_feature_names(self) -> List[str]:
        return list(self.categorical_cols_)
