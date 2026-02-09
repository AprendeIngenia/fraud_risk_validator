# src/train/trainer.py
import os
import sys
import json
import logging as log
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.paths import PROCESSED_DIR, ARTIFACTS_DIR, MODELS_DIR
from config.model_params import lgbm_fast_inference_params
from src.models.lgbm_wrapper import LightGBMFraudModel


def main():
    log.info("🚀 Iniciando entrenamiento del modelo...")

    # 1. load metadata
    with open(ARTIFACTS_DIR / "meta.json", "r") as f:
        meta = json.load(f)

    cat_features = meta["categorical_feature_names"]

    # 2. load splits
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["y"]
    X_valid = pd.read_parquet(PROCESSED_DIR / "X_valid.parquet")
    y_valid = pd.read_parquet(PROCESSED_DIR / "y_valid.parquet")["y"]

    log.info(f"📊 Dataset cargado. Train: {X_train.shape}, Valid: {X_valid.shape}")

    # 3. cinfog params
    params = lgbm_fast_inference_params()
    model_wrapper = LightGBMFraudModel(params)

    # 4. train
    model_wrapper.fit(
        X_train, y_train,
        X_valid, y_valid,
        cat_features=cat_features
    )

    # 5. save model artifact
    model_path = model_wrapper.save("LGBM/lgbm_fraud_model_v1.pkl")
    log.info(f"✅ Modelo entrenado y guardado en: {model_path}")


if __name__ == "__main__":
    main()