import os
import sys
import joblib
import lightgbm as lgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.paths import MODELS_DIR

class LightGBMFraudModel:
    def __init__(self, params: dict):
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val, cat_features):
        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=cat_features)

        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=10)
            ]
        )

    def save(self, filename: str):
        path = MODELS_DIR / filename
        joblib.dump(self.model, path)
        return path