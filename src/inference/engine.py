import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.paths import MODELS_DIR, ARTIFACTS_DIR
from src.inference.schemas import Decision, FraudResponse


class InferenceEngine:
    def __init__(self):
        # load artifacts
        self.fe = joblib.load(ARTIFACTS_DIR / "feature_engineer.pkl")
        self.model = joblib.load(MODELS_DIR / "LGBM" / "lgbm_fraud_model_v1.pkl")

        # thresholds
        self.T_low = 0.2036
        self.T_high = 0.7117

    def _get_decision(self, score: float) -> Decision:
        if score < self.T_low:
            return Decision.APPROVE
        elif score >= self.T_high:
            return Decision.REJECT
        return Decision.REVIEW

    def predict(self, raw_data: dict) -> FraudResponse:
        # 1. dataframe
        df = pd.DataFrame([raw_data])

        # 2. FEATURE ENGINEERING
        # En producción, 'cnt_account_24h' vendría de una consulta a Redis
        X = self.fe.transform(df)

        # 3. SCORING
        score = float(self.model.predict(X)[0])

        # 4. DECISION LOGIC
        decision = self._get_decision(score)

        return FraudResponse(
            transaction_id=raw_data.get("Order Id", "unknown"),
            score=round(score, 4),
            decision=decision,
            thresholds={"T_low": self.T_low, "T_high": self.T_high}
        )