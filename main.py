import time
import logging as log
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from src.inference.engine import InferenceEngine
from src.inference.schemas import TransactionRequest, FraudResponse

model_assets = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lógica de "Warm-up"
    log.info("🚛 Cargando artefactos de Machine Learning...")
    model_assets["engine"] = InferenceEngine()
    log.info("✅ Servicio listo para recibir transacciones.")
    yield
    # Lógica de "Clean-up" (si fuera necesaria)
    model_assets.clear()


app = FastAPI(
    title="Fraud Risk Validator API",
    description="Servicio para detección de fraude en transacciones",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_version": "v1.0.0"}


@app.post("/v1/validate", response_model=FraudResponse)
async def validate_transaction(request: TransactionRequest):
    try:
        start_time = time.perf_counter()

        # global state
        engine: InferenceEngine = model_assets["engine"]

        # input
        raw_data = request.model_dump(by_alias=True)

        # predict
        result = engine.predict(raw_data)
        latency = (time.perf_counter() - start_time) * 1000

        # return
        return FraudResponse(
            transaction_id=result.transaction_id,
            score=result.score,
            decision=result.decision,
            thresholds=result.thresholds,
            latency_ms=round(latency, 2)
        )

    except Exception as e:
        log.error(f"❌ Error procesando transacción: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Error in Fraud Engine")