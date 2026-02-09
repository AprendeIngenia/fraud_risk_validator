import os
import sys
import time
import logging as log

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference.engine import InferenceEngine

def main():
    engine = InferenceEngine()

    # Simulación de transacción: Cliente en Colombia con tarjeta de US (Riesgo Geográfico)
    new_transaction = {
        "Creation date": "2026-02-08 14:00:00",
        "Country": "CO",
        "Valor": 1500000,
        "Processing value": 1500000,
        "Merchant Id": "234543",
        "Account Id": "833883",
        "BIN Bank": "549659",
        "Country BIN ISO": "US",  # <--- Mismatch país
        "Card type": "CREDIT",
        "Franchise": "MASTERCARD",
        "Transaction type": "AUTHORIZATION_AND_CAPTURE",
        "Payment method": "CRED_MASTERCARD",
        "Payment model": "GATEWAY",
        "Transaction origin": "POST_API",
        "Accreditation model": "IMMEDIATE",
        "Transaction currency": "COP",
        "Processing currency": "COP",
        "Visible number": "549659******9663",
        "Days to deposit": 0,
        "mes": 2
    }

    # Medición de latencia
    start_time = time.perf_counter()
    response = engine.predict(new_transaction)
    latency = (time.perf_counter() - start_time) * 1000

    log.info(f"\n--- Resultado de Validación de Riesgo ---")
    log.info(f"ID: {response.transaction_id}")
    log.info(f"Score: {response.score}")
    log.info(f"Decisión: {response.decision.value}")
    log.info(f"Latencia: {latency:.2f} ms")

if __name__ == "__main__":
    main()