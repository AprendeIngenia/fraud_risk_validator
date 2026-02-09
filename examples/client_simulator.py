import requests
import time
import json

BASE_URL = "http://localhost:8000/v1/validate"

test_scenarios = [
    {
        "name": "Cliente VIP (Bajo Riesgo)",
        "payload": {
            "Creation date": "2026-02-08 15:00:00",
            "Country": "CO",
            "Valor": 25000.0,
            "Processing value": 25000.0,
            "Merchant Id": "234543",
            "Account Id": "833883",
            "BIN Bank": "549659",
            "Country BIN ISO": "CO",
            "Card type": "DEBIT",
            "Franchise": "VISA",
            "Transaction type": "AUTHORIZATION_AND_CAPTURE",
            "Payment method": "DEB_VISA",
            "Payment model": "GATEWAY",
            "Transaction origin": "POST_API",
            "Accreditation model": "IMMEDIATE",
            "Transaction currency": "COP",
            "Processing currency": "COP",
            "Visible number": "549659******1111"
        }
    },
    {
        "name": "Intento Internacional (Riesgo Alto - Sesgo US)",
        "payload": {
            "Creation date": "2026-02-08 03:00:00",
            "Country": "CO",
            "Valor": 900000.0,
            "Processing value": 900000.0,
            "Merchant Id": "234543",
            "Account Id": "999999",
            "BIN Bank": "411111",
            "Country BIN ISO": "US",
            "Card type": "CREDIT",
            "Franchise": "MASTERCARD",
            "Transaction type": "AUTHORIZATION_AND_CAPTURE",
            "Payment method": "CRED_MASTERCARD",
            "Payment model": "GATEWAY",
            "Transaction origin": "POST_API",
            "Accreditation model": "IMMEDIATE",
            "Transaction currency": "COP",
            "Processing currency": "COP",
            "Visible number": "411111******2222"
        }
    }
]


def run_simulation():
    print(f"🚀 Iniciando simulación de cliente...\n" + "=" * 50)

    for scenario in test_scenarios:
        print(f"Testing: {scenario['name']}")
        start = time.perf_counter()

        try:
            response = requests.post(BASE_URL, json=scenario['payload'])
            elapsed = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                res = response.json()
                print(f"  Result: {res['decision']} | Score: {res['score']}")
                print(f"  Internal Latency: {res['latency_ms']}ms | Round-trip: {elapsed:.2f}ms")
            else:
                print(f"  ❌ Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"  ❌ Error de conexión: {str(e)}")
        print("-" * 50)


if __name__ == "__main__":
    run_simulation()