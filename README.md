# Fraud Risk Validator — API v1.0.0

Servicio de validación de riesgo de fraude basado en **LightGBM**, expuesto vía **FastAPI** y listo para contenerizar.
El sistema utiliza una arquitectura modular basada en principios SOLID para garantizar la mantenibilidad y la escalabilidad.

> Objetivo: operar con **triage por umbrales** (APPROVE / REVIEW / REJECT) para **recolectar mejores etiquetas**, ampliar dataset y reducir sesgos/overfitting de forma incremental.

---

## 1) Estructura del proyecto (alto nivel)

- `src/features/`
  - `transformers.py`: Feature engineering (tiempo, montos, mismatch, velocity 24h, encoding robusto)
  - `velocity.py`: rolling count 24h **sin fuga temporal**
- `src/data/`
  - `make_splits.py`: split temporal (train/valid/test por mes) + generación de `X_*.parquet`
- `src/train/`
  - `trainer.py`: entrenamiento LightGBM y guardado de modelo
  - `eval.py`: evaluación, curvas ROC/PR, thresholds triage, reportes JSON/summary
- `src/inference/`
  - `engine.py`: carga artifacts + scoring + decisión por thresholds
  - `schemas.py`: esquemas Pydantic para API
- `main.py`: API FastAPI (warmup, endpoint `/v1/validate`)
- `examples/`
  - `client_simulator.py`: cliente simple para probar el servicio

---

## 2) Entrenamiento y evaluación

### 2.1. Ubicar el dataset (CSV)
Para actualizar el modelo con nuevos datos, siga este flujo secuencial. Asegúrese de ubicar el archivo fuente en:

```
src/data/raw/base_datos_fraude.csv.
```

> Nota: la parte de datos debe versionarse con **DVC**. El contenedor de inferencia **no** necesita los datos.

### 2.2. Ejecutar el pipeline de entrenamiento

```bash
# 1) Split temporal + features (genera processed parquet + artifacts del feature engineer)
python src/data/make_splits.py

# 2) Entrenar modelo (guarda el modelo en src/models/LGBM/)
python src/train/trainer.py

# 3) Evaluación (genera curvas + reportes en src/artifacts/eval/)
python src/train/eval.py
```

### 2.3. Artefactos esperados

- `src/data/processed/`
  - `X_train.parquet`, `X_valid.parquet`, `X_test.parquet`
  - `y_train.parquet`, `y_valid.parquet`, `y_test.parquet`
- `src/artifacts/`
  - `feature_engineer.pkl`
  - `meta.json`
- `src/artifacts/eval/`
  - `roc_curve.png`
  - `pr_curve.png`
  - `eval_results.json` (reporte detallado)
  - `eval_summary.*` (resumen)
  - `feature_importance.csv`, `feature_importance.png`

---

## 3) Feature engineering (resumen técnico)

El modelo se alimenta **solo con variables disponibles pre-autorización** para evitar leakage.

1) **Tiempo (timestamp) sin fuga**
- `Creation date` (primaria) o `Fecha` (fallback)
- Features: `tx_hour`, `tx_dow`, `tx_day`, `tx_month`, `tx_is_weekend`

2) **Transformaciones monetarias**
- `valor_log1p` desde `Valor`
- `processing_value_log1p` desde `Processing value`

3) **Mismatch flags**
- `country_mismatch`: `Country` vs `Country BIN ISO`
- `currency_mismatch`: `Transaction currency` vs `Processing currency`

4) **Velocity 24h sin fuga**
- `cnt_account_24h`: conteo por `Account Id` en últimas 24h
- `cnt_card_24h`: conteo por `Visible number` en últimas 24h
- Se excluye el evento actual: `count-1` (evita leakage)

5) **Categóricas robustas**
- Mapeo categoría→int durante `fit()`
- Desconocidos en `transform()` → `-1`

6) **Drop de columnas peligrosas**
- Se eliminan columnas que son resultado o muy cercanas al label (`Response code`, `Authorization code`, etc.)
- Se eliminan columnas quasi-unique (p.ej. `Order Id`, `Reference`, `Description`) para reducir overfit

---

## 4) Construcción y ejecución del servicio (FastAPI)

### 4.1. Dockerfile (referencia)

```dockerfile
# Usamos una imagen ligera de Python 3.11
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

WORKDIR /app

# Instalamos dependencias del sistema necesarias para LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalamos librerías de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy
COPY config/ ./config/
COPY src/inference/ ./src/inference/
COPY src/artifacts/ ./src/artifacts/
COPY src/features/ ./src/features/
COPY src/models/ ./src/models/

COPY main.py .

# Exponemos el puerto de FastAPI
EXPOSE 8000

# Comando para arrancar el servicio en modo producción
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### 4.2. Build & run

```bash
docker build -t fraud-validator:v1 .
docker run -p 8000:8000 fraud-validator:v1
```

### 4.3. Health check

```bash
curl http://localhost:8000/health
```

### 4.4. Inferencia (ejemplo de request)

Endpoint:

- `POST /v1/validate`

El payload usa los **mismos nombres** (aliases) del dataset, por ejemplo:

```json
{
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
```

---

## 5) Simulador de cliente

Ejecuta:

```bash
python examples/client_simulator.py
```

Salida esperada (ejemplo):

- Cliente bajo riesgo → `APPROVE`, score bajo
- Caso internacional (p.ej. BIN ISO US) → score más alto y posible `REJECT`

---

## 6) Operación: triage por umbrales

La lógica de decisión usa dos thresholds:

- **T_low**: debajo → `APPROVE`
- **T_high**: encima → `REJECT`
- Entre ambos → `REVIEW`

Esto permite:
- Operar con riesgo controlado
- Enviar zona gris a revisión y **crear etiquetas reales** (menos proxy)
- **Aumentar dataset** y reducir sesgo/overfit iterativamente

> Recomendación v1: versionar thresholds en un artifact (p.ej. `src/artifacts/thresholds.json`) para evitar hardcode.

---

## 7) Preguntas de operación (incidentes y calidad)

### 7.1. El fraude aumenta de forma repentina. ¿Qué revisarías primero?

- Velocity Spikes: Revisar si los ataques se concentran en un nuevo BIN Bank o Account Id que no hayamos visto (Fraude de cuenta nueva).
- Feature Drift: Verificar si la distribución de las variables de entrada ha cambiado.
- Acción: Modificar los thresholds para pasar mas casos pro procesos de revisión.

---

### 7.2. El modelo funciona bien offline pero falla en producción. ¿Qué podría estar pasando?

Causas típicas:
- Probablemente una variable se calcula distinto en el script de entrenamiento que en la API.
- Data Leakage: El modelo podría haber usado variables del "futuro" durante el entrenamiento que no existen en el momento del pago.
- Latencia de Features: Si la variable de velocidad tarda en actualizarse en la base de datos de producción, el modelo recibe datos "viejos" y el score será erróneo.

---

### 7.3. Se están rechazando demasiadas transacciones legítimas. ¿Cómo lo abordarías?

- Calibración de Probabilidades: El modelo podría estar "sobre-excitado". Aplicar Isotonic Regression (aproximación no decreciente) para suavizar los scores.
- Ajuste de $T_{low}$: Aumentar el umbral de aprobación automática para permitir más volumen, aceptando un riesgo marginalmente mayor a cambio de mejor experiencia de usuario.
- Segmentación: Si el error ocurre en montos altos, crear una regla de excepción o una feature de "Monto relativo" para clientes recurrentes.

---

### 7.4. Las etiquetas de fraude llegan con retraso. ¿Cómo evaluarías el sistema?

- Proxy Metrics: Usar la tasa de rechazo del procesador de pagos y las discrepancias geográficas como señales tempranas.
- Active Learning: Utilizar los resultados de la revisión manual (que son inmediatos) como etiquetas temporales para medir la precisión del modelo en la "zona gris".

---

## 9) Licencia / Notas

Este repositorio implementa un pipeline de ML y un servicio de inferencia.  
Los datasets se gestionan fuera del contenedor (idealmente con DVC) y no se incluyen en la imagen.
