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