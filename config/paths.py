import logging as log
from pathlib import Path

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log.info(f"Root project path: {PROJECT_ROOT}.")

DATA_DIR = PROJECT_ROOT / "src" / "data"
RAW_DIR = DATA_DIR  / "raw"
SPLITS_DIR = DATA_DIR  / "splits"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "src" / "artifacts"
MODELS_DIR = PROJECT_ROOT / "src" / "models"

RAW_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

log.info(f"📂 Raw data path: {RAW_DIR}.")
log.info(f"📂 Splits data path: {SPLITS_DIR}.")
log.info(f"📂 Processed data path: {PROCESSED_DIR}.")
log.info(f"📂 Artifacts data path: {ARTIFACTS_DIR}.")
log.info(f"📂 Models data path: {MODELS_DIR}.")

# raw data
RAW_CSV_PATH = RAW_DIR / "base_datos_fraude.csv"
