"""Central path/config definitions. Every other module imports paths from here
instead of recomputing them, so the project can be moved/renamed freely."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MERGED_CSV = PROCESSED_DATA_DIR / "final_dataset.csv"
CLEANED_CSV = PROCESSED_DATA_DIR / "final_dataset_clean_new.csv"

MODELS_DIR = ROOT_DIR / "models"
REGISTRY_FILE = MODELS_DIR / "registry.json"

FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"

SERVER_PORT = 8000

ATP_DATA_URL = "http://www.tennis-data.co.uk/alldata.php"
ATP_DATA_BASE = "http://www.tennis-data.co.uk/"

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
