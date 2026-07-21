from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "T1.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"

DATETIME_COL = "Date/Time"
TARGET_COL = "LV ActivePower (kW)"

TRAIN_RATIO = 0.8
RESAMPLE_FREQ = "10min"
FORECAST_HORIZON = 48

RAMP_THRESHOLD = 0.15