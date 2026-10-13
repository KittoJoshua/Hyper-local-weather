"""
Configuration settings for the Micro-Climate Machine Learning Engine.
Defines parameters for data ingestion, feature engineering, models (LSTM, XGBoost, Hybrid),
and walk-forward validation.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ML_ENGINE_DIR = BASE_DIR / "ml_engine"
WEIGHTS_DIR = ML_ENGINE_DIR / "weights"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIXTURES_DIR = DATA_DIR / "fixtures"

# Ensure directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

# Target Forecast Variables
TARGET_VARIABLES = [
    "temperature_c",
    "humidity_percent",
    "precipitation_mm"
]

# Forecast Horizons
HOURLY_HORIZON_STEPS = 72  # 72 hours (3 days)
DAILY_HORIZON_STEPS = 7    # 7 days

# Historical Sequence Length for LSTM (e.g. past 14 days of hourly observations)
SEQUENCE_LENGTH = 336      # 14 days * 24 hours

# Feature Engineering Lags (in hours)
LAG_STEPS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLLING_WINDOWS = [6, 12, 24, 72]

# Walk-Forward Validation Configuration
VALIDATION_SPLIT = 0.20
TEST_SPLIT = 0.15
N_SPLITS = 4               # Expanding window folds
DEFAULT_TRAINING_RECORDS = 8760  # 1 full year (365 days * 24h) for fast, responsive training

# Model Hyperparameters
LSTM_PARAMS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 30,
}

XGBOOST_PARAMS = {
    "n_estimators": 250,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# Agronomic Alert Thresholds (Kenya agro-climatic zones)
AGRONOMIC_THRESHOLDS = {
    "fertilizer_washout_rain_mm": 20.0,    # > 20mm rain in 24-48h washes out top-dressed fertilizer
    "seed_desiccation_temp_c": 30.0,       # High heat > 30C with low soil moisture kills germinating seeds
    "seed_desiccation_rain_threshold_mm": 1.0,
    "fungal_rot_humidity_percent": 82.0,   # Relative humidity > 82% during harvest promotes fungal growth
    "dry_spell_days": 7                    # 7 consecutive days with < 1mm rainfall during vegetative stage
}
