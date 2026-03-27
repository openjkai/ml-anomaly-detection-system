"""Project paths and default constants."""

from pathlib import Path

import features as _features

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Re-export for callers using ``from config import FEATURE_COLUMNS``.
FEATURE_COLUMNS = _features.FEATURE_COLUMNS

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
OUTPUTS_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
OUTPUTS_PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

RAW_METRICS_CSV = DATA_RAW_DIR / "metrics.csv"
PROCESSED_TRAIN_CSV = DATA_PROCESSED_DIR / "train.csv"
PROCESSED_TEST_CSV = DATA_PROCESSED_DIR / "test.csv"

MODEL_ISOLATION_FOREST = MODELS_DIR / "isolation_forest.pkl"
MODEL_AUTOENCODER = MODELS_DIR / "autoencoder.keras"
MODEL_AUTOENCODER_THRESHOLD = MODELS_DIR / "autoencoder_threshold.json"
MODEL_SCALER = MODELS_DIR / "scaler.pkl"

RANDOM_SEED = 42
DEFAULT_TEST_SIZE = 0.2
