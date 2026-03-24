"""Feature matrix helpers for metric columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS


def validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return (n_samples, n_features) float array in ``FEATURE_COLUMNS`` order."""
    validate_feature_columns(df)
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
