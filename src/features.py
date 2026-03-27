"""Feature column spec and matrix extraction for metric columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical order for models, scaler, and CSV feature columns (see also config re-export).
FEATURE_COLUMNS: tuple[str, ...] = (
    "cpu_usage",
    "memory_usage",
    "request_latency_ms",
    "error_rate",
    "request_count",
    "disk_io",
    "network_in_mb",
)

N_FEATURES: int = len(FEATURE_COLUMNS)


def validate_feature_columns(df: pd.DataFrame) -> None:
    """Ensure all feature names exist (e.g. after loading CSV). Does not require numeric dtypes."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def assert_numeric_feature_columns(df: pd.DataFrame) -> None:
    """Require present feature columns with numeric dtypes (for model/scaler input)."""
    validate_feature_columns(df)
    bad: list[str] = []
    for c in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            bad.append(f"{c} ({df[c].dtype})")
    if bad:
        raise TypeError(
            "Feature columns must be numeric: " + ", ".join(bad),
        )


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return (n_samples, n_features) float array in ``FEATURE_COLUMNS`` order."""
    assert_numeric_feature_columns(df)
    # list() so pandas does not treat a tuple as a MultiIndex key
    return df.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64, copy=False)
