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

# Pandas treats a tuple of labels as a MultiIndex key; use a list for normal columns.
FEATURE_COLUMNS_LIST: list[str] = list(FEATURE_COLUMNS)

__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_COLUMNS_LIST",
    "N_FEATURES",
    "assert_finite_feature_array",
    "assert_numeric_feature_columns",
    "feature_frame",
    "feature_matrix",
    "validate_feature_columns",
]


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


def assert_finite_feature_array(X: np.ndarray) -> None:
    """Reject NaN/inf in a feature matrix before scaling or model forward passes."""
    if X.size and not np.isfinite(X).all():
        raise ValueError("Feature matrix contains non-finite values (NaN or inf)")


def feature_frame(df: pd.DataFrame, *, copy: bool = False) -> pd.DataFrame:
    """Return only feature columns in ``FEATURE_COLUMNS`` order (for inspection or export)."""
    assert_numeric_feature_columns(df)
    out = df.loc[:, FEATURE_COLUMNS_LIST]
    return out.copy() if copy else out


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return (n_samples, n_features) float array in ``FEATURE_COLUMNS`` order."""
    assert_numeric_feature_columns(df)
    X = df.loc[:, FEATURE_COLUMNS_LIST].to_numpy(dtype=np.float64, copy=False)
    assert_finite_feature_array(X)
    return X
