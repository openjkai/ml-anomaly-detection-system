"""Feature column spec and matrix extraction for metric columns."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class SupportsTransform(Protocol):
    """Fitted sklearn-style scaler or any object with ``transform(X) -> ndarray``."""

    def transform(self, X: np.ndarray) -> np.ndarray: ...


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

# Short labels for plots, logs, and APIs (keys match ``FEATURE_COLUMNS``).
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "cpu_usage": "CPU usage (%)",
    "memory_usage": "Memory (%)",
    "request_latency_ms": "Latency (ms)",
    "error_rate": "Error rate",
    "request_count": "Request count",
    "disk_io": "Disk I/O",
    "network_in_mb": "Network in (MB)",
}


def feature_display_name(column: str) -> str:
    """Human-readable label for a feature column, or ``column`` if unknown."""
    return FEATURE_DISPLAY_NAMES.get(column, column)


__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_COLUMNS_LIST",
    "FEATURE_DISPLAY_NAMES",
    "N_FEATURES",
    "feature_display_name",
    "SupportsTransform",
    "assert_finite_feature_array",
    "assert_numeric_feature_columns",
    "feature_frame",
    "feature_matrix",
    "read_metrics_csv",
    "read_train_test_csv",
    "scaled_feature_matrix",
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


def scaled_feature_matrix(
    df: pd.DataFrame,
    scaler: SupportsTransform,
    *,
    dtype: type | np.dtype = np.float64,
) -> np.ndarray:
    """``scaler.transform(feature_matrix(df))`` with optional output dtype (e.g. float32 for TF)."""
    X = scaler.transform(feature_matrix(df))
    if np.dtype(dtype) == np.dtype(np.float64):
        return X
    return X.astype(dtype, copy=False)


def _parse_timestamp_if_present(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def read_train_test_csv(
    train_path: Path, test_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train/test CSVs and validate feature columns."""
    train = _parse_timestamp_if_present(pd.read_csv(train_path))
    test = _parse_timestamp_if_present(pd.read_csv(test_path))
    validate_feature_columns(train)
    validate_feature_columns(test)
    return train, test


def read_metrics_csv(path: Path, *, expect_labels: bool = True) -> pd.DataFrame:
    """Load a metrics CSV (train, test, or raw-shaped frame) with feature validation."""
    df = _parse_timestamp_if_present(pd.read_csv(path))
    validate_feature_columns(df)
    if expect_labels and "is_anomaly" not in df.columns:
        raise ValueError("Expected column is_anomaly")
    return df
