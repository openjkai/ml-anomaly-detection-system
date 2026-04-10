"""Tests for feature column helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from features import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_LIST,
    FEATURE_DISPLAY_NAMES,
    N_FEATURES,
    assert_finite_feature_array,
    assert_numeric_feature_columns,
    feature_display_name,
    feature_frame,
    feature_matrix,
    feature_metadata,
    read_metrics_csv,
    read_train_test_csv,
    scaled_feature_matrix,
    validate_feature_columns,
)


def _minimal_numeric_row():
    return {
        "cpu_usage": 1.0,
        "memory_usage": 2.0,
        "request_latency_ms": 3.0,
        "error_rate": 0.01,
        "request_count": 100.0,
        "disk_io": 10.0,
        "network_in_mb": 20.0,
    }


def test_feature_constants():
    assert N_FEATURES == len(FEATURE_COLUMNS) == 7
    assert FEATURE_COLUMNS_LIST == list(FEATURE_COLUMNS)
    assert list(FEATURE_COLUMNS)[0] == "cpu_usage"
    assert set(FEATURE_DISPLAY_NAMES.keys()) == set(FEATURE_COLUMNS)
    assert feature_display_name("cpu_usage") == "CPU usage (%)"
    assert feature_display_name("unknown_col") == "unknown_col"


def test_feature_metadata_order_and_labels():
    meta = feature_metadata()
    assert len(meta) == N_FEATURES
    names = [m["name"] for m in meta]
    assert names == list(FEATURE_COLUMNS)
    assert all(m["display_name"] == FEATURE_DISPLAY_NAMES[m["name"]] for m in meta)


def test_validate_feature_columns_ok():
    df = pd.DataFrame([_minimal_numeric_row()])
    validate_feature_columns(df)


def test_validate_feature_columns_missing():
    row = _minimal_numeric_row()
    del row["cpu_usage"]
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="Missing feature columns"):
        validate_feature_columns(df)


def test_feature_matrix_order_and_shape():
    row = _minimal_numeric_row()
    df = pd.DataFrame([row])
    X = feature_matrix(df)
    assert X.shape == (1, N_FEATURES)
    assert np.allclose(X[0], [row[c] for c in FEATURE_COLUMNS])


def test_assert_numeric_rejects_non_numeric():
    row = _minimal_numeric_row()
    row["cpu_usage"] = "not-a-number"
    df = pd.DataFrame([row])
    validate_feature_columns(df)
    with pytest.raises(TypeError, match="numeric"):
        assert_numeric_feature_columns(df)


def test_feature_frame_matches_matrix():
    row = _minimal_numeric_row()
    df = pd.DataFrame([row])
    fr = feature_frame(df)
    assert list(fr.columns) == list(FEATURE_COLUMNS)
    assert np.allclose(feature_matrix(df), fr.to_numpy(dtype=np.float64))


def test_feature_matrix_rejects_non_finite():
    row = _minimal_numeric_row()
    row["cpu_usage"] = float("nan")
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="non-finite"):
        feature_matrix(df)


def test_assert_finite_feature_array():
    assert_finite_feature_array(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        assert_finite_feature_array(np.array([[np.nan]]))


def test_read_train_test_csv(tmp_path: Path):
    row = _minimal_numeric_row()
    train = pd.DataFrame([row, row])
    test = pd.DataFrame([row])
    tp = tmp_path / "train.csv"
    tsp = tmp_path / "test.csv"
    train.to_csv(tp, index=False)
    test.to_csv(tsp, index=False)
    a, b = read_train_test_csv(tp, tsp)
    assert len(a) == 2 and len(b) == 1


def test_read_metrics_csv_requires_labels(tmp_path: Path):
    row = _minimal_numeric_row()
    p = tmp_path / "m.csv"
    pd.DataFrame([row]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="is_anomaly"):
        read_metrics_csv(p)
    df = pd.DataFrame([{**row, "is_anomaly": 0}])
    df.to_csv(p, index=False)
    assert len(read_metrics_csv(p)) == 1


def test_scaled_feature_matrix_dtype():
    row = _minimal_numeric_row()
    df = pd.DataFrame([row])
    scaler = StandardScaler()
    scaler.fit(feature_matrix(df))
    X64 = scaled_feature_matrix(df, scaler)
    X32 = scaled_feature_matrix(df, scaler, dtype=np.float32)
    assert X64.dtype == np.float64
    assert X32.dtype == np.float32
