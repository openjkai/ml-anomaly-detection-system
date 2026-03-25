"""Tests for preprocessing pipeline."""

import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from generate_data import generate_dataframe  # noqa: E402
from preprocess import (  # noqa: E402
    clean_dataframe,
    run_preprocess,
    temporal_train_test_split,
)


def test_temporal_split_order():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
            "cpu_usage": range(10),
            "memory_usage": range(10),
            "request_latency_ms": range(10),
            "error_rate": [0.01] * 10,
            "request_count": [100] * 10,
            "disk_io": [20] * 10,
            "network_in_mb": [100] * 10,
            "is_anomaly": [0] * 10,
        }
    )
    train, test = temporal_train_test_split(df, test_size=0.2)
    assert len(train) == 8 and len(test) == 2
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_clean_drops_duplicate_timestamps():
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 01:00:00",
            ],
            "cpu_usage": [1.0, 99.0, 2.0],
            "memory_usage": [50.0, 50.0, 51.0],
            "request_latency_ms": [100.0, 900.0, 101.0],
            "error_rate": [0.01, 0.5, 0.01],
            "request_count": [100.0, 10.0, 100.0],
            "disk_io": [20.0, 20.0, 21.0],
            "network_in_mb": [100.0, 100.0, 101.0],
            "is_anomaly": [0, 1, 0],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cleaned = clean_dataframe(df)
    assert len(cleaned) == 2
    assert cleaned.iloc[0]["cpu_usage"] == 1.0


def test_run_preprocess_end_to_end(tmp_path: Path):
    raw = tmp_path / "raw.csv"
    generate_dataframe(days=5, seed=7).to_csv(raw, index=False)
    train, test, scaler = run_preprocess(
        input_path=raw,
        train_out=tmp_path / "train.csv",
        test_out=tmp_path / "test.csv",
        scaler_path=tmp_path / "scaler.pkl",
        test_size=0.2,
        seed=42,
    )
    assert len(train) + len(test) == 5 * 24 * 12
    assert scaler.mean_.shape == (7,)
    loaded = joblib.load(tmp_path / "scaler.pkl")
    assert loaded.mean_.shape == (7,)
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_invalid_test_size():
    df = generate_dataframe(days=1, seed=0)
    with pytest.raises(ValueError):
        temporal_train_test_split(df, test_size=0.0)
