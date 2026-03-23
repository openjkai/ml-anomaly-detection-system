"""Smoke tests for synthetic data generation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from generate_data import generate_dataframe  # noqa: E402


def test_generate_dataframe_shape_and_columns():
    df = generate_dataframe(days=2, seed=0)
    assert len(df) == 2 * 24 * 12  # 5-minute steps
    expected = [
        "timestamp",
        "cpu_usage",
        "memory_usage",
        "request_latency_ms",
        "error_rate",
        "request_count",
        "disk_io",
        "network_in_mb",
        "is_anomaly",
    ]
    assert list(df.columns) == expected
    assert df["is_anomaly"].isin([0, 1]).all()


def test_has_some_anomalies():
    df = generate_dataframe(days=14, seed=123)
    assert df["is_anomaly"].sum() > 0
