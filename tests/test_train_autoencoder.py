"""Smoke tests for autoencoder training (requires TensorFlow)."""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

pytest.importorskip("tensorflow")

from generate_data import generate_dataframe  # noqa: E402
from preprocess import run_preprocess  # noqa: E402
from train_autoencoder import run_train  # noqa: E402


def test_run_autoencoder_smoke(tmp_path: Path):
    raw = tmp_path / "metrics.csv"
    generate_dataframe(days=3, seed=1).to_csv(raw, index=False)
    run_preprocess(
        input_path=raw,
        train_out=tmp_path / "train.csv",
        test_out=tmp_path / "test.csv",
        scaler_path=tmp_path / "scaler.pkl",
        test_size=0.25,
        seed=1,
    )
    run_train(
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        scaler_path=tmp_path / "scaler.pkl",
        model_out=tmp_path / "ae.keras",
        threshold_out=tmp_path / "threshold.json",
        metrics_dir=tmp_path / "metrics",
        predictions_dir=tmp_path / "pred",
        random_state=0,
        epochs=3,
        batch_size=64,
        mse_percentile=95.0,
        validation_fraction=0.15,
    )
    assert (tmp_path / "ae.keras").exists()
    assert (tmp_path / "threshold.json").exists()
    assert (tmp_path / "pred" / "autoencoder_test.csv").exists()
