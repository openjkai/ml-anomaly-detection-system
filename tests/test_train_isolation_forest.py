"""Smoke tests for Isolation Forest training."""

import sys
from pathlib import Path

import joblib

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from generate_data import generate_dataframe  # noqa: E402
from preprocess import run_preprocess  # noqa: E402
from train_isolation_forest import run_train  # noqa: E402


def test_run_train_smoke(tmp_path: Path):
    raw = tmp_path / "metrics.csv"
    generate_dataframe(days=3, seed=0).to_csv(raw, index=False)
    run_preprocess(
        input_path=raw,
        train_out=tmp_path / "train.csv",
        test_out=tmp_path / "test.csv",
        scaler_path=tmp_path / "scaler.pkl",
        test_size=0.25,
        seed=0,
    )
    run_train(
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        scaler_path=tmp_path / "scaler.pkl",
        model_out=tmp_path / "if.pkl",
        metrics_dir=tmp_path / "metrics",
        predictions_dir=tmp_path / "pred",
        n_estimators=50,
        random_state=0,
    )
    assert (tmp_path / "if.pkl").exists()
    joblib.load(tmp_path / "if.pkl")
    assert (tmp_path / "pred" / "isolation_forest_test.csv").exists()
