"""Smoke tests for batch inference."""

import sys
from pathlib import Path

import joblib
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from generate_data import generate_dataframe  # noqa: E402
from preprocess import run_preprocess  # noqa: E402
from train_autoencoder import run_train as run_train_ae  # noqa: E402
from train_isolation_forest import run_train as run_train_if  # noqa: E402


def test_predict_smoke(tmp_path: Path):
    raw = tmp_path / "metrics.csv"
    generate_dataframe(days=3, seed=2).to_csv(raw, index=False)
    run_preprocess(
        input_path=raw,
        train_out=tmp_path / "train.csv",
        test_out=tmp_path / "test.csv",
        scaler_path=tmp_path / "scaler.pkl",
        test_size=0.25,
        seed=2,
    )
    run_train_if(
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        scaler_path=tmp_path / "scaler.pkl",
        model_out=tmp_path / "if.pkl",
        metrics_dir=tmp_path / "m1",
        predictions_dir=tmp_path / "p1",
        n_estimators=30,
        random_state=2,
    )
    run_train_ae(
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        scaler_path=tmp_path / "scaler.pkl",
        model_out=tmp_path / "ae.keras",
        threshold_out=tmp_path / "th.json",
        metrics_dir=tmp_path / "m2",
        predictions_dir=tmp_path / "p2",
        random_state=2,
        epochs=2,
        batch_size=64,
    )

    from predict import load_predictors, run_predict_csv  # noqa: E402
    from scoring import combined_anomaly_alert  # noqa: E402

    bundle = load_predictors(
        scaler_path=tmp_path / "scaler.pkl",
        isolation_forest_path=tmp_path / "if.pkl",
        autoencoder_path=tmp_path / "ae.keras",
        threshold_path=tmp_path / "th.json",
    )
    out_path = tmp_path / "out.csv"
    df = run_predict_csv(
        tmp_path / "test.csv",
        out_path,
        bundle,
        expect_labels=True,
    )
    assert out_path.exists()
    assert "if_score" in df.columns and "ae_mse" in df.columns
    assert "anomaly_alert" in df.columns
    assert np.array_equal(
        df["anomaly_alert"].to_numpy(),
        combined_anomaly_alert(
            df["if_pred"].to_numpy(),
            df["ae_pred"].to_numpy(),
        ),
    )
    joblib.load(tmp_path / "if.pkl")
