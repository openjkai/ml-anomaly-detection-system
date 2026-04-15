"""Integration tests for the FastAPI scoring service."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient  # noqa: E402

from api import MetricRow, create_app  # noqa: E402
from config import API_VERSION  # noqa: E402
from features import FEATURE_COLUMNS  # noqa: E402
from generate_data import generate_dataframe  # noqa: E402
from predict import PREDICTION_SCORE_COLUMNS, load_predictors  # noqa: E402
from preprocess import run_preprocess  # noqa: E402
from train_autoencoder import run_train as run_train_ae  # noqa: E402
from train_isolation_forest import run_train as run_train_if  # noqa: E402


def _trained_bundle(tmp_path: Path):
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
    return load_predictors(
        scaler_path=tmp_path / "scaler.pkl",
        isolation_forest_path=tmp_path / "if.pkl",
        autoencoder_path=tmp_path / "ae.keras",
        threshold_path=tmp_path / "th.json",
    )


def test_metric_row_matches_feature_columns():
    assert list(MetricRow.model_fields.keys()) == list(FEATURE_COLUMNS)


def test_root_returns_discovery_json(tmp_path: Path):
    bundle = _trained_bundle(tmp_path)
    app = create_app(bundle=bundle)
    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "anomaly-detection"
    assert body["version"] == API_VERSION
    assert body["health"] == "/health"
    assert body["predict"] == "/predict"


def test_health_includes_feature_display_names(tmp_path: Path):
    bundle = _trained_bundle(tmp_path)
    app = create_app(bundle=bundle)
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == API_VERSION
    assert body["prediction_columns"] == list(PREDICTION_SCORE_COLUMNS)
    assert "cpu_usage" in body["feature_display_names"]
    assert body["feature_display_names"]["cpu_usage"] == "CPU usage (%)"


def test_predict_batch_returns_scores(tmp_path: Path):
    bundle = _trained_bundle(tmp_path)
    app = create_app(bundle=bundle)
    payload = {
        "rows": [
            {
                "cpu_usage": 50.0,
                "memory_usage": 60.0,
                "request_latency_ms": 100.0,
                "error_rate": 0.01,
                "request_count": 1000.0,
                "disk_io": 10.0,
                "network_in_mb": 5.0,
            }
        ]
    }
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert "if_score" in row and "ae_mse" in row
    assert "if_pred" in row and "ae_pred" in row
    assert "anomaly_alert" in row
    assert all(c in row for c in PREDICTION_SCORE_COLUMNS)
