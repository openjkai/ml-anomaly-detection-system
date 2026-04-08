"""Batch inference: load scaler, Isolation Forest, autoencoder, and score a metrics CSV."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest

from config import (
    MODEL_AUTOENCODER,
    MODEL_AUTOENCODER_THRESHOLD,
    MODEL_ISOLATION_FOREST,
    MODEL_SCALER,
    OUTPUTS_PREDICTIONS_DIR,
    PROCESSED_TEST_CSV,
)
from features import (
    SupportsTransform,
    assert_numeric_feature_columns,
    read_metrics_csv,
    scaled_feature_matrix,
)
from scoring import load_ae_threshold, reconstruction_mse, score_points


@dataclass(frozen=True)
class PredictorBundle:
    scaler: SupportsTransform
    if_model: IsolationForest
    ae_model: tf.keras.Model
    ae_threshold: float


def load_predictors(
    scaler_path: Path = MODEL_SCALER,
    isolation_forest_path: Path = MODEL_ISOLATION_FOREST,
    autoencoder_path: Path = MODEL_AUTOENCODER,
    threshold_path: Path = MODEL_AUTOENCODER_THRESHOLD,
) -> PredictorBundle:
    try:
        tf.keras.utils.disable_interactive_logging()
    except AttributeError:
        pass
    scaler = joblib.load(scaler_path)
    if_model = joblib.load(isolation_forest_path)
    ae_model = tf.keras.models.load_model(autoencoder_path)
    threshold = load_ae_threshold(threshold_path)
    return PredictorBundle(
        scaler=scaler,
        if_model=if_model,
        ae_model=ae_model,
        ae_threshold=threshold,
    )


def predict_dataframe(df: pd.DataFrame, bundle: PredictorBundle) -> pd.DataFrame:
    """Append IF/AE scores and binary predictions (copies ``df``)."""
    assert_numeric_feature_columns(df)
    X = scaled_feature_matrix(df, bundle.scaler, dtype=np.float32)
    if_scores, if_flags = score_points(bundle.if_model, X)
    ae_scores = reconstruction_mse(bundle.ae_model, X)
    ae_flags = (ae_scores > bundle.ae_threshold).astype(np.int8)
    out = df.copy()
    out["if_score"] = if_scores
    out["if_pred"] = if_flags
    out["ae_mse"] = ae_scores
    out["ae_pred"] = ae_flags
    return out


def run_predict_csv(
    input_path: Path,
    output_path: Path,
    bundle: PredictorBundle,
    *,
    expect_labels: bool = True,
) -> pd.DataFrame:
    df = read_metrics_csv(input_path, expect_labels=expect_labels)
    out = predict_dataframe(df, bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score metrics CSV with trained Isolation Forest and autoencoder",
    )
    parser.add_argument("--input", type=Path, default=PROCESSED_TEST_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_PREDICTIONS_DIR / "batch_predictions.csv",
    )
    parser.add_argument("--scaler", type=Path, default=MODEL_SCALER)
    parser.add_argument("--isolation-forest", type=Path, default=MODEL_ISOLATION_FOREST)
    parser.add_argument("--autoencoder", type=Path, default=MODEL_AUTOENCODER)
    parser.add_argument("--threshold", type=Path, default=MODEL_AUTOENCODER_THRESHOLD)
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="CSV has no is_anomaly column (inference-only)",
    )
    args = parser.parse_args()

    bundle = load_predictors(
        scaler_path=args.scaler,
        isolation_forest_path=args.isolation_forest,
        autoencoder_path=args.autoencoder,
        threshold_path=args.threshold,
    )
    run_predict_csv(
        args.input,
        args.output,
        bundle,
        expect_labels=not args.no_labels,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
