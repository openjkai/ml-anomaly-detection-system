"""
Train a dense autoencoder on **normal-only** scaled training rows; anomaly score = per-row MSE reconstruction error.

Saves ``models/autoencoder.keras``, ``models/autoencoder_threshold.json`` (95th percentile of train errors),
and ``outputs/predictions/autoencoder_test.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    MODEL_AUTOENCODER,
    MODEL_AUTOENCODER_THRESHOLD,
    MODEL_SCALER,
    OUTPUTS_METRICS_DIR,
    OUTPUTS_PREDICTIONS_DIR,
    PROCESSED_TEST_CSV,
    PROCESSED_TRAIN_CSV,
    RANDOM_SEED,
)
from features import feature_matrix, validate_feature_columns
from utils import set_random_seed


def load_processed(
    train_path: Path, test_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path, parse_dates=["timestamp"])
    test = pd.read_csv(test_path, parse_dates=["timestamp"])
    validate_feature_columns(train)
    validate_feature_columns(test)
    return train, test


def build_autoencoder(input_dim: int) -> keras.Model:
    """Dense 7→32→16→8→16→32→7 as in project spec."""
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(32, activation="relu")(inp)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(input_dim, activation="linear")(x)
    model = keras.Model(inp, out, name="metrics_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def reconstruction_mse(model: keras.Model, X: np.ndarray) -> np.ndarray:
    pred = model.predict(X, verbose=0)
    return np.mean((X - pred) ** 2, axis=1)


def run_train(
    train_path: Path = PROCESSED_TRAIN_CSV,
    test_path: Path = PROCESSED_TEST_CSV,
    scaler_path: Path = MODEL_SCALER,
    model_out: Path = MODEL_AUTOENCODER,
    threshold_out: Path = MODEL_AUTOENCODER_THRESHOLD,
    metrics_dir: Path = OUTPUTS_METRICS_DIR,
    predictions_dir: Path = OUTPUTS_PREDICTIONS_DIR,
    random_state: int = RANDOM_SEED,
    epochs: int = 100,
    batch_size: int = 32,
    mse_percentile: float = 95.0,
    validation_fraction: float = 0.1,
) -> keras.Model:
    set_random_seed(random_state)
    tf.random.set_seed(random_state)

    train_df, test_df = load_processed(train_path, test_path)
    scaler = joblib.load(scaler_path)

    normal_mask = train_df["is_anomaly"].to_numpy() == 0
    fit_df = train_df.loc[normal_mask] if normal_mask.sum() >= 32 else train_df
    X_fit = scaler.transform(feature_matrix(fit_df)).astype(np.float32)

    n_features = X_fit.shape[1]
    model = build_autoencoder(n_features)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_fit,
        X_fit,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_fraction,
        callbacks=callbacks,
        verbose=1,
    )

    train_errors = reconstruction_mse(model, X_fit)
    threshold = float(np.percentile(train_errors, mse_percentile))

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)

    threshold_payload = {
        "mse_percentile": mse_percentile,
        "threshold": threshold,
        "n_train_fit_rows": int(len(fit_df)),
        "n_features": int(n_features),
    }
    threshold_out.write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")

    X_test = scaler.transform(feature_matrix(test_df)).astype(np.float32)
    test_errors = reconstruction_mse(model, X_test)
    pred_flag = (test_errors > threshold).astype(np.int8)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "autoencoder_summary.txt").write_text(
        "\n".join(
            [
                f"train_fit_rows={len(fit_df)}",
                f"epochs_requested={epochs}",
                f"mse_percentile={mse_percentile}",
                f"threshold={threshold:.8f}",
                f"test_rows={len(test_df)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_df = test_df[["timestamp", "is_anomaly"]].copy()
    out_df["reconstruction_mse"] = test_errors
    out_df["pred_anomaly"] = pred_flag
    out_df.to_csv(predictions_dir / "autoencoder_test.csv", index=False)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train dense autoencoder for anomaly detection"
    )
    parser.add_argument("--train", type=Path, default=PROCESSED_TRAIN_CSV)
    parser.add_argument("--test", type=Path, default=PROCESSED_TEST_CSV)
    parser.add_argument("--scaler", type=Path, default=MODEL_SCALER)
    parser.add_argument("--output", type=Path, default=MODEL_AUTOENCODER)
    parser.add_argument(
        "--threshold-out", type=Path, default=MODEL_AUTOENCODER_THRESHOLD
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mse-percentile", type=float, default=95.0)
    args = parser.parse_args()

    run_train(
        train_path=args.train,
        test_path=args.test,
        scaler_path=args.scaler,
        model_out=args.output,
        threshold_out=args.threshold_out,
        random_state=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mse_percentile=args.mse_percentile,
    )
    print(f"Saved model to {args.output}")
    print(f"Saved threshold to {args.threshold_out}")


if __name__ == "__main__":
    main()
