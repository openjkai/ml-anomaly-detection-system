"""Shared scoring helpers for Isolation Forest and autoencoder (inference and evaluation)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from tensorflow import keras


def load_ae_threshold(path: Path) -> float:
    """Load MSE threshold from ``autoencoder_threshold.json``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["threshold"])


def score_points(
    model: IsolationForest, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (anomaly_score, is_outlier).

    ``anomaly_score`` increases when the point is more anomalous (negated ``score_samples``).
    ``is_outlier`` is 1 if the model flags an anomaly, else 0.
    """
    raw = model.score_samples(X)
    anomaly_score = -raw
    pred = model.predict(X)
    is_outlier = (pred == -1).astype(np.int8)
    return anomaly_score, is_outlier


def reconstruction_mse(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Per-row mean squared reconstruction error."""
    pred = model.predict(X, verbose=0)
    return np.mean((X - pred) ** 2, axis=1)
