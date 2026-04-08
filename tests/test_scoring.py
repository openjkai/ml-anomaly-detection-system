"""Tests for shared scoring helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from scoring import load_ae_threshold, score_points  # noqa: E402


def test_score_points_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 7))
    model = IsolationForest(n_estimators=20, random_state=0, contamination=0.1)
    model.fit(X)
    scores, flags = score_points(model, X)
    assert scores.shape == (40,) and flags.shape == (40,)
    assert np.unique(flags).tolist() in ([0], [1], [0, 1])


def test_load_ae_threshold(tmp_path: Path):
    p = tmp_path / "t.json"
    p.write_text('{"threshold": 0.42, "n_features": 7}', encoding="utf-8")
    assert load_ae_threshold(p) == pytest.approx(0.42)
