"""Shared helpers."""

from __future__ import annotations

import numpy as np


def set_random_seed(seed: int) -> None:
    """Set NumPy global RNG seed for reproducible synthetic data and splits."""
    np.random.seed(seed)
