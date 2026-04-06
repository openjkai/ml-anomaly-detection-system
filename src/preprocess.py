"""
Load raw metrics, clean, time-ordered train/test split, fit and save a scaler on train only.

Writes ``data/processed/train.csv``, ``data/processed/test.csv``, and ``models/scaler.pkl``.
Feature values in CSVs are **unscaled**; training scripts apply ``scaler.pkl`` to ``FEATURE_COLUMNS``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    DEFAULT_TEST_SIZE,
    FEATURE_COLUMNS,
    MODEL_SCALER,
    PROCESSED_TEST_CSV,
    PROCESSED_TRAIN_CSV,
    RANDOM_SEED,
    RAW_METRICS_CSV,
)
from features import FEATURE_COLUMNS_LIST, feature_matrix, validate_feature_columns
from utils import set_random_seed


def load_raw_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    validate_feature_columns(df)
    if "is_anomaly" not in df.columns:
        raise ValueError("Expected column is_anomaly")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time, drop duplicate timestamps, fill missing numeric values."""
    out = df.sort_values("timestamp").reset_index(drop=True)
    dup = out.duplicated(subset=["timestamp"], keep="first")
    if dup.any():
        out = out.loc[~dup].reset_index(drop=True)
    num_cols = [*FEATURE_COLUMNS, "is_anomaly"]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")
    out[FEATURE_COLUMNS_LIST] = out[FEATURE_COLUMNS_LIST].ffill().bfill()
    out = out.dropna(subset=FEATURE_COLUMNS_LIST)
    out["is_anomaly"] = out["is_anomaly"].fillna(0).astype(int)
    return out


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: first (1 - test_size) rows train, remainder test."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    n = len(df)
    if n < 3:
        raise ValueError("Need at least 3 rows for train/test split")
    split = int(n * (1.0 - test_size))
    split = max(1, min(split, n - 1))
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)
    return train_df, test_df


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(feature_matrix(train_df))
    return scaler


def run_preprocess(
    input_path: Path = RAW_METRICS_CSV,
    train_out: Path = PROCESSED_TRAIN_CSV,
    test_out: Path = PROCESSED_TEST_CSV,
    scaler_path: Path = MODEL_SCALER,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    set_random_seed(seed)
    raw = load_raw_metrics(input_path)
    cleaned = clean_dataframe(raw)
    train_df, test_df = temporal_train_test_split(cleaned, test_size=test_size)
    scaler = fit_scaler(train_df)

    train_out.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    joblib.dump(scaler, scaler_path)
    return train_df, test_df, scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw metrics CSV")
    parser.add_argument("--input", type=Path, default=RAW_METRICS_CSV)
    parser.add_argument("--train-out", type=Path, default=PROCESSED_TRAIN_CSV)
    parser.add_argument("--test-out", type=Path, default=PROCESSED_TEST_CSV)
    parser.add_argument("--scaler-out", type=Path, default=MODEL_SCALER)
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of newest rows for test (time-ordered)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    train_df, test_df, _ = run_preprocess(
        input_path=args.input,
        train_out=args.train_out,
        test_out=args.test_out,
        scaler_path=args.scaler_out,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"Train rows: {len(train_df)} -> {args.train_out}")
    print(f"Test rows:  {len(test_df)} -> {args.test_out}")
    print(f"Scaler:     {args.scaler_out}")


if __name__ == "__main__":
    main()
