"""
Train a baseline Isolation Forest on scaled training features (normal rows only when available).

Saves ``models/isolation_forest.pkl`` and test-set scores under ``outputs/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import (
    MODEL_ISOLATION_FOREST,
    MODEL_SCALER,
    OUTPUTS_METRICS_DIR,
    OUTPUTS_PREDICTIONS_DIR,
    PROCESSED_TEST_CSV,
    PROCESSED_TRAIN_CSV,
    RANDOM_SEED,
)
from features import read_train_test_csv, scaled_feature_matrix
from scoring import score_points
from utils import set_random_seed


def contamination_from_labels(train_df: pd.DataFrame) -> float:
    """Use observed anomaly rate on train, clamped for IsolationForest."""
    rate = float(train_df["is_anomaly"].mean())
    if rate <= 0.0:
        return 0.05
    return float(min(0.45, max(rate, 0.001)))


def train_isolation_forest(
    X: np.ndarray,
    contamination: float,
    random_state: int,
    n_estimators: int = 200,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def run_train(
    train_path: Path = PROCESSED_TRAIN_CSV,
    test_path: Path = PROCESSED_TEST_CSV,
    scaler_path: Path = MODEL_SCALER,
    model_out: Path = MODEL_ISOLATION_FOREST,
    metrics_dir: Path = OUTPUTS_METRICS_DIR,
    predictions_dir: Path = OUTPUTS_PREDICTIONS_DIR,
    random_state: int = RANDOM_SEED,
    n_estimators: int = 200,
) -> IsolationForest:
    set_random_seed(random_state)
    train_df, test_df = read_train_test_csv(train_path, test_path)
    scaler = joblib.load(scaler_path)

    normal_mask = train_df["is_anomaly"].to_numpy() == 0
    if normal_mask.sum() >= 50:
        fit_df = train_df.loc[normal_mask]
    else:
        fit_df = train_df

    X_fit = scaled_feature_matrix(fit_df, scaler)
    contamination = contamination_from_labels(train_df)

    model = train_isolation_forest(
        X_fit,
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    X_test = scaled_feature_matrix(test_df, scaler)
    scores, flags = score_points(model, X_test)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "isolation_forest_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"train_rows={len(train_df)}",
                f"fit_rows={len(fit_df)}",
                f"contamination={contamination:.6f}",
                f"n_estimators={n_estimators}",
                f"test_rows={len(test_df)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_df = test_df[["timestamp", "is_anomaly"]].copy()
    out_df["anomaly_score"] = scores
    out_df["pred_anomaly"] = flags
    pred_csv = predictions_dir / "isolation_forest_test.csv"
    out_df.to_csv(pred_csv, index=False)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Isolation Forest baseline")
    parser.add_argument("--train", type=Path, default=PROCESSED_TRAIN_CSV)
    parser.add_argument("--test", type=Path, default=PROCESSED_TEST_CSV)
    parser.add_argument("--scaler", type=Path, default=MODEL_SCALER)
    parser.add_argument("--output", type=Path, default=MODEL_ISOLATION_FOREST)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    run_train(
        train_path=args.train,
        test_path=args.test,
        scaler_path=args.scaler,
        model_out=args.output,
        random_state=args.seed,
        n_estimators=args.n_estimators,
    )
    print(f"Saved model to {args.output}")
    print(f"Wrote {OUTPUTS_METRICS_DIR / 'isolation_forest_summary.txt'}")
    print(f"Wrote {OUTPUTS_PREDICTIONS_DIR / 'isolation_forest_test.csv'}")


if __name__ == "__main__":
    main()
