"""
Evaluate Isolation Forest and autoencoder on the held-out test split using injected labels.

Writes text reports under ``outputs/metrics/`` and figures under ``outputs/plots/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import (
    MODEL_AUTOENCODER,
    MODEL_AUTOENCODER_THRESHOLD,
    MODEL_ISOLATION_FOREST,
    MODEL_SCALER,
    OUTPUTS_METRICS_DIR,
    OUTPUTS_PLOTS_DIR,
    PROCESSED_TEST_CSV,
)
from features import read_metrics_csv, scaled_feature_matrix
from scoring import (
    combined_anomaly_alert,
    load_ae_threshold,
    reconstruction_mse,
    score_points,
)


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["pred 0", "pred 1"],
        yticklabels=["true 0", "true 1"],
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timeline(
    df: pd.DataFrame,
    score_col: str,
    flag_col: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(
        df["timestamp"],
        df[score_col],
        color="steelblue",
        linewidth=0.8,
        label=score_col,
    )
    hits = df[df[flag_col] == 1]
    if not hits.empty:
        ax.scatter(
            hits["timestamp"],
            hits[score_col],
            color="crimson",
            s=8,
            label="pred anomaly",
        )
    true_hits = df[df["is_anomaly"] == 1]
    if not true_hits.empty:
        ax.scatter(
            true_hits["timestamp"],
            true_hits[score_col],
            facecolors="none",
            edgecolors="orange",
            s=36,
            linewidths=0.8,
            label="true anomaly",
        )
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_hist(
    scores: np.ndarray, labels: np.ndarray, title: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(scores[labels == 0], bins=40, alpha=0.7, label="normal", color="steelblue")
    ax.hist(scores[labels == 1], bins=40, alpha=0.7, label="anomaly", color="crimson")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    metrics = {"precision": float(p), "recall": float(r), "f1": float(f1)}
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    metrics["model"] = name
    return metrics


def run_evaluate(
    test_path: Path = PROCESSED_TEST_CSV,
    scaler_path: Path = MODEL_SCALER,
    if_path: Path = MODEL_ISOLATION_FOREST,
    ae_path: Path = MODEL_AUTOENCODER,
    threshold_path: Path = MODEL_AUTOENCODER_THRESHOLD,
    metrics_dir: Path = OUTPUTS_METRICS_DIR,
    plots_dir: Path = OUTPUTS_PLOTS_DIR,
) -> None:
    try:
        tf.keras.utils.disable_interactive_logging()
    except AttributeError:
        pass

    test_df = read_metrics_csv(test_path)
    y_true = test_df["is_anomaly"].to_numpy(dtype=int)

    scaler = joblib.load(scaler_path)
    X_test = scaled_feature_matrix(test_df, scaler, dtype=np.float32)

    if_model: IsolationForest = joblib.load(if_path)
    if_scores, if_flags = score_points(if_model, X_test)

    ae_model = tf.keras.models.load_model(ae_path)
    threshold = load_ae_threshold(threshold_path)
    ae_scores = reconstruction_mse(ae_model, X_test)
    ae_flags = (ae_scores > threshold).astype(np.int8)
    alert_flags = combined_anomaly_alert(if_flags, ae_flags)

    lines: list[str] = []
    for name, y_pred in (
        ("isolation_forest", if_flags),
        ("autoencoder", ae_flags),
        ("combined_or", alert_flags),
    ):
        lines.append(f"=== {name} ===")
        lines.append(classification_report(y_true, y_pred, digits=4))
        lines.append("")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "classification_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    summary_lines = []
    for name, y_pred, scores in (
        ("isolation_forest", if_flags, if_scores),
        ("autoencoder", ae_flags, ae_scores),
        ("combined_or", alert_flags, None),
    ):
        m = evaluate_binary(y_true, y_pred, name, y_score=scores)
        summary_lines.append(
            f"{name}: precision={m['precision']:.4f} recall={m['recall']:.4f} "
            f"f1={m['f1']:.4f} roc_auc={m['roc_auc']}"
        )
    (metrics_dir / "summary_metrics.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    cm_if = confusion_matrix(y_true, if_flags)
    cm_ae = confusion_matrix(y_true, ae_flags)
    cm_alert = confusion_matrix(y_true, alert_flags)
    plot_confusion_matrix(
        cm_if, "Isolation Forest", plots_dir / "confusion_matrix_if.png"
    )
    plot_confusion_matrix(
        cm_ae, "Autoencoder (MSE)", plots_dir / "confusion_matrix_ae.png"
    )
    plot_confusion_matrix(
        cm_alert,
        "Combined (OR)",
        plots_dir / "confusion_matrix_combined.png",
    )

    df_if = test_df[["timestamp", "is_anomaly"]].copy()
    df_if["score"] = if_scores
    df_if["pred"] = if_flags
    plot_timeline(
        df_if, "score", "pred", "Isolation Forest scores", plots_dir / "timeline_if.png"
    )

    df_ae = test_df[["timestamp", "is_anomaly"]].copy()
    df_ae["score"] = ae_scores
    df_ae["pred"] = ae_flags
    plot_timeline(
        df_ae,
        "score",
        "pred",
        "Autoencoder reconstruction MSE",
        plots_dir / "timeline_ae.png",
    )

    plot_score_hist(
        if_scores,
        y_true,
        "IF anomaly score distribution",
        plots_dir / "score_hist_if.png",
    )
    plot_score_hist(
        ae_scores,
        y_true,
        "AE reconstruction MSE distribution",
        plots_dir / "score_hist_ae.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate anomaly detectors on test split"
    )
    parser.add_argument("--test", type=Path, default=PROCESSED_TEST_CSV)
    parser.add_argument("--scaler", type=Path, default=MODEL_SCALER)
    parser.add_argument("--isolation-forest", type=Path, default=MODEL_ISOLATION_FOREST)
    parser.add_argument("--autoencoder", type=Path, default=MODEL_AUTOENCODER)
    parser.add_argument("--threshold", type=Path, default=MODEL_AUTOENCODER_THRESHOLD)
    args = parser.parse_args()

    run_evaluate(
        test_path=args.test,
        scaler_path=args.scaler,
        if_path=args.isolation_forest,
        ae_path=args.autoencoder,
        threshold_path=args.threshold,
    )
    print(f"Wrote reports to {OUTPUTS_METRICS_DIR}")
    print(f"Wrote plots to {OUTPUTS_PLOTS_DIR}")


if __name__ == "__main__":
    main()
