from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "f1": f1_score(labels, predictions, zero_division=0),
                "precision": precision_score(labels, predictions, zero_division=0),
                "recall": recall_score(labels, predictions, zero_division=0),
            }
        )
    frame = pd.DataFrame(rows)
    best = frame.sort_values(["f1", "recall", "threshold"], ascending=[False, False, True]).iloc[0]
    return float(best["threshold"]), frame


def metrics_at_threshold(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
    }


def save_evaluation(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    output_dir: Path,
    model_name: str,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = metrics_at_threshold(labels, probabilities, threshold)
    predictions = (probabilities >= threshold).astype(int)
    (output_dir / f"{model_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(labels, predictions)
    fig, axis = plt.subplots(figsize=(5.8, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Real", "AI generated"],
        yticklabels=["Real", "AI generated"],
        ax=axis,
    )
    axis.set(xlabel="Predicted label", ylabel="True label", title=f"{model_name} confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=180)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    fig, axis = plt.subplots(figsize=(6.4, 5))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC = {metrics['roc_auc']:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Random classifier")
    axis.set(
        xlabel="False-positive rate", ylabel="True-positive rate", title=f"{model_name} ROC curve"
    )
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_roc_curve.png", dpi=180)
    plt.close(fig)
    return metrics


def save_training_curves(
    history: dict[str, list[float]], output_path: Path, best_epoch: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(history["loss"]) + 1)
    for axis, metric in zip(axes, ("loss", "accuracy"), strict=True):
        axis.plot(epochs, history[metric], label=f"Training {metric}")
        axis.plot(epochs, history[f"val_{metric}"], label=f"Validation {metric}")
        axis.axvline(best_epoch, color="#ef4444", linestyle="--", label=f"Best epoch {best_epoch}")
        axis.set(xlabel="Epoch", ylabel=metric.title(), title=f"Training and validation {metric}")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
