# src/evaluation.py
from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Compute common classification metrics. If y_proba is provided, also returns ROC-AUC.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    ax: plt.Axes | None = None,
    label: str = "model",
) -> plt.Axes:
    """
    Plot ROC curve and return the Axes object.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    ax: plt.Axes | None = None,
    label: str = "model",
) -> plt.Axes:
    """
    Plot Precision–Recall curve and return the Axes object.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(recall, precision, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend()
    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax: plt.Axes | None = None,
    labels: Tuple[str, str] = ("No Churn", "Churn"),
) -> plt.Axes:
    """
    Simple confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    return ax


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Wrapper to print sklearn classification report.
    """
    print(classification_report(y_true, y_pred, digits=3))
