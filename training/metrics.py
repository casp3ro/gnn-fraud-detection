"""
Evaluation metrics for imbalanced binary classification (fraud detection).

We care about recall (catching frauds) and precision (limiting false alarms),
plus F1 and AUC-ROC. All metrics are computed from predicted logits/probs and labels.
"""

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, F1, and optionally AUC-ROC.

    For fraud detection, recall (sensitivity) = fraction of frauds caught;
    precision = fraction of predicted frauds that are real frauds.

    Args:
        y_true: Ground truth binary labels (0 or 1), shape (n,).
        y_pred: Predicted binary labels (0 or 1), shape (n,).
        y_prob: Predicted probabilities for the positive class, shape (n,). If None, AUC is skipped.

    Returns:
        Dictionary of metric names to float values. Keys: accuracy, precision, recall, f1;
        and "roc_auc" if y_prob is provided.
    """
    # sklearn metrics expect 1d arrays; ensure we pass 1d.
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob).ravel()
        if len(np.unique(y_true)) >= 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        else:
            metrics["roc_auc"] = 0.0
    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Confusion matrix: rows = true, cols = predicted.
    [[TN, FP], [FN, TP]] for (neg, pos) order.
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def logits_to_pred_and_prob(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw logits (N, 1) to binary predictions and probabilities.

    Returns:
        y_pred: 0/1 predictions (rounded sigmoid).
        y_prob: Probabilities for the positive class (sigmoid).
    """
    prob = torch.sigmoid(logits).detach().cpu().numpy().ravel()
    pred = (prob >= 0.5).astype(np.int64)
    return pred, prob
