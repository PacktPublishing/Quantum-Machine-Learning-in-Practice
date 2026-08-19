"""Metric helpers for classical and quantum-model comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def classification_scores(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return common classification scores with stable numeric types."""

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, y_pred_array)
        ),
        "f1_weighted": float(
            f1_score(y_true_array, y_pred_array, average="weighted", zero_division=0)
        ),
    }


def confusion_counts(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, Any]:
    """Return a compact confusion-matrix payload that is easy to serialize."""

    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    labels = unique_labels(y_true_array, y_pred_array)
    matrix = confusion_matrix(y_true_array, y_pred_array, labels=labels)
    return {
        "labels": labels.tolist(),
        "matrix": matrix.astype(int).tolist(),
    }
