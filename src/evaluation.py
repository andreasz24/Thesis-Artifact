"""
Shared evaluation helpers.

In the original script the metric computation, classification report, and
confusion-matrix code were copy-pasted under each of the three models. Here
they live once and every model notebook calls them.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src import config


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy + macro/weighted F1 in one call."""
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def present_labels(y_true):
    """Class indices actually present in y_true, plus their readable labels."""
    present_classes = sorted(np.unique(y_true))
    labels = [config.RATING_LABELS[i] for i in present_classes]
    return present_classes, labels


def print_metrics(name, metrics, train_time=None, inference_time=None, n_test=None):
    """Pretty-print the headline metrics block (mirrors the original output)."""
    print(f"\n-- {name} Results " + "-" * (40 - len(name)))
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)    : {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted) : {metrics['f1_weighted']:.4f}")
    if train_time is not None:
        print(f"  Training time : {train_time:.2f}s")
    if inference_time is not None:
        extra = f"  ({n_test} samples)" if n_test is not None else ""
        print(f"  Inference time: {inference_time:.4f}s{extra}")
    print("-" * 50)


def per_class_report(y_true, y_pred):
    """Return the sklearn per-class classification report as a string."""
    present_classes, labels = present_labels(y_true)
    return classification_report(
        y_true, y_pred,
        labels=present_classes,
        target_names=labels,
        zero_division=0,
    )


def confusion_matrices(y_true, y_pred):
    """Return (counts, row_normalised, present_classes, labels)."""
    present_classes, labels = present_labels(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    return cm, cm_norm, present_classes, labels
