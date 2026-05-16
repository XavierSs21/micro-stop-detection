"""Binary and multiclass classification metrics — numpy only."""

import numpy as np


def confusion_matrix_manual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels,
) -> np.ndarray:
    """Confusion matrix with explicit label ordering.

    Args:
        y_true: 1-D array of ground-truth labels.
        y_pred: 1-D array of predicted labels (same length).
        labels: sequence of labels in the desired row/column order.

    Returns:
        np.ndarray of shape (len(labels), len(labels)) of int counts.
        cm[i, j] = number of samples whose true label is labels[i] and
        whose predicted label is labels[j].
    """
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for i, real in enumerate(labels):
        for j, pred_lbl in enumerate(labels):
            cm[i, j] = np.sum((y_true == real) & (y_pred == pred_lbl))
    return cm


def accuracy_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified samples."""
    return float(np.mean(y_true == y_pred))


def precision_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary precision for the positive class (label == 1)."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary recall for the positive class (label == 1)."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1 for the positive class (label == 1)."""
    p = precision_manual(y_true, y_pred)
    r = recall_manual(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def f1_macro_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 over the distinct classes present in y_true."""
    scores = []
    for c in np.unique(y_true):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        scores.append(f1)
    return float(np.mean(scores))


def roc_curve_manual(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple:
    """ROC curve evaluated at every distinct score plus an extra +inf threshold.

    Args:
        y_true:  1-D array of binary labels (0/1).
        y_score: 1-D array of real-valued scores (higher = more positive).

    Returns:
        Tuple (fpr, tpr, thresholds) of np.ndarrays.
    """
    thresholds = np.concatenate(
        [[1.0 + 1e-8], np.sort(np.unique(y_score))[::-1]]
    )
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)

    tprs, fprs = [], []
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fp = np.sum((y_true == 0) & (preds == 1))
        tprs.append(tp / pos if pos > 0 else 0.0)
        fprs.append(fp / neg if neg > 0 else 0.0)

    return np.array(fprs), np.array(tprs), thresholds


def buscar_mejor_umbral(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_thresholds: int = 300,
) -> tuple:
    """Grid-search the score threshold that maximises F1.

    Args:
        y_true:  1-D array of binary labels (0/1).
        y_score: 1-D array of real-valued scores.
        n_thresholds: number of thresholds in the linspace grid.

    Returns:
        Tuple (best_threshold, metrics_tuple) where metrics_tuple is
        (tp, fp, fn, tn, precision, recall, f1) at the best threshold.
    """
    thresholds = np.linspace(np.min(y_score), np.max(y_score), n_thresholds)

    mejor_f1 = -1.0
    mejor_umbral = None
    mejores_metricas = None

    for t in thresholds:
        y_pred_tmp = (y_score >= t).astype(int)

        tp = np.sum((y_true == 1) & (y_pred_tmp == 1))
        fp = np.sum((y_true == 0) & (y_pred_tmp == 1))
        fn = np.sum((y_true == 1) & (y_pred_tmp == 0))
        tn = np.sum((y_true == 0) & (y_pred_tmp == 0))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_umbral = t
            mejores_metricas = (int(tp), int(fp), int(fn), int(tn),
                                float(p), float(r), float(f1))

    return mejor_umbral, mejores_metricas
