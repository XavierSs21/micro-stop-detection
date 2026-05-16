"""Sliding-window construction and sequential train/val/test split — numpy only."""

import numpy as np


def create_windows(
    features: np.ndarray,
    *label_series: np.ndarray,
    window_size: int = 20,
) -> tuple:
    """Build sliding windows over a time-series feature matrix.

    Args:
        features: np.ndarray of shape (N, n_features).
        *label_series: zero or more (N,) np.ndarrays. Each label is taken at
                       index i + window_size for window i — i.e. the label
                       at the timestep immediately after the window.
        window_size: number of timesteps per window.

    Returns:
        If no label_series:
            X of shape (N - window_size, window_size, n_features).
        Otherwise:
            Tuple (X, y_1, y_2, ...) where each y_k has shape
            (N - window_size,) with the same dtype as the input series.
    """
    n = len(features)
    n_windows = n - window_size

    X = np.array([features[i:i + window_size] for i in range(n_windows)])

    if not label_series:
        return X

    labels_out = tuple(
        np.array([series[i + window_size] for i in range(n_windows)])
        for series in label_series
    )
    return (X,) + labels_out


def split_dataset(
    *arrays: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple:
    """Sequentially split each input array into (train, val, test) chunks.

    Order is preserved — no shuffling — to keep temporal causality.

    Args:
        *arrays: np.ndarrays of identical length on axis 0.
        train_frac: fraction of samples used for training.
        val_frac:   fraction used for validation. Test takes the remainder.

    Returns:
        Flat tuple of 3 * len(arrays) splits, ordered as
        (a1_train, a1_val, a1_test, a2_train, a2_val, a2_test, ...).
    """
    n = len(arrays[0])
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    out = []
    for a in arrays:
        out.append(a[:train_end])
        out.append(a[train_end:val_end])
        out.append(a[val_end:])
    return tuple(out)
