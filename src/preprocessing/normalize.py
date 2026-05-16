"""Missing-value interpolation and min-max normalization — numpy only."""

import numpy as np


def interpolate_nans(col: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1-D series.

    Args:
        col: 1-D np.ndarray, may contain NaNs.

    Returns:
        np.ndarray of the same shape with NaNs filled by linear interpolation
        between valid neighbours. Leading/trailing NaNs are filled with the
        nearest valid value.
    """
    indices = np.arange(len(col))
    valid = ~np.isnan(col)
    if valid.any():
        col = np.interp(indices, indices[valid], col[valid])
    return col


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize each column of a 2-D array independently.

    Args:
        arr: 2-D np.ndarray of shape (N, n_features).

    Returns:
        np.ndarray of the same shape with each column scaled to [0, 1].
        Columns that are constant are left unchanged (denominator clamped to 1).
    """
    arr = arr.astype(np.float64, copy=True)
    col_min = arr.min(axis=0, keepdims=True)
    col_max = arr.max(axis=0, keepdims=True)
    denom = col_max - col_min
    denom[denom == 0] = 1.0
    return (arr - col_min) / denom
