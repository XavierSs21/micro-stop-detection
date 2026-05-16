"""Activation functions and shared numeric constants — numpy only."""

import numpy as np


GRAD_CLIP = 5.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid.

    Args:
        x: Input array, any shape.

    Returns:
        Element-wise sigmoid of x, same shape as x.
    """
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax along a given axis.

    Args:
        x: Input array, any shape.
        axis: Axis along which softmax is computed.

    Returns:
        Softmax-normalized array, same shape as x.
    """
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)
