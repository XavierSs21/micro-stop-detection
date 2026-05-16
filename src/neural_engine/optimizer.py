"""Adam optimizer for manual neural networks — numpy only."""

import numpy as np


class AdamOptimizer:
    """Adam optimizer for parameter updates.

    Args:
        lr: Learning rate.
        beta1: Exponential decay rate for first moment.
        beta2: Exponential decay rate for second moment.
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict = {}
        self.v: dict = {}
        self.t: int = 0

    def update(self, params: dict, grads: dict) -> dict:
        """Apply one Adam step.

        Args:
            params: Dict mapping param name -> np.ndarray.
            grads:  Dict mapping param name -> gradient np.ndarray (same keys).

        Returns:
            Updated params dict (same keys, new arrays).
        """
        self.t += 1
        updated = {}
        for key, p in params.items():
            g = grads[key]
            self.m.setdefault(key, np.zeros_like(p))
            self.v.setdefault(key, np.zeros_like(p))

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            updated[key] = p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return updated
