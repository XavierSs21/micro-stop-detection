"""Softmax attention layer over LSTM hidden states — numpy only."""

import numpy as np

from neural_engine.activations import softmax, GRAD_CLIP
from neural_engine.optimizer import AdamOptimizer


class Attention:
    """Softmax attention that condenses a sequence of hidden states into a context vector.

    Args:
        hidden_size: Dimensionality of LSTM hidden states.
    """

    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        self.W_a: np.ndarray = np.random.randn(hidden_size, hidden_size) * 0.01
        self.last_alpha: np.ndarray | None = None
        self._cache_all_h: np.ndarray | None = None
        self._optimizer = AdamOptimizer()

    def forward(self, all_h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute attention-weighted context vector.

        Args:
            all_h: LSTM hidden states, shape (batch, T, hidden_size).

        Returns:
            Tuple of:
                context: Attention-weighted sum of hidden states,
                         shape (batch, hidden_size).
                alpha:   Attention weights (sum to 1 along T),
                         shape (batch, T).
        """
        scores = np.tanh(all_h @ self.W_a.T)           # (batch, T, hidden_size)
        energy = scores.sum(axis=-1)                    # (batch, T)
        alpha = softmax(energy, axis=1)                 # (batch, T)
        context = (alpha[:, :, np.newaxis] * all_h).sum(axis=1)  # (batch, hidden_size)
        self.last_alpha = alpha
        self._cache_all_h = all_h
        return context, alpha

    def backward(self, dL_dcontext: np.ndarray) -> np.ndarray:
        """Backpropagate through attention, update W_a via Adam.

        Args:
            dL_dcontext: Gradient of loss w.r.t. context, shape (batch, hidden_size).

        Returns:
            Gradient w.r.t. all_h, shape (batch, T, hidden_size).
        """
        all_h = self._cache_all_h
        alpha = self.last_alpha                          # (batch, T)
        batch, T, H = all_h.shape

        # gradient through context = sum_t(alpha_t * h_t)
        dL_dall_h_direct = alpha[:, :, np.newaxis] * dL_dcontext[:, np.newaxis, :]  # (batch, T, H)
        dL_dalpha = (dL_dcontext[:, np.newaxis, :] * all_h).sum(axis=-1)            # (batch, T)

        # gradient through softmax
        dL_denergy = alpha * (dL_dalpha - (alpha * dL_dalpha).sum(axis=1, keepdims=True))  # (batch, T)

        # gradient through energy = scores.sum(-1)
        dL_dscores = dL_denergy[:, :, np.newaxis]       # (batch, T, H) via broadcast

        # gradient through scores = tanh(all_h @ W_a.T)
        scores = np.tanh(all_h @ self.W_a.T)            # (batch, T, H)
        dL_dpretanh = dL_dscores * (1.0 - scores ** 2) # (batch, T, H)

        dW_a = dL_dpretanh.reshape(-1, H).T @ all_h.reshape(-1, H)  # (H, H)
        dW_a = np.clip(dW_a, -GRAD_CLIP, GRAD_CLIP)

        # propagate before updating weights
        dL_dall_h_score = dL_dpretanh @ self.W_a        # (batch, T, H)

        params = {"W_a": self.W_a}
        grads = {"W_a": dW_a}
        updated = self._optimizer.update(params, grads)
        self.W_a = updated["W_a"]

        return dL_dall_h_direct + dL_dall_h_score        # (batch, T, H)
