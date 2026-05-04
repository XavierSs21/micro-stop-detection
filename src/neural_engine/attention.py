"""Softmax attention layer over LSTM hidden states — numpy only."""

import numpy as np

try:
    from neural_engine.lstm_cell import sigmoid, softmax
except ImportError:
    from lstm_cell import sigmoid, softmax  # direct script execution


class Attention:
    """Softmax attention that condenses a sequence of hidden states into a context vector.

    Args:
        hidden_size: Dimensionality of LSTM hidden states.
    """

    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        self.W_a: np.ndarray = np.random.randn(hidden_size, hidden_size) * 0.01
        self.last_alpha: np.ndarray | None = None

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
        return context, alpha


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)

    att = Attention(hidden_size=32)
    all_h = np.random.randn(8, 20, 32)

    context, alpha = att.forward(all_h)

    assert context.shape == (8, 32), f"context shape mismatch: {context.shape}"
    assert np.allclose(alpha.sum(axis=1), 1.0, atol=1e-6), \
        f"alpha does not sum to 1: {alpha.sum(axis=1)}"

    print("Attention OK ✓")
