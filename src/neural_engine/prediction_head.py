"""PredictionHead — temporal regression module (Module C) — numpy only."""

import numpy as np

try:
    from neural_engine.lstm_cell import AdamOptimizer
except ImportError:
    from lstm_cell import AdamOptimizer  # direct script execution

GRAD_CLIP = 5.0


class PredictionHead:
    """Regression head that predicts steps to next micro-stop.

    Uses a single linear layer followed by ReLU activation.
    Owns its own AdamOptimizer instance.

    Args:
        hidden_size: Dimensionality of the input context vector.
    """

    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        self.W: np.ndarray = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b: np.ndarray = np.zeros(1)
        self._optimizer = AdamOptimizer()
        self._cache: dict = {}

    def forward(self, context: np.ndarray) -> np.ndarray:
        """Compute predicted steps to next micro-stop.

        Args:
            context: Context vectors, shape (batch, hidden_size).

        Returns:
            Predicted steps, shape (batch,). Non-negative via ReLU.
        """
        linear = context @ self.W + self.b       # (batch, 1)
        out = np.maximum(0.0, linear)            # ReLU, (batch, 1)
        self._cache = {"context": context, "linear": linear, "out": out}
        return out.flatten()                     # (batch,)

    def loss(self, pred: np.ndarray, y_time: np.ndarray) -> float:
        """Compute MSE loss.

        Args:
            pred:   Predicted steps, shape (batch,).
            y_time: Ground-truth steps, shape (batch,).

        Returns:
            Scalar MSE loss.
        """
        return float(np.mean((pred - y_time) ** 2))

    def backward(self, pred: np.ndarray, y_time: np.ndarray) -> np.ndarray:
        """Backpropagate MSE loss, update weights via Adam, and return upstream gradient.

        Args:
            pred:   Predicted steps, shape (batch,).
            y_time: Ground-truth steps, shape (batch,).

        Returns:
            Gradient w.r.t. context, shape (batch, hidden_size).
        """
        context = self._cache["context"]
        linear = self._cache["linear"].flatten()

        dL_dpred = 2.0 * (pred - y_time) / len(pred)           # (batch,)
        dL_dpred_col = dL_dpred[:, np.newaxis]                  # (batch, 1)

        relu_mask = (linear > 0).astype(np.float64)[:, np.newaxis]
        dL_dlinear = dL_dpred_col * relu_mask                   # (batch, 1)

        dW = context.T @ dL_dlinear                             # (hidden, 1)
        db = dL_dlinear.sum(axis=0)                             # (1,)

        dW = np.clip(dW, -GRAD_CLIP, GRAD_CLIP)
        db = np.clip(db, -GRAD_CLIP, GRAD_CLIP)

        # propagate before updating weights
        dL_dcontext = dL_dlinear @ self.W.T                     # (batch, hidden_size)

        params = {"W": self.W, "b": self.b}
        grads = {"W": dW, "b": db}
        updated = self._optimizer.update(params, grads)
        self.W = updated["W"]
        self.b = updated["b"]

        return dL_dcontext


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(7)

    ph = PredictionHead(hidden_size=32)
    context = np.random.randn(16, 32).astype(np.float32)
    y_time = np.random.uniform(0, 50, size=16)

    losses = []
    for _ in range(20):
        pred = ph.forward(context)
        losses.append(ph.loss(pred, y_time))
        ph.backward(pred, y_time)

    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print("PredictionHead OK ✓")
