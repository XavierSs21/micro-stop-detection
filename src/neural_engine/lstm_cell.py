"""LSTM cell implementation with BPTT — numpy only."""

import numpy as np

from neural_engine.activations import sigmoid, softmax, GRAD_CLIP
from neural_engine.optimizer import AdamOptimizer


# ---------------------------------------------------------------------------
# LSTM Cell
# ---------------------------------------------------------------------------

class LSTMCell:
    """Single-layer LSTM cell with BPTT using a concatenated weight matrix.

    Gate order in the concatenated axis: forget | input | output | candidate.

    Args:
        input_size:  Dimensionality of input features.
        hidden_size: Dimensionality of hidden / cell state.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size

        fan_in = input_size + hidden_size
        # W: (fan_in, 4*H), b: (4*H,)
        self.W: np.ndarray = np.random.randn(fan_in, 4 * hidden_size) * np.sqrt(2.0 / fan_in)
        self.b: np.ndarray = np.zeros(4 * hidden_size)

        self.cache: dict = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Full sequence forward pass.

        Args:
            X_seq: Input sequence, shape (batch, window_size, input_size).

        Returns:
            Tuple of:
                all_h:  Hidden states for every timestep,
                        shape (batch, window_size, hidden_size).
                h_last: Hidden state at final timestep,
                        shape (batch, hidden_size).
        """
        batch, T, _ = X_seq.shape
        H = self.hidden_size

        h = np.zeros((batch, H))
        c = np.zeros((batch, H))

        steps: list[dict] = []
        all_h = np.zeros((batch, T, H))

        for t in range(T):
            x_t = X_seq[:, t, :]                        # (batch, input_size)
            concat = np.concatenate([x_t, h], axis=1)   # (batch, fan_in)
            gates_raw = concat @ self.W + self.b         # (batch, 4*H)

            f = sigmoid(gates_raw[:, 0 * H: 1 * H])
            i = sigmoid(gates_raw[:, 1 * H: 2 * H])
            o = sigmoid(gates_raw[:, 2 * H: 3 * H])
            g = np.tanh(gates_raw[:, 3 * H: 4 * H])

            c_new = f * c + i * g
            tanh_c = np.tanh(c_new)
            h_new = o * tanh_c

            steps.append({
                "concat": concat,
                "gates_raw": gates_raw,
                "f": f, "i": i, "o": o, "g": g,
                "c_prev": c,
                "c": c_new,
                "tanh_c": tanh_c,
                "h_prev": h,
                "h": h_new,
            })
            all_h[:, t, :] = h_new
            h, c = h_new, c_new

        self.cache = {"steps": steps, "X_seq": X_seq, "T": T}
        return all_h, h

    # ------------------------------------------------------------------
    # Backward (BPTT)
# ------------------------------------------------------------------

    def backward(self, dL_dall_h: np.ndarray) -> dict:
        """Backpropagation through time.

        Args:
            dL_dall_h: Gradient of loss w.r.t. all hidden states,
                       shape (batch, window_size, hidden_size).

        Returns:
            Dict with keys 'W' and 'b' containing accumulated gradients,
            clipped to [-GRAD_CLIP, GRAD_CLIP].
        """
        H = self.hidden_size
        steps = self.cache["steps"]
        T = self.cache["T"]

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        dh_next = np.zeros_like(steps[0]["h"])
        dc_next = np.zeros_like(steps[0]["c"])

        for t in reversed(range(T)):
            s = steps[t]
            dh = dL_dall_h[:, t, :] + dh_next       # (batch, H)

            # h = o * tanh(c)
            do = dh * s["tanh_c"]
            dtanh_c = dh * s["o"]

            # tanh_c = tanh(c)  ->  d/dc = 1 - tanh^2
            dc = dtanh_c * (1 - s["tanh_c"] ** 2) + dc_next

            # c = f * c_prev + i * g
            df = dc * s["c_prev"]
            di = dc * s["g"]
            dg = dc * s["i"]
            dc_prev = dc * s["f"]

            # gate activations
            df_raw = df * s["f"] * (1 - s["f"])
            di_raw = di * s["i"] * (1 - s["i"])
            do_raw = do * s["o"] * (1 - s["o"])
            dg_raw = dg * (1 - s["g"] ** 2)

            dgates = np.concatenate([df_raw, di_raw, do_raw, dg_raw], axis=1)  # (batch, 4H)

            dW += s["concat"].T @ dgates
            db += dgates.sum(axis=0)

            dconcat = dgates @ self.W.T
            dh_next = dconcat[:, self.input_size:]
            dc_next = dc_prev

        dW = np.clip(dW, -GRAD_CLIP, GRAD_CLIP)
        db = np.clip(db, -GRAD_CLIP, GRAD_CLIP)
        return {"W": dW, "b": db}

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, grads: dict, optimizer: AdamOptimizer) -> None:
        """Apply optimizer step to weights.

        Args:
            grads:     Gradient dict with keys 'W' and 'b'.
            optimizer: An AdamOptimizer instance.
        """
        params = {"W": self.W, "b": self.b}
        updated = optimizer.update(params, grads)
        self.W = updated["W"]
        self.b = updated["b"]
