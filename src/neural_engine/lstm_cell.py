"""LSTM cell implementation with BPTT and Adam optimizer — numpy only."""

import numpy as np


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------

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
            clipped to [-5.0, 5.0].
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

        dW = np.clip(dW, -5.0, 5.0)
        db = np.clip(db, -5.0, 5.0)
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


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    lstm = LSTMCell(input_size=4, hidden_size=32)
    X = np.random.randn(8, 20, 4)

    all_h, h_last = lstm.forward(X)
    assert all_h.shape == (8, 20, 32), f"all_h shape mismatch: {all_h.shape}"
    assert h_last.shape == (8, 32), f"h_last shape mismatch: {h_last.shape}"

    opt = AdamOptimizer(lr=1e-2)
    target = np.zeros((8, 20, 32))
    losses = []

    for _ in range(10):
        all_h, _ = lstm.forward(X)
        loss = float(np.mean((all_h - target) ** 2))
        losses.append(loss)

        dL = 2 * (all_h - target) / all_h.size
        grads = lstm.backward(dL)
        lstm.update(grads, opt)

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    print("LSTMCell OK ✓")
