---
name: numpy-nn
description: Patterns and constraints for implementing neural networks manually with numpy only. Use this skill before writing ANY neural network code in this project — LSTM, GRU, attention, loss functions, optimizers, activations. This is the most critical skill for correctness.
---

# Manual Neural Networks with NumPy — micro-stop-detection

## Hard constraints (professor requirements)
- NO Keras, PyTorch, TensorFlow, JAX, or any ML framework
- NO `autograd` or automatic differentiation
- NO `model.fit()`, `model.forward()` from any framework
- YES numpy only for all math
- YES sklearn ONLY for metrics (accuracy_score, f1_score, recall_score, confusion_matrix)

## Activation functions (implement manually)
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def tanh(x):  # use np.tanh directly
    return np.tanh(x)
```

## Weight initialization (He for tanh/sigmoid)
```python
# For LSTM gates (input_size + hidden_size -> hidden_size)
fan_in = input_size + hidden_size
W = np.random.randn(fan_in, hidden_size) * np.sqrt(2.0 / fan_in)
b = np.zeros(hidden_size)
```

## LSTM cell pattern
```python
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        # Concatenated weight matrix for all 4 gates: [f, i, o, c_tilde]
        fan_in = input_size + hidden_size
        self.W = np.random.randn(fan_in, 4 * hidden_size) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(4 * hidden_size)
        self.hidden_size = hidden_size
        self.cache = {}  # store for backward pass
    
    def forward(self, X_seq):
        # X_seq: (batch, window_size, input_size)
        batch, T, n = X_seq.shape
        h = np.zeros((batch, self.hidden_size))
        c = np.zeros((batch, self.hidden_size))
        all_h = []
        
        for t in range(T):
            x_t = X_seq[:, t, :]
            concat = np.concatenate([x_t, h], axis=1)  # (batch, input+hidden)
            gates = concat @ self.W + self.b            # (batch, 4*hidden)
            H = self.hidden_size
            f = sigmoid(gates[:, 0*H:1*H])
            i = sigmoid(gates[:, 1*H:2*H])
            o = sigmoid(gates[:, 2*H:3*H])
            g = np.tanh(gates[:, 3*H:4*H])
            c = f * c + i * g
            h = o * np.tanh(c)
            all_h.append(h)
        
        # all_h: (batch, window_size, hidden_size)
        return np.stack(all_h, axis=1), h
```

## Gradient clipping (mandatory for BPTT)
```python
# After computing all gradients, before weight update:
GRAD_CLIP = 5.0
for key in grads:
    grads[key] = np.clip(grads[key], -GRAD_CLIP, GRAD_CLIP)
```

## Adam optimizer (simplified numpy)
```python
class AdamOptimizer:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.eps = eps
        self.m = {}; self.v = {}; self.t = 0
    
    def update(self, params, grads):
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params
```

## Loss functions (manual)
```python
def binary_cross_entropy(probs, y, class_weight=None):
    eps = 1e-12
    loss = -(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
    if class_weight is not None:
        weights = np.where(y == 1, class_weight, 1.0)
        loss = loss * weights
    return loss.mean()

def categorical_cross_entropy(probs, y_int):
    eps = 1e-12
    n = len(y_int)
    return -np.log(probs[np.arange(n), y_int.astype(int)] + eps).mean()

def mse_loss(pred, target):
    return np.mean((pred.flatten() - target.flatten()) ** 2)
```

## Attention pattern
```python
class Attention:
    def __init__(self, hidden_size):
        self.W_a = np.random.randn(hidden_size, hidden_size) * 0.01
    
    def forward(self, all_h):
        # all_h: (batch, T, hidden)
        scores = np.tanh(all_h @ self.W_a.T)   # (batch, T, hidden)
        energy = scores.sum(axis=-1)            # (batch, T)
        alpha = softmax(energy, axis=1)         # (batch, T)
        context = (alpha[:, :, np.newaxis] * all_h).sum(axis=1)  # (batch, hidden)
        return context, alpha
```

## Interface contract with rest of team
The `extract_context(X_batch)` function is the integration point:
```python
def extract_context(X_batch):
    """Drop-in replacement for simple_rnn_forward placeholder.
    
    Args:
        X_batch: np.ndarray of shape (batch, window_size, 4)
    Returns:
        context: np.ndarray of shape (batch, hidden_size=32)
    """
    all_h, _ = lstm.forward(X_batch)
    context, _ = attention.forward(all_h)
    return context
```
Input/output shape is identical to the placeholder — zero breaking changes for Jesus.
