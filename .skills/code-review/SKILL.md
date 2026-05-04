---
name: code-review
description: Deep review of implemented neural network modules. Use this skill after completing any phase (lstm, attention, prediction, tests, notebook) to verify correctness before merging to main. Always run this before any merge commit.
---

# Code Review — neural_engine modules

## When to trigger
Run this review after EVERY phase implementation, before merging to main.

## Review checklist

### 1. Correctness
- [ ] Forward pass output shapes match contract: `extract_context` returns `(batch, 32)`
- [ ] No NaN or Inf in any forward pass output
- [ ] Attention weights sum to 1.0 along time axis (± 1e-5)
- [ ] Loss decreases over training (gradient flow confirmed)
- [ ] Gradient clipping applied in BPTT: `np.clip(grad, -5, 5)`
- [ ] LSTM gates use correct activations: f/i/o → sigmoid, g → tanh

### 2. Framework compliance
- [ ] Zero imports from: keras, torch, tensorflow, jax
- [ ] sklearn used ONLY in metrics (accuracy, f1, recall, confusion_matrix)
- [ ] All math is explicit numpy operations

### 3. Code quality
- [ ] Google-style docstrings on every class and method
- [ ] Type hints using np.ndarray
- [ ] No magic numbers — constants named (HIDDEN_SIZE, GRAD_CLIP, etc.)
- [ ] Cache stored for backward pass in LSTMCell

### 4. Integration contract
- [ ] `extract_context(X_batch)` accepts `(batch, 20, 4)` → returns `(batch, 32)`
- [ ] Shape is hardcoded nowhere — uses `self.hidden_size` throughout
- [ ] `AdamOptimizer` importable from `lstm_cell.py` by other modules

### 5. Tests
- [ ] All 6 pytest tests pass: `python3 -m pytest tests/ -v`
- [ ] Smoke test in each module runs clean
- [ ] No test imports ML frameworks

## How to run the review
1. Run all smoke tests
```bash
wsl -d Debian bash -c "PYTHONPATH=src .venv/bin/python3 src/neural_engine/lstm_cell.py"
wsl -d Debian bash -c "PYTHONPATH=src .venv/bin/python3 src/neural_engine/attention.py"
wsl -d Debian bash -c "PYTHONPATH=src .venv/bin/python3 src/neural_engine/prediction_head.py"
```
2. Run full test suite
```bash
wsl -d Debian bash -c "PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_neural_engine.py -v -s"
```
3. Check for forbidden imports
```bash
wsl -d Debian bash -c "grep -r 'import torch\|import keras\|import tensorflow\|import jax' src/neural_engine/"
```
4. Verify full pipeline shapes
```bash
wsl -d Debian bash -c "PYTHONPATH=src .venv/bin/python3 -c \"
import numpy as np
from neural_engine.lstm_cell import LSTMCell
from neural_engine.attention import Attention
lstm = LSTMCell(4, 32)
att = Attention(32)
X = np.random.randn(8,20,4).astype('float32')
all_h, _ = lstm.forward(X)
ctx, alpha = att.forward(all_h)
assert ctx.shape == (8,32)
assert np.allclose(alpha.sum(axis=1), 1.0, atol=1e-5)
print('Full pipeline OK ✓')
\""
```

## What to report after review
For each phase, output a review summary:
```
## Review — Phase N: <name>
- Shapes:      ✓ / ✗ (detail)
- Gradients:   ✓ / ✗ (loss[0] vs loss[-1])
- Compliance:  ✓ / ✗ (any forbidden imports?)
- Tests:       N/6 passing
- Contract:    ✓ / ✗ (extract_context shape confirmed)
- Merge ready: YES / NO
```
Only output "Merge ready: YES" if ALL checks pass.
