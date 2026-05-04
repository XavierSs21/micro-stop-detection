"""pytest test suite for the neural_engine module — 6 tests."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_engine.lstm_cell import LSTMCell, AdamOptimizer
from neural_engine.attention import Attention
from neural_engine.prediction_head import PredictionHead


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lstm():
    return LSTMCell(input_size=4, hidden_size=32)


def _make_attention():
    return Attention(hidden_size=32)


def _make_prediction_head():
    return PredictionHead(hidden_size=32)


def extract_context(X_batch: np.ndarray) -> np.ndarray:
    lstm = _make_lstm()
    attention = _make_attention()
    all_h, _ = lstm.forward(X_batch)
    context, _ = attention.forward(all_h)
    return context


# ---------------------------------------------------------------------------
# Test 1 — LSTM forward shapes
# ---------------------------------------------------------------------------

def test_lstm_forward_shapes():
    lstm = _make_lstm()
    X = np.random.randn(8, 20, 4).astype(np.float32)
    all_h, h_last = lstm.forward(X)

    assert all_h.shape == (8, 20, 32), f"all_h shape: {all_h.shape}"
    assert h_last.shape == (8, 32), f"h_last shape: {h_last.shape}"
    assert not np.any(np.isnan(all_h)), "NaN in all_h"
    assert not np.any(np.isinf(all_h)), "Inf in all_h"


# ---------------------------------------------------------------------------
# Test 2 — LSTM gradient flow
# ---------------------------------------------------------------------------

def test_lstm_gradient_flow():
    np.random.seed(0)
    lstm = _make_lstm()
    opt = AdamOptimizer(lr=1e-2)
    X = np.random.randn(8, 20, 4).astype(np.float32)
    target = np.zeros((8, 20, 32))

    losses = []
    for _ in range(30):
        all_h, _ = lstm.forward(X)
        loss = float(np.mean((all_h - target) ** 2))
        losses.append(loss)
        dL = 2.0 * (all_h - target) / all_h.size
        grads = lstm.backward(dL)
        lstm.update(grads, opt)

    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"


# ---------------------------------------------------------------------------
# Test 3 — Attention weights sum to one
# ---------------------------------------------------------------------------

def test_attention_weights_sum_to_one():
    np.random.seed(1)
    att = _make_attention()
    all_h = np.random.randn(8, 20, 32)
    context, alpha = att.forward(all_h)

    assert context.shape == (8, 32), f"context shape: {context.shape}"
    assert np.allclose(alpha.sum(axis=1), 1.0, atol=1e-5), \
        f"alpha row sums: {alpha.sum(axis=1)}"


# ---------------------------------------------------------------------------
# Test 4 — PredictionHead gradient flow
# ---------------------------------------------------------------------------

def test_prediction_head_gradient_flow():
    np.random.seed(2)
    ph = _make_prediction_head()
    context = np.random.randn(16, 32).astype(np.float32)
    y_time = np.random.uniform(0, 50, size=16)

    losses = []
    for _ in range(20):
        pred = ph.forward(context)
        losses.append(ph.loss(pred, y_time))
        ph.backward(pred, y_time)

    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# ---------------------------------------------------------------------------
# Test 5 — extract_context integration contract
# ---------------------------------------------------------------------------

def test_extract_context_contract():
    np.random.seed(3)
    X = np.random.randn(8, 20, 4).astype(np.float32)
    context = extract_context(X)

    assert context.shape == (8, 32), \
        f"Expected (8, 32), got {context.shape}"
    assert not np.any(np.isnan(context)), "NaN in context"
    assert not np.any(np.isinf(context)), "Inf in context"


# ---------------------------------------------------------------------------
# Test 6 — no framework imports in source files
# ---------------------------------------------------------------------------

def test_no_framework_imports():
    forbidden = ["keras", "torch", "tensorflow", "jax"]
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src", "neural_engine")
    target_files = ["lstm_cell.py", "attention.py", "prediction_head.py"]

    for fname in target_files:
        fpath = os.path.join(src_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            source = f.read()
        for lib in forbidden:
            assert lib not in source, \
                f"Forbidden import '{lib}' found in {fname}"
