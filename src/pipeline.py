"""Canonical pipeline: extract_context() bridges raw windows to the heads."""

import numpy as np

from neural_engine import Attention, LSTMCell


HIDDEN_SIZE = 32
INPUT_SIZE = 4
WINDOW_SIZE = 20


_lstm = LSTMCell(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
_attention = Attention(hidden_size=HIDDEN_SIZE)


def extract_context(X_batch: np.ndarray) -> np.ndarray:
    """Integration contract: (batch, 20, 4) -> (batch, 32).

    Esta es la única implementación canónica. No reimplementar en notebooks.
    """
    all_h, _ = _lstm.forward(X_batch)
    context, _ = _attention.forward(all_h)
    return context
