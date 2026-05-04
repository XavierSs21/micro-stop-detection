## Summary

<!-- One paragraph: what this PR does and why. -->

- 
- 
- 

## Changes

| File | Change |
|------|--------|
| `src/neural_engine/lstm_cell.py` | New: LSTMCell, AdamOptimizer, helpers |

## Architecture notes

<!-- Gate layout, weight shapes, anything non-obvious about the implementation. -->

- Concatenated weight matrix `W: (input_size + hidden_size, 4 * hidden_size)` — gate order: **forget | input | output | candidate**
- He initialization: `* sqrt(2 / fan_in)`
- BPTT with gradient clipping to `[-5.0, 5.0]`
- Adam optimizer (`lr=1e-3, β1=0.9, β2=0.999`) with bias-correction

## Test plan

- [ ] Smoke test passes: `python3 src/neural_engine/lstm_cell.py` prints `LSTMCell OK ✓`
- [ ] Shape assertions: `all_h (8, 20, 32)`, `h_last (8, 32)`
- [ ] Loss decreases over 10 training steps
- [ ] No framework dependencies — numpy only

## Related issues

<!-- Closes # -->

## Checklist

- [ ] Smoke test passes locally
- [ ] No ML framework imports
- [ ] Docstrings (Google style) on every method
- [ ] Type hints on all public methods
