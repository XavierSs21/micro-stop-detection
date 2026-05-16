# Graph Report - src  (2026-05-16)

## Corpus Check
- Corpus is ~1,762 words - fits in a single context window. You may not need a graph.

## Summary
- 42 nodes · 43 edges · 11 communities (8 shown, 3 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Attention Layer|Attention Layer]]
- [[_COMMUNITY_LSTM Cell Core|LSTM Cell Core]]
- [[_COMMUNITY_Adam Optimizer & Init|Adam Optimizer & Init]]
- [[_COMMUNITY_LSTM Forward & Sigmoid|LSTM Forward & Sigmoid]]
- [[_COMMUNITY_Prediction Head Forward|Prediction Head Forward]]
- [[_COMMUNITY_Attention Forward & Softmax|Attention Forward & Softmax]]
- [[_COMMUNITY_Prediction Module|Prediction Module]]
- [[_COMMUNITY_MSE Loss Function|MSE Loss Function]]
- [[_COMMUNITY_Backpropagation|Backpropagation]]

## God Nodes (most connected - your core abstractions)
1. `AdamOptimizer` - 8 edges
2. `PredictionHead` - 7 edges
3. `Attention` - 6 edges
4. `LSTMCell` - 6 edges
5. `sigmoid()` - 3 edges
6. `softmax()` - 3 edges
7. `Softmax attention layer over LSTM hidden states — numpy only.` - 1 edges
8. `Softmax attention that condenses a sequence of hidden states into a context vect` - 1 edges
9. `Compute attention-weighted context vector.          Args:             all_h: LST` - 1 edges
10. `Backpropagate through attention, update W_a via Adam.          Args:` - 1 edges

## Surprising Connections (you probably didn't know these)
- `Attention` --uses--> `AdamOptimizer`  [INFERRED]
  neural_engine/attention.py → neural_engine/lstm_cell.py
- `PredictionHead` --uses--> `AdamOptimizer`  [INFERRED]
  neural_engine/prediction_head.py → neural_engine/lstm_cell.py

## Communities (11 total, 3 thin omitted)

### Community 0 - "Attention Layer"
Cohesion: 0.29
Nodes (4): Attention, Softmax attention layer over LSTM hidden states — numpy only., Softmax attention that condenses a sequence of hidden states into a context vect, Backpropagate through attention, update W_a via Adam.          Args:

### Community 1 - "LSTM Cell Core"
Cohesion: 0.29
Nodes (4): LSTMCell, Backpropagation through time.          Args:             dL_dall_h: Gradient of, Apply optimizer step to weights.          Args:             grads:     Gradient, Single-layer LSTM cell with BPTT using a concatenated weight matrix.      Gate o

### Community 2 - "Adam Optimizer & Init"
Cohesion: 0.33
Nodes (3): AdamOptimizer, Adam optimizer for parameter updates.      Args:         lr: Learning rate., Apply one Adam step.          Args:             params: Dict mapping param name

### Community 3 - "LSTM Forward & Sigmoid"
Cohesion: 0.33
Nodes (4): LSTM cell implementation with BPTT and Adam optimizer — numpy only., Numerically stable sigmoid.      Args:         x: Input array, any shape.      R, Full sequence forward pass.          Args:             X_seq: Input sequence, sh, sigmoid()

### Community 4 - "Prediction Head Forward"
Cohesion: 0.5
Nodes (3): PredictionHead, Regression head that predicts steps to next micro-stop.      Uses a single linea, Compute predicted steps to next micro-stop.          Args:             context:

### Community 5 - "Attention Forward & Softmax"
Cohesion: 0.5
Nodes (3): Compute attention-weighted context vector.          Args:             all_h: LST, Softmax along a given axis.      Args:         x: Input array, any shape., softmax()

## Knowledge Gaps
- **18 isolated node(s):** `Softmax attention layer over LSTM hidden states — numpy only.`, `Softmax attention that condenses a sequence of hidden states into a context vect`, `Compute attention-weighted context vector.          Args:             all_h: LST`, `Backpropagate through attention, update W_a via Adam.          Args:`, `LSTM cell implementation with BPTT and Adam optimizer — numpy only.` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AdamOptimizer` connect `Adam Optimizer & Init` to `Attention Layer`, `LSTM Forward & Sigmoid`, `Prediction Head Forward`?**
  _High betweenness centrality (0.611) - this node is a cross-community bridge._
- **Why does `PredictionHead` connect `Prediction Head Forward` to `Backpropagation`, `Adam Optimizer & Init`, `Prediction Module`, `MSE Loss Function`?**
  _High betweenness centrality (0.368) - this node is a cross-community bridge._
- **Why does `LSTMCell` connect `LSTM Cell Core` to `LSTM Forward & Sigmoid`?**
  _High betweenness centrality (0.293) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `AdamOptimizer` (e.g. with `Attention` and `PredictionHead`) actually correct?**
  _`AdamOptimizer` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Softmax attention layer over LSTM hidden states — numpy only.`, `Softmax attention that condenses a sequence of hidden states into a context vect`, `Compute attention-weighted context vector.          Args:             all_h: LST` to the rest of the system?**
  _18 weakly-connected nodes found - possible documentation gaps or missing edges._