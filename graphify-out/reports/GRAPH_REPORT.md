# Graph Report - notebook  (2026-05-16)

## Corpus Check
- 4 files · ~9,144 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 212 nodes · 294 edges · 18 communities (16 shown, 2 thin omitted)
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 64 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Notebook Dataset AST|Notebook Dataset AST]]
- [[_COMMUNITY_Training & Evaluation Pipeline|Training & Evaluation Pipeline]]
- [[_COMMUNITY_Micro-Stop Causes|Micro-Stop Causes]]
- [[_COMMUNITY_Classification Metrics|Classification Metrics]]
- [[_COMMUNITY_Views Notebook AST|Views Notebook AST]]
- [[_COMMUNITY_Dataset Generation Functions|Dataset Generation Functions]]
- [[_COMMUNITY_Attention Layer (AST)|Attention Layer (AST)]]
- [[_COMMUNITY_Error Analysis & Events|Error Analysis & Events]]
- [[_COMMUNITY_DetectionHead (AST)|DetectionHead (AST)]]
- [[_COMMUNITY_Sensor Signal Charts|Sensor Signal Charts]]
- [[_COMMUNITY_Anticipation Analysis|Anticipation Analysis]]
- [[_COMMUNITY_Class Distribution|Class Distribution]]
- [[_COMMUNITY_Industrial Impact Simulation|Industrial Impact Simulation]]
- [[_COMMUNITY_Views2 Evaluation Functions|Views2 Evaluation Functions]]
- [[_COMMUNITY_Training Rationale|Training Rationale]]
- [[_COMMUNITY_Classification Rationale|Classification Rationale]]

## God Nodes (most connected - your core abstractions)
1. `Línea de tiempo: predicciones vs eventos reales` - 11 edges
2. `evaluate()` - 10 edges
3. `evaluate (dataset)` - 10 edges
4. `Bar chart: Precision, Recall y F1-score por causa de micro-parada` - 8 edges
5. `Confusion Matrix — Binary Micro-Stop Detection` - 8 edges
6. `Señales de sensores con micro-paradas marcadas (rojo)` - 7 edges
7. `Binary Classification Task (micro-stop vs. no micro-stop)` - 7 edges
8. `Vibration Signal` - 6 edges
9. `Current Signal` - 6 edges
10. `Micro-Stop Event (red vertical lines)` - 6 edges

## Surprising Connections (you probably didn't know these)
- `sigmoid (dataset inline)` --semantically_similar_to--> `sigmoid (views2)`  [INFERRED] [semantically similar]
  graphify-out/.nb_tmp/dataset.py → graphify-out/.nb_tmp/views2.py
- `extract_context (dataset)` --semantically_similar_to--> `extract_context (pruebas_manuales_v2)`  [INFERRED] [semantically similar]
  graphify-out/.nb_tmp/dataset.py → graphify-out/.nb_tmp/pruebas_manuales_v2.py
- `Binary Classification Task (micro-stop vs. no micro-stop)` --affects--> `Class Imbalance — Normal dominates over Micro-parada in dataset`  [INFERRED]
  notebook/outputs/04_curva_roc.png → notebook/outputs/01_matriz_confusion_db.png
- `create_windows (dataset)` --semantically_similar_to--> `create_windows (views)`  [INFERRED] [semantically similar]
  graphify-out/.nb_tmp/dataset.py → graphify-out/.nb_tmp/views.py
- `split_dataset (dataset)` --semantically_similar_to--> `split_dataset (views)`  [INFERRED] [semantically similar]
  graphify-out/.nb_tmp/dataset.py → graphify-out/.nb_tmp/views.py

## Hyperedges (group relationships)
- **Full Training Pipeline: LSTM → Attention → Heads → Evaluation** — dataset_extract_context, dataset_train_epoch, dataset_evaluate [EXTRACTED 1.00]
- **Dataset Generation Pipeline: inject → noise → missing → CSV** — dataset_inject_micro_stop, dataset_add_noise, dataset_add_missing [EXTRACTED 1.00]
- **Visualization & Evaluation Pipeline: ROC + confusion matrix + anticipation** — views2_calcular_roc_auc, views2_matriz_confusion, views_calcular_anticipaciones [INFERRED 0.85]

## Communities (18 total, 2 thin omitted)

### Community 0 - "Notebook Dataset AST"
Cohesion: 0.11
Nodes (14): accuracy_manual(), AdamOptimizer, calcular_anticipacion(), confusion_matrix_manual(), evaluate(), extract_context(), f1_macro_manual(), f1_manual() (+6 more)

### Community 1 - "Training & Evaluation Pipeline"
Cohesion: 0.1
Nodes (27): accuracy_manual, AdamOptimizer (dataset inline), Attention (dataset inline), calcular_anticipacion (dataset), ClassificationHead (dataset inline), confusion_matrix_manual (dataset), DetectionHead (dataset inline), evaluate (dataset) (+19 more)

### Community 2 - "Micro-Stop Causes"
Cohesion: 0.17
Nodes (20): Cause: Desalineación (Misalignment), Cause: Error humano (Human error), Cause: Fallo mecánico (Mechanical failure), Cause: Falta de material (Material shortage), Bar chart: Precision, Recall y F1-score por causa de micro-parada, ClassificationHead - Cause prediction model, Confusion Matrix - Cause Classification (Clasificación de causa), Insight: Two out of four classes have zero performance — suggests class imbalance or insufficient training for those classes (+12 more)

### Community 3 - "Classification Metrics"
Cohesion: 0.18
Nodes (18): AUC Score (0.898), Binary Classification Task (micro-stop vs. no micro-stop), Class Imbalance — Normal dominates over Micro-parada in dataset, Class: Micro-parada (micro-stop event), Class: Normal (operating state), Confusion Matrix — Binary Micro-Stop Detection, DetectionHead Output (binary classification output evaluated here), False Negative — Micro-parada incorrectly predicted as Normal (2) (+10 more)

### Community 4 - "Views Notebook AST"
Cohesion: 0.13
Nodes (6): ClassificationHead, DetectionHead, Replica sklearn.metrics.roc_curve — mismo resultado, numpy puro., roc_curve_manual(), sigmoid(), softmax()

### Community 5 - "Dataset Generation Functions"
Cohesion: 0.12
Nodes (16): add_missing, add_noise, compute_y_time, create_windows (dataset), industrial_dataset.csv (output), inject_micro_stop, pesos_modelo.npy (output), split_dataset (dataset) (+8 more)

### Community 6 - "Attention Layer (AST)"
Cohesion: 0.14
Nodes (7): Attention, ClassificationHead, Módulo B: clasifica la causa de la micro-parada.     Solo se activa si Detection, Args:             h: (batch, input_size)         Returns:             probs: (ba, Categorical Cross-Entropy: -mean[ sum_c( y_c * log(p_c) ) ], Gradiente de CCE + softmax. Actualiza pesos con SGD., softmax()

### Community 7 - "Error Analysis & Events"
Cohesion: 0.36
Nodes (12): Análisis de errores de clasificación (TP/FP/FN), Evento real (micro-parada ground truth), Falso negativo, Falso positivo, Línea de tiempo: predicciones vs eventos reales, Clase Micro-parada (estado binario), Micro-parada evento ~paso 250, Micro-parada evento ~paso 330 (+4 more)

### Community 9 - "DetectionHead (AST)"
Cohesion: 0.22
Nodes (5): DetectionHead, Módulo A: detecta si hay micro-parada en una ventana de tiempo.     Salida: prob, Args:             h: (batch, input_size) — vector de contexto del LSTM de Xavi, Binary Cross-Entropy ponderada.         Penaliza más los errores en micro-parada, Gradiente de BCE. Actualiza W y b con SGD.

### Community 10 - "Sensor Signal Charts"
Cohesion: 0.82
Nodes (8): Señales de sensores con micro-paradas marcadas (rojo), Current Signal, Industrial Dataset (20000 timesteps), Micro-Stop Event (red vertical lines), Speed Signal, Temperature Signal, Time Axis (Pasos de tiempo, 0–20000), Vibration Signal

### Community 11 - "Anticipation Analysis"
Cohesion: 0.36
Nodes (8): Anticipación 0 pasos antes (frecuencia: 2), Anticipación 2 pasos antes (frecuencia: 2), Anticipación 3 pasos antes (frecuencia: 1), Anticipación 5 pasos antes (frecuencia: 1), Insight: distribución irregular del tiempo de anticipación, picos en 0 y 2 pasos, Histograma: Distribución del tiempo de anticipación, Evento: Micro-parada industrial, Tiempo de anticipación (pasos antes de la micro-parada)

### Community 12 - "Class Distribution"
Cohesion: 0.6
Nodes (6): Distribución de Clases (Bar + Pie Chart), Class Imbalance — Dataset Desbalanceado, Pesos de Clase (Class Weights — solución al desbalance), Dataset Industrial (ventanas de series temporales), Clase Micro-parada (54 ventanas, 0.27%), Clase Normal (19926 ventanas, 99.73%)

### Community 13 - "Industrial Impact Simulation"
Cohesion: 0.53
Nodes (6): Downtime with detection system: 12.0 steps, Estimated downtime reduction: 50.00%, Downtime without detection system: 24.0 steps, Industrial impact simulation comparing scenarios with and without the AI detection system, Micro-stop detection system (LSTM + Attention neural network), Simulation Impact Chart: Downtime Comparison (Sin sistema vs Con sistema)

### Community 14 - "Views2 Evaluation Functions"
Cohesion: 0.4
Nodes (5): buscar_mejor_umbral, calcular_pasos_hasta_siguiente_evento, calcular_roc_auc, sigmoid (views2), roc_curve_manual (views)

## Knowledge Gaps
- **37 isolated node(s):** `ClassificationHead - Cause prediction model`, `Insight: Falta de material — high recall (1.0), low precision (~0.15), low F1 (~0.25); model finds all instances but with many false positives`, `Insight: Fallo mecánico — high precision (1.0), moderate recall (~0.5), F1 ~0.67; model is precise but misses half the instances`, `Clase Normal (estado binario)`, `Anticipación 3 pasos antes (frecuencia: 1)` (+32 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ClassificationHead` connect `Attention Layer (AST)` to `Notebook Dataset AST`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Why does `DetectionHead` connect `DetectionHead (AST)` to `Notebook Dataset AST`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Confusion Matrix — Binary Micro-Stop Detection` (e.g. with `High Specificity — Strong performance on Normal class (2950 TN, 39 FP)` and `Low Recall on Micro-parada — only 6 TP vs 2 FN (class imbalance challenge)`) actually correct?**
  _`Confusion Matrix — Binary Micro-Stop Detection` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ClassificationHead - Cause prediction model`, `Insight: Falta de material — high recall (1.0), low precision (~0.15), low F1 (~0.25); model finds all instances but with many false positives`, `Insight: Fallo mecánico — high precision (1.0), moderate recall (~0.5), F1 ~0.67; model is precise but misses half the instances` to the rest of the system?**
  _37 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Notebook Dataset AST` be split into smaller, more focused modules?**
  _Cohesion score 0.11 - nodes in this community are weakly interconnected._
- **Should `Training & Evaluation Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._
- **Should `Views Notebook AST` be split into smaller, more focused modules?**
  _Cohesion score 0.13 - nodes in this community are weakly interconnected._