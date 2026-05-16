# Sistema de detección de micro-paradas

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![numpy](https://img.shields.io/badge/numpy-only-green.svg)]()
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg)]()

Detección de micro-paradas en líneas de producción industrial mediante una red
neuronal recurrente (LSTM + Atención) implementada **100% en numpy**, sin
frameworks de deep learning.

---

## Descripción

Una micro-parada es una interrupción breve (1-10 s) en una línea de producción
que reduce la eficiencia sin activar alarmas convencionales. El sistema aprende
patrones temporales de cuatro sensores para detectar, clasificar y predecir
estas interrupciones.

**Entradas:** vibración, corriente, velocidad, temperatura.

**Salidas:**
- Detección binaria (hay / no hay micro-parada).
- Clasificación de causa (5 etiquetas, incluida la clase 0 = normal).
- Predicción temporal: pasos hasta la próxima micro-parada.

### Por qué numpy puro

Proyecto de la asignatura de Inteligencia Artificial. El objetivo es
implementar BPTT y Adam desde cero para entender cada operación matemática
involucrada, sin autograd ni capas preentrenadas.

---

## Arquitectura

```
Entrada (batch, 20, 4)        20 pasos × 4 sensores
       │
   [LSTMCell]                 (batch, 20, 32)
       │
   [Attention]                contexto: (batch, 32)
       │
   ┌───────────────┬──────────────────────┐
   │               │                      │
[DetectionHead] [ClassificationHead] [PredictionHead]
binario          5 causas             pasos al próximo
```

El vector de contexto `(batch, 32)` es el punto de integración entre la parte
secuencial y las cabezas de salida.

**Contrato de integración — no romper:**

```python
def extract_context(X_batch: np.ndarray) -> np.ndarray:
    # (batch, 20, 4) -> (batch, 32)
    all_h, _ = lstm.forward(X_batch)
    context, _ = attention.forward(all_h)
    return context
```

La implementación canónica vive en `src/pipeline.py` y es la única que debe
usarse fuera de los tests.

---

## Estructura del repositorio

```
src/
├── neural_engine/
│   ├── __init__.py            # API pública del paquete
│   ├── activations.py         # sigmoid, softmax, GRAD_CLIP (única fuente)
│   ├── optimizer.py           # AdamOptimizer (única fuente)
│   ├── lstm_cell.py           # LSTMCell
│   ├── attention.py           # Attention
│   └── prediction_head.py     # PredictionHead
├── preprocessing/
│   ├── __init__.py
│   ├── windows.py             # create_windows, split_dataset
│   └── normalize.py           # interpolate_nans, minmax_normalize
├── metrics/
│   ├── __init__.py
│   ├── classification.py      # confusion_matrix, accuracy, precision,
│   │                          # recall, f1, f1_macro, roc_curve,
│   │                          # buscar_mejor_umbral
│   └── anticipation.py        # calcular_anticipaciones,
│                              # calcular_anticipacion
└── pipeline.py                # extract_context() canónico

notebook/
├── dataset.ipynb              # generación de dataset y entrenamiento
├── views.ipynb                # visualizaciones principales
├── views2.ipynb               # evaluación con el modelo real
└── pruebas_manuales_v2.ipynb  # pruebas manuales

tests/
└── test_neural_engine.py      # 8 tests

data/                          # CSV generado (gitignored)
weights/                       # pesos_modelo.npy (gitignored)
```

---

## Módulos

### `neural_engine`

| Módulo | Contiene |
|--------|----------|
| `activations` | `sigmoid`, `softmax`, constante `GRAD_CLIP = 5.0` |
| `optimizer` | `AdamOptimizer` (lr=1e-3, beta1=0.9, beta2=0.999) |
| `lstm_cell` | `LSTMCell` con forward, BPTT y `update(grads, opt)` |
| `attention` | `Attention` con backward completo, expone `last_alpha` |
| `prediction_head` | `PredictionHead` (lineal + ReLU + MSE) |

Importación:

```python
from neural_engine import (
    LSTMCell, Attention, PredictionHead,
    AdamOptimizer, sigmoid, softmax, GRAD_CLIP,
)
```

### `preprocessing`

```python
from preprocessing.windows import create_windows, split_dataset
from preprocessing.normalize import interpolate_nans, minmax_normalize
```

- `create_windows(features, *label_series, window_size=20)`: variádico, soporta
  cualquier número de series de etiquetas; devuelve tuplas con las ventanas y
  los labels alineados al timestep posterior a la ventana.
- `split_dataset(*arrays, train_frac=0.70, val_frac=0.15)`: split temporal
  (sin shuffle) de N arrays en `(a₁_train, a₁_val, a₁_test, a₂_train, …)`.

### `metrics`

```python
from metrics.classification import (
    accuracy_manual, precision_manual, recall_manual,
    f1_manual, f1_macro_manual, confusion_matrix_manual,
    roc_curve_manual, buscar_mejor_umbral,
)
from metrics.anticipation import calcular_anticipaciones, calcular_anticipacion
```

Todo implementado en numpy puro. Sin sklearn ni siquiera para métricas.

`calcular_anticipaciones` devuelve el array de look-backs (primera alerta en la
ventana). `calcular_anticipacion` devuelve el promedio del look-back más corto
por evento (semántica del notebook canónico). Ambas conviven porque generan
métricas distintas.

### `pipeline`

```python
from pipeline import extract_context
context = extract_context(X_batch)   # (batch, 20, 4) -> (batch, 32)
```

Singleton interno: `_lstm` + `_attention` se instancian al importar.

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib pandas jupyter nbformat pytest
```

`pandas` se usa solo dentro de los notebooks (carga del CSV). `matplotlib` se
usa solo en notebooks; dentro de `src/` está restringido.

---

## Tests

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_neural_engine.py -v
```

| Test | Verifica |
|------|---------|
| `test_lstm_forward_shapes` | Shapes correctos en el forward |
| `test_lstm_gradient_flow` | Pérdida decrece en 30 iteraciones |
| `test_attention_weights_sum_to_one` | `alpha` suma 1 en el eje temporal |
| `test_prediction_head_gradient_flow` | Pérdida decrece en 20 iteraciones |
| `test_extract_context_contract` | `(8,20,4) → (8,32)` sin NaN |
| `test_no_framework_imports` | Ningún archivo en `src/` importa frameworks |
| `test_end_to_end_training` | Pipeline completo: pérdida decrece en 10 iter |
| `test_pipeline_extract_context_contract` | `pipeline.extract_context` respeta el contrato |

Los 8 tests deben pasar en verde en todo momento.

---

## Dataset y entrenamiento

```bash
.venv/bin/jupyter notebook notebook/dataset.ipynb
# Ejecutar todas las celdas
```

Esto genera:
- `data/industrial_dataset.csv` (20 000 muestras × 6 columnas).
- `weights/pesos_modelo.npy` con los pesos entrenados.

`pesos_modelo.npy` ahora guarda **todos** los pesos del pipeline:
`lstm_W`, `lstm_b`, `att_W_a`, `det_W`, `det_b`, `cls_W`, `cls_b`, `history`.
Antes guardaba solo los pesos de las cabezas, lo que impedía reconstruir el
modelo real en `views2.ipynb`.

---

## Fix de `views2.ipynb`

Antes del refactor, `views2.ipynb` evaluaba un modelo lineal sobre 8
estadísticos × 4 sensores = 32 features manuales (`extraer_caracteristicas`).
Esas métricas **no correspondían al modelo entrenado**.

Tras el refactor:
1. `dataset.ipynb` guarda también los pesos LSTM y Attention.
2. `views2.ipynb` reconstruye el pipeline real con esos pesos y produce el
   contexto mediante `extract_context_eval()` — el mismo espacio en el que se
   entrenaron `W_det` y `W_cls`.

Las métricas reportadas por `views2` ahora reflejan el modelo entrenado real.

---

## Restricciones

### Dentro de `src/`

| Permitido | Prohibido |
|-----------|-----------|
| `numpy` (toda la matemática) | Keras, PyTorch, TensorFlow, JAX |
| `math` (sólo escalar, sólo dentro de `neural_engine/`) | sklearn, scipy, statsmodels |
| `matplotlib` (sólo en `src/utils/plotters.py`) | `pandas` |
| | `os`, `sys` |
| | Autograd o backprop automático |

### Fuera de `src/` (notebooks y `scripts/`)

- `sys` está permitido en notebooks (para `sys.path.insert`).
- `os.path` está permitido en `scripts/` para construir rutas a `data/` y
  `weights/`.
- `pandas` está permitido para la carga del CSV en notebooks.

El test `test_no_framework_imports` verifica automáticamente que ningún
archivo en `src/neural_engine/` importe frameworks prohibidos.

---

Proyecto académico — Inteligencia Artificial.
