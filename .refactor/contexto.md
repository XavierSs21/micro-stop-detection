# Contexto de refactor — micro-stop-detection

Rama: `feat/refactor-modular`  
Objetivo: hacer el proyecto legible y modular sin romper las restricciones de la materia
(numpy only, sin frameworks, sin `os`/`sys` dentro de `src/neural_engine/`).

---

## Estado actual del código

### `src/neural_engine/` — lo que funciona bien

Los tres módulos core están limpios. Cada archivo tiene una sola responsabilidad y
los 7 tests pasan. Esto NO se toca salvo para reorganizar dependencias internas.

```
src/neural_engine/
  lstm_cell.py       280 líneas — LSTMCell + AdamOptimizer + sigmoid + softmax
  attention.py       105 líneas — Attention
  prediction_head.py 112 líneas — PredictionHead
  __init__.py        vacío
```

### Problemas concretos en `src/`

**1. `sigmoid` y `softmax` viven en `lstm_cell.py`**
Ambas funciones son utilidades genéricas que usan `attention.py` y las reimplementan
los notebooks. Deberían estar en un archivo separado de activaciones/utilidades.

**2. `AdamOptimizer` vive en `lstm_cell.py`**
`attention.py` y `prediction_head.py` lo importan desde ahí. Es un acoplamiento
accidental — el optimizador no le pertenece a la celda LSTM.

**3. Doble import en `attention.py` y `prediction_head.py`**
```python
try:
    from neural_engine.lstm_cell import sigmoid, softmax, AdamOptimizer
except ImportError:
    from lstm_cell import sigmoid, softmax, AdamOptimizer  # ejecución directa
```
Este patrón existe porque los archivos se ejecutaban directamente con `python3 attention.py`.
El smoke test `if __name__ == "__main__"` al final de cada archivo es la causa.
Con el paquete instalado correctamente y PYTHONPATH=src, el fallback no debería ser
necesario.

**4. `__init__.py` vacío**
Nadie puede hacer `from neural_engine import LSTMCell` desde fuera. La API pública
del paquete no está expuesta.

**5. `GRAD_CLIP = 5.0` duplicado**
Definido como constante de módulo en `attention.py` y `prediction_head.py`.
En `lstm_cell.py` está hardcodeado dentro del método (`np.clip(dW, -5.0, 5.0)`).

---

### Problemas en los notebooks

**6. `extract_context()` reimplementada 3 veces**
- `dataset.ipynb` — versión con LSTM+Atención real (la canónica)
- `views.ipynb` — versión con RNN simple (diferente arquitectura)
- `pruebas_manuales_v2.ipynb` — importa desde `src/` correctamente

La única implementación que debería existir fuera de los tests es la de `src/`.

**7. `views2.ipynb` usa features manuales en lugar de LSTM**
`extraer_caracteristicas()` computa 8 estadísticos × 4 sensores = 32 valores por
ventana, lo que coincide dimensionalmente con la salida de `extract_context()` (batch, 32).
La razón: `pesos_modelo.npy` solo guarda `det_W`, `det_b`, `cls_W`, `cls_b`
(pesos de las cabezas), pero NO los pesos LSTM ni Attention. Sin ellos, no se puede
reconstruir el pipeline real.
Consecuencia: las métricas de `views2` corresponden a un modelo lineal sobre features
estadísticas, no al modelo LSTM+Atención entrenado.

**8. `confusion_matrix`, `create_windows`, `split_dataset` duplicadas**
Reimplementadas en `dataset.ipynb`, `views.ipynb` y `views2.ipynb` con pequeñas
variaciones. Deberían vivir en `src/` y ser importadas por los notebooks.

---

## Estructura objetivo

```
src/
  neural_engine/
    activations.py     — sigmoid, softmax  (extraídas de lstm_cell)
    optimizer.py       — AdamOptimizer     (extraído de lstm_cell)
    lstm_cell.py       — LSTMCell          (solo la celda)
    attention.py       — Attention
    prediction_head.py — PredictionHead
    __init__.py        — expone la API pública
  preprocessing/
    __init__.py
    windows.py         — create_windows, split_dataset
    normalize.py       — interpolate_nans, minmax_normalize
  metrics/
    __init__.py
    classification.py  — confusion_matrix, precision, recall, f1, roc_auc
    anticipation.py    — calcular_anticipacion
  pipeline.py          — extract_context() canónica (la única)
```

---

## Orden de cambios recomendado

### Paso 1 — Separar utilidades (sin romper tests)

Crear `activations.py` con `sigmoid` y `softmax`.
Crear `optimizer.py` con `AdamOptimizer`.
Actualizar imports en `lstm_cell.py`, `attention.py`, `prediction_head.py`.
Verificar que los 7 tests siguen pasando.

### Paso 2 — Limpiar imports y smoke tests

Eliminar el bloque `try/except ImportError` de `attention.py` y `prediction_head.py`.
Usar solo `from neural_engine.xxx import ...`.
Eliminar los bloques `if __name__ == "__main__"` de los archivos fuente
(la cobertura ya está en `tests/`).

### Paso 3 — Exponer API en `__init__.py`

```python
from neural_engine.lstm_cell import LSTMCell
from neural_engine.attention import Attention
from neural_engine.prediction_head import PredictionHead
from neural_engine.optimizer import AdamOptimizer
from neural_engine.activations import sigmoid, softmax
```

### Paso 4 — Extraer preprocessing y métricas

Mover a `src/preprocessing/` y `src/metrics/` las funciones que hoy están
duplicadas en los notebooks. Los notebooks pasan a importarlas.

### Paso 5 — Crear `src/pipeline.py`

```python
from neural_engine import LSTMCell, Attention

HIDDEN_SIZE = 32

_lstm = LSTMCell(input_size=4, hidden_size=HIDDEN_SIZE)
_attention = Attention(hidden_size=HIDDEN_SIZE)

def extract_context(X_batch):
    """(batch, 20, 4) -> (batch, 32). Contrato de integración."""
    all_h, _ = _lstm.forward(X_batch)
    context, _ = _attention.forward(all_h)
    return context
```

### Paso 6 — Corregir `views2.ipynb`

Al exportar pesos del modelo entrenado, incluir también los pesos LSTM y Attention:
```python
np.save('pesos_modelo.npy', {
    'lstm_W': lstm.W, 'lstm_b': lstm.b,
    'att_W_a': attention.W_a,
    'det_W': detection_head.W, 'det_b': detection_head.b,
    'cls_W': classification_head.W, 'cls_b': classification_head.b,
})
```
Con eso, `views2` puede reconstruir `extract_context()` real en lugar de usar
`extraer_caracteristicas()` como proxy estadístico.

---

## Restricciones que NO cambian

- Sin Keras / PyTorch / TensorFlow / JAX. Solo numpy.
- Sin `os` ni `sys` dentro de `src/neural_engine/`.
- `math` es la única stdlib permitida dentro de `neural_engine/`.
- El contrato de integración `extract_context(X_batch)` con entrada `(batch, 20, 4)`
  y salida `(batch, 32)` no cambia.
- Los 7 tests deben seguir pasando en todo momento.

---

## Lo que NO refactorizar

- La lógica de generación de datos en `dataset.ipynb` (restricción explícita del proyecto).
- Los nombres de variables — el código ya usa nombres descriptivos.
- `DetectionHead` y `ClassificationHead` — pertenecen al compañero Jesús.
