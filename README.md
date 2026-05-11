# Sistema de Deteccion de Micro-paradas

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![numpy](https://img.shields.io/badge/numpy-only-green.svg)]()
[![Tests](https://img.shields.io/badge/tests-7%2F7%20passing-brightgreen.svg)]()

Deteccion de micro-paradas en lineas de produccion industrial mediante una red neuronal recurrente implementada **100% en numpy**, sin frameworks de deep learning.

---

## Descripcion

Una micro-parada es una interrupcion breve (menos de 5 segundos) en una linea de produccion que reduce la eficiencia sin activar alarmas convencionales. El sistema aprende patrones temporales de cuatro sensores industriales para detectar, clasificar y predecir estas interrupciones.

**Entradas**: vibracion (Hz), corriente (A), velocidad (RPM), temperatura (grados C)

**Salidas**:
- Deteccion binaria: hay o no hay micro-parada
- Clasificacion de causa: 10 tipos de falla posibles
- Prediccion temporal: pasos hasta la proxima micro-parada

### Por que numpy puro

Proyecto de la asignatura de Inteligencia Artificial. El objetivo es implementar backpropagation y Adam desde cero para entender cada operacion matematica involucrada, sin depender de autograd ni capas preentrenadas.

---

## Arquitectura

```
Entrada (batch, 20, 4)   -- 20 pasos temporales, 4 sensores
        |
   [LSTMCell]            -- procesa la secuencia completa (batch, 20, 32)
        |
   [Attention]           -- pondera timesteps relevantes  (batch, 32)
        |
   -----+------------------+---------------------+
        |                  |                     |
[DetectionHead]  [ClassificationHead]   [PredictionHead]
  binario (0/1)    causa (0-9)           pasos hasta proxima
```

El vector de contexto `(batch, 32)` es el punto de integracion entre la parte secuencial y las cabezas de salida.

**Contrato de integracion** -- no romper:

```python
def extract_context(X_batch: np.ndarray) -> np.ndarray:
    # Entrada: (batch, 20, 4)  ->  Salida: (batch, 32)
    all_h, _ = lstm.forward(X_batch)
    context, _ = attention.forward(all_h)
    return context
```

---

## Modulos

### LSTMCell

Archivo: `src/neural_engine/lstm_cell.py`

LSTM de una capa con matriz de pesos concatenada `(fan_in, 4H)`. Implementa forward completo, BPTT sin truncamiento y optimizador Adam propio.

```python
lstm = LSTMCell(input_size=4, hidden_size=32)
all_h, h_last = lstm.forward(X)    # (batch, 20, 4) -> (batch, 20, 32)
grads = lstm.backward(dL_dall_h)   # (batch, 20, 32) -> dict con W y b
lstm.update(grads, optimizer)
```

### Attention

Archivo: `src/neural_engine/attention.py`

Atencion softmax sobre los hidden states del LSTM. Calcula energias con `tanh(all_h @ W_a.T)`, normaliza con softmax y produce un vector de contexto por muestra. Incluye backward completo: gradientes hacia `all_h` y actualizacion de `W_a` via Adam.

```python
att = Attention(hidden_size=32)
context, alpha = att.forward(all_h)      # (batch, 20, 32) -> (batch, 32), (batch, 20)
dL_dall_h = att.backward(dL_dcontext)   # (batch, 32) -> (batch, 20, 32)
```

`alpha` suma 1 a lo largo del eje temporal. `att.last_alpha` queda disponible para visualizacion.

### PredictionHead

Archivo: `src/neural_engine/prediction_head.py`

Cabeza de regresion temporal: capa lineal seguida de ReLU, perdida MSE. El metodo `backward` actualiza los pesos y retorna el gradiente hacia el contexto para continuar la cadena.

```python
ph = PredictionHead(hidden_size=32)
pred = ph.forward(context)               # (batch, 32) -> (batch,)
loss = ph.loss(pred, y_time)             # escalar MSE
dL_dcontext = ph.backward(pred, y_time) # actualiza pesos, retorna (batch, 32)
```

### AdamOptimizer

Archivo: `src/neural_engine/lstm_cell.py`

Implementacion de Adam (beta1=0.9, beta2=0.999, lr=1e-3) reutilizable por cualquier modulo del proyecto.

---

## Pipeline de entrenamiento

El flujo completo conecta los tres modulos en un ciclo forward-backward:

```python
opt = AdamOptimizer(lr=1e-3)

for epoch in range(n_epochs):
    # forward
    all_h, _   = lstm.forward(X)
    context, _ = att.forward(all_h)
    pred       = ph.forward(context)
    loss       = ph.loss(pred, y_time)

    # backward -- gradientes fluyen de salida a entrada
    dL_dcontext = ph.backward(pred, y_time)
    dL_dall_h   = att.backward(dL_dcontext)
    grads       = lstm.backward(dL_dall_h)
    lstm.update(grads, opt)
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / WSL
# .venv\Scripts\activate         # Windows

pip install numpy scikit-learn pandas matplotlib seaborn jupyter nbformat pytest
```

---

## Tests

```bash
PYTHONPATH=src python3 -m pytest tests/test_neural_engine.py -v
```

| Test | Que verifica |
|------|-------------|
| `test_lstm_forward_shapes` | Shapes correctos en el forward pass |
| `test_lstm_gradient_flow` | La perdida decrece en 30 iteraciones |
| `test_attention_weights_sum_to_one` | alpha suma 1 en el eje temporal |
| `test_prediction_head_gradient_flow` | La perdida decrece en 20 iteraciones |
| `test_extract_context_contract` | Entrada (8,20,4) produce contexto (8,32) sin NaN |
| `test_no_framework_imports` | Ningun archivo en src/ importa torch/keras/jax |
| `test_end_to_end_training` | Pipeline completo: la perdida decrece en 10 iteraciones |

---

## Dataset

Generado por `notebook/dataset.ipynb`. No modificar la logica de generacion.

```bash
jupyter notebook notebook/dataset.ipynb
# Ejecutar todas las celdas -> genera data/industrial_dataset.csv
```

| Columna | Descripcion |
|---------|-------------|
| vibration | Vibracion en Hz (0-500) |
| current | Corriente en A (0-100) |
| speed | Velocidad en RPM (0-3000) |
| temperature | Temperatura en grados C (0-100) |
| micro_stop | Etiqueta binaria (0 o 1) |
| cause | Tipo de falla (0-9) |

20 000 muestras -- ventana deslizante de 20 timesteps -- 80/20 train/test.

---

## Restricciones

| Permitido | Prohibido |
|-----------|-----------|
| `numpy` -- operaciones matriciales | Keras, PyTorch, TensorFlow, JAX |
| `math` -- funciones matematicas escalares | Autograd automatico de cualquier framework |
| `sklearn` -- solo metricas (accuracy, F1, confusion_matrix) | `os` y `sys` dentro del codigo de red neuronal |
| `pandas` -- carga de datos | |
| `matplotlib` / `seaborn` -- visualizacion | |

Los tests verifican automaticamente que ningun archivo en `src/neural_engine/` importe frameworks prohibidos.

---

Proyecto academico -- Inteligencia Artificial -- Mayo 2026
