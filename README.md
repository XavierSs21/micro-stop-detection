# Sistema de Detección de Micro-paradas 🏭

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![numpy](https://img.shields.io/badge/numpy-only-green.svg)]()
[![Tests Passing](https://img.shields.io/badge/tests-6%2F6%20passing-brightgreen.svg)]()

**Detección de micro-paradas en líneas de producción industrial** usando una red neuronal recurrente manual implementada **100% en numpy**.

Un proyecto fin de asignatura de Inteligencia Artificial que implementa arquitectura **LSTM + Attention** completamente desde cero, sin dependencias de frameworks de deep learning.

---

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Equipo](#equipo)
- [Arquitectura](#arquitectura)
- [Características](#características)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Setup](#setup)
- [Uso](#uso)
- [Tests](#tests)
- [Dataset](#dataset)
- [Restricciones](#restricciones)

---

## 📝 Descripción

Una micro-parada es una interrupción muy breve (< 5 segundos) en líneas de producción industrial que causa pérdida de eficiencia. Este proyecto desarrolla un sistema automático de detección basado en:

- **Datos de sensores**: vibración, corriente eléctrica, velocidad, temperatura
- **Aprendizaje temporal**: LSTM procesa secuencias de 20 pasos temporales
- **Mecanismo de atención**: resalta los pasos más relevantes
- **Predicción**: estima pasos hasta la próxima micro-parada

### ¿Por qué numpy puro?

Este es un project académico que busca:
- ✅ Entender profundamente cómo funcionan LSTMs y mecanismos de atención
- ✅ Implementar matemáticas de backpropagation manualmente
- ✅ Evitar dependencias de alto nivel (Keras, PyTorch, TensorFlow)
- ✅ Tener control total sobre cada operación

---

## 👥 Equipo

| Rol | Responsable | Módulo |
|-----|-------------|--------|
| LSTM + Attention + PredictionHead | Xavier | `src/neural_engine/` |
| DetectionHead + ClassificationHead | Jesús | `src/neural_engine/` |
| Generación de Dataset | --- | `notebook/dataset.ipynb` |
| Integración & Cuaderno | --- | `notebook/dataset.ipynb` |

---

## 🧠 Arquitectura

```
INPUT (batch=8, timesteps=20, features=4)
   ↓
┌────────────────────────────────────────┐
│         LSTM Cell (32 hidden units)    │
│  Forward + BPTT + Adam Optimizer       │
│  - Input gate, Forget gate, Cell gate  │
│  - Output gate, Cell state, Hidden     │
└────────────────────────────────────────┘
   ↓ (batch, 20, 32) — todas las hidden states
┌────────────────────────────────────────┐
│     Attention Layer (softmax)          │
│  - Puntuación por timestamp            │
│  - Suma ponderada → contexto           │
└────────────────────────────────────────┘
   ↓ (batch, 32) — contexto final
   ├─→ DetectionHead (Jesús)
   │   └─→ [0, 1] → ¿hay micro-parada?
   │
   ├─→ ClassificationHead (Jesús)
   │   └─→ [0, 9] → tipo de causa
   │
   └─→ PredictionHead (Xavier)
       └─→ [0, ∞] → pasos hasta próxima
```

### Detalles de cada módulo:

#### **LSTMCell** (`src/neural_engine/lstm_cell.py`)
- Implementación manual de LSTM
- Forward pass: procesa secuencias de (batch, T, input_size)
- Backward pass: BPTT completo sin truncamiento
- Optimizer: Adam (β₁=0.9, β₂=0.999, lr=0.001)

```python
lstm = LSTMCell(input_size=4, hidden_size=32)
all_h, h_last = lstm.forward(X_batch)  # (batch, 20, 4) → (batch, 20, 32), (batch, 32)
```

#### **Attention** (`src/neural_engine/attention.py`)
- Mecanismo softmax sobre hidden states
- Identifica timesteps más relevantes
- Salida: vector de contexto de 32 dimensiones

```python
attention = Attention(hidden_size=32)
context, alpha = attention.forward(all_h)  # (batch, 20, 32) → (batch, 32), (batch, 20)
```

#### **PredictionHead** (`src/neural_engine/prediction_head.py`)
- Regresión temporal: predice steps hasta próxima micro-parada
- Arquitectura: Linear → ReLU (de 32 → 1 unidad)
- Loss: MSE + regularización L2

```python
prediction_head = PredictionHead(hidden_size=32)
steps = prediction_head.forward(context)  # (batch, 32) → (batch,)
loss = prediction_head.backward(target)
```

#### **Contrato de integración**
```python
def extract_context(X_batch: np.ndarray) -> np.ndarray:
    """
    Input:  X_batch de shape (batch, 20, 4)
    Output: context de shape (batch, 32)
    """
    lstm = LSTMCell(input_size=4, hidden_size=32)
    attention = Attention(hidden_size=32)
    all_h, _ = lstm.forward(X_batch)
    context, _ = attention.forward(all_h)
    return context
```

---

## ✨ Características

- ✅ **LSTM manual**: implementación completa sin frameworks
- ✅ **Mecanismo de atención**: softmax sobre secuencias
- ✅ **Backpropagation Through Time (BPTT)**: gradientes completamente diferenciables
- ✅ **Adam Optimizer**: convergencia rápida
- ✅ **Numeric stability**: clipping, normalización
- ✅ **Test suite**: 6 pruebas unitarias (shapes, gradientes, contrato)
- ✅ **Notebook integrado**: demostración en Jupyter

---

## 📁 Estructura del Proyecto

```
micro-stop-detection/
├── README.md                           ← Este archivo
├── CLAUDE.md                           ← Instrucciones internas del proyecto
├── requirements.txt                    ← Dependencias (numpy, scikit-learn, etc)
│
├── src/
│   ├── __init__.py
│   └── neural_engine/
│       ├── __init__.py
│       ├── lstm_cell.py                ← LSTM + AdamOptimizer + helpers
│       ├── attention.py                ← Attention layer
│       └── prediction_head.py           ← PredictionHead (Xavier)
│       # + DetectionHead, ClassificationHead (Jesús)
│
├── tests/
│   └── test_neural_engine.py          ← 6 pytest tests
│
├── notebook/
│   └── dataset.ipynb                   ← Generación de dataset + integración
│
├── data/
│   └── industrial_dataset.csv          ← 20,000 muestras (gitignored)
│
├── weights/                            ← Pesos entrenados (gitignored)
│
└── docs/                               ← Documentación adicional
```

---

## 🚀 Setup

### Prerequisitos
- Python 3.10 o superior
- WSL2 con Debian (en Windows) o Linux nativo

### Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone <repo-url>
   cd micro-stop-detection
   ```

2. **Crear el virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install --upgrade pip
   pip install numpy scikit-learn pandas matplotlib seaborn jupyter nbformat pytest
   ```

### En WSL (Windows)

Todos los comandos deben ejecutarse con el patrón:
```bash
wsl -d Debian bash -c "PYTHONPATH=src .venv/bin/python3 <comando>"
```

---

## 📚 Uso

### Extracción de contexto (integraciones)

```python
import numpy as np
from neural_engine.lstm_cell import LSTMCell
from neural_engine.attention import Attention

# Instanciar módulos
lstm = LSTMCell(input_size=4, hidden_size=32)
attention = Attention(hidden_size=32)

# Datos simulados: 8 muestras, 20 timesteps, 4 features
X_batch = np.random.randn(8, 20, 4).astype(np.float32)

# Forward pass
all_h, _ = lstm.forward(X_batch)    # (8, 20, 32)
context, alpha = attention.forward(all_h)  # (8, 32), (8, 20)

print(f"Context shape: {context.shape}")  # (8, 32)
print(f"Attention weights shape: {alpha.shape}")  # (8, 20)
```

### Predicción de pasos jusqu'à próxima micro-parada

```python
from neural_engine.prediction_head import PredictionHead

# Instanciar cabeza de predicción
prediction_head = PredictionHead(hidden_size=32)

# Forward
predicted_steps = prediction_head.forward(context)  # (8,)
print(f"Predicted steps: {predicted_steps}")

# Backward (con target real)
target = np.array([10.5, 15.2, 8.3, 20.1, 12.5, 18.0, 9.2, 14.8])
loss = prediction_head.backward(target)
print(f"MSE Loss: {loss:.4f}")
```

---

## 🧪 Tests

### Ejecutar todas las pruebas

```bash
# Local
PYTHONPATH=src python3 -m pytest tests/test_neural_engine.py -v

# WSL
wsl -d Debian bash -c "PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_neural_engine.py -v"
```

### Pruebas incluidas (6 tests)

| Test | Validación |
|------|-----------|
| `test_lstm_forward_shapes` | Shapes correctos en forward |
| `test_attention_shapes` | Attention output de 32 dims |
| `test_prediction_head_shapes` | PredictionHead regresa escalar |
| `test_gradient_flow` | Gradientes fluyen sin NaN |
| `test_attention_sums_to_one` | Pesos attention suman 1 |
| `test_extract_context_contract` | Contrato (batch, 20, 4) → (batch, 32) |

**Salida esperada**:
```
test_neural_engine.py::test_lstm_forward_shapes PASSED
test_neural_engine.py::test_attention_shapes PASSED
test_neural_engine.py::test_prediction_head_shapes PASSED
test_neural_engine.py::test_gradient_flow PASSED
test_neural_engine.py::test_attention_sums_to_one PASSED
test_neural_engine.py::test_extract_context_contract PASSED

====== 6 passed in 0.52s ======
```

---

## 📊 Dataset

### Generación

El dataset se genera desde `notebook/dataset.ipynb`:

```python
# Simulación de sensores industriales
features = [
    "vibration",      # Hz (0-500)
    "current",        # Amperios (0-100)
    "speed",          # RPM (0-3000)
    "temperature",    # Celsius (0-100)
]
target = [
    "micro_stop",     # 0 o 1 (binario)
    "cause",          # 0-9 (categoría: rodamiento, correa, etc)
]
```

### Distribución

- **Total**: 20,000 muestras
- **Split**: 80% train, 20% test
- **Secuencias**: ventana deslizante de 20 timesteps
- **Causas**: 10 tipos (rodamiento, correa, lubricación, etc)

### Generar dataset

```bash
# Ejecutar el notebook
jupyter notebook notebook/dataset.ipynb

# O directamente (python)
wsl -d Debian bash -c "PYTHONPATH=src .venv/bin/python3 notebook/dataset.ipynb"
```

Salida: `data/industrial_dataset.csv`

---

## 🔒 Restricciones

### Prohibido ❌

- ❌ Keras, PyTorch, TensorFlow
- ❌ Cualquier framework de deep learning
- ❌ `torch.nn`, `tf.layers`, `keras.layers`
- ❌ Autograd automático

### Permitido ✅

- ✅ Numpy (operaciones matriciales)
- ✅ Scikit-learn (solo métricas: accuracy, F1, confusion_matrix)
- ✅ Pandas (carga de datos)
- ✅ Matplotlib/Seaborn (visualización)
- ✅ Jupyter (notebooks)

### Validación en tests

Cada test verifica:
```python
import torch
assert not any('torch' in str(module) for module in sys.modules)  # No PyTorch
```

---

## 🔗 Modelo de Branching (Git)

Todas las features se desarrollan en ramas feature y se integran via Pull Request:

```
main (siempre estable)
  ↑
  └─ feat/lstm-cell ─→ PR → merge ✅
  └─ feat/attention ─→ PR → merge ✅
  └─ feat/prediction-head ─→ PR → merge ✅
  └─ feat/tests ─→ PR → merge ✅
  └─ feat/notebook-integration ─→ PR → merge ✅
  └─ feat/readme ─→ PR → merge (this one!)
```

---

## 📖 Documentación

- **CLAUDE.md**: Instrucciones del proyecto y fase de ejecución
- **Docstrings**: Cada función y clase tiene docstrings en formato Google
- **Notebook**: `notebook/dataset.ipynb` incluye secciones explicativas

---

## 📞 Soporte

Para problemas:

1. Revisar `CLAUDE.md` → phase execution plan
2. Ejecutar tests: `PYTHONPATH=src python3 -m pytest tests/ -v`
3. Validar imports: `python3 -c "import sys; sys.path.insert(0, 'src'); from neural_engine.lstm_cell import LSTMCell"`

---

## 📜 Licencia

Proyecto académico. Propósito educativo.

---

**Última actualización**: Mayo 2026  
**Status**: Fase 6 (README) - En Desarrollo