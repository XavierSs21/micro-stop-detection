---
name: jupyter-notebook
description: Conventions for editing and running the project Jupyter notebook. Use before inserting cells, running the notebook end-to-end, or verifying outputs. Critical for the final deliverable since professor requires all cells executed.
---

# Jupyter Notebook — micro-stop-detection

## Notebook path
`notebook/dataset.ipynb` — this is the single source of truth for the final deliverable.

## Section structure (mandatory order)
The notebook has 7 sections. Do NOT reorder them:
1. Generacion del dataset (Pedro)
2. Preprocesamiento y limpieza (Aaly)
3. **Arquitectura del modelo — LSTM + Atencion (Xavi)** <- your section
4. Entrenamiento — Deteccion y Clasificacion (Jesus)
5. Prediccion temporal (Xavi + Jesus)
6. Resultados y visualizaciones (Yess + Aaly)
7. Impacto y conclusiones (Yess + Victor)

## Xavi's section structure (Section 3)
Insert these cells in order after Section 2 ends:

```
[markdown] ## Seccion 3: Arquitectura del Modelo — LSTM + Atencion (Xavi)
[markdown] ### 3.1 Justificacion: LSTM sobre RNN simple
[code]     # imports
[markdown] ### 3.2 LSTMCell — Implementacion manual
[code]     # LSTMCell class (full implementation)
[markdown] ### 3.3 Capa de Atencion
[code]     # Attention class
[markdown] ### 3.4 PredictionHead — Modulo C
[code]     # PredictionHead class
[markdown] ### 3.5 Integracion con el equipo
[code]     # extract_context() replacement + quick test
[code]     # Gradient flow test: assert loss[-1] < loss[0]
```

## How to programmatically add cells (nbformat)
```python
import nbformat

nb = nbformat.read('notebook/dataset.ipynb', as_version=4)

new_cell = nbformat.v4.new_markdown_cell("## Seccion 3: ...")
nb.cells.append(new_cell)

code_cell = nbformat.v4.new_code_cell("# your code here")
nb.cells.append(code_cell)

nbformat.write(nb, 'notebook/dataset.ipynb')
```

## Running all cells (final deliverable check)
```bash
# Execute all cells in-place (saves outputs into the notebook)
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 \
  notebook/dataset.ipynb \
  --output notebook/dataset.ipynb
```
This is what Pedro needs for the final submission. All outputs must be visible.

## Critical rules
- ALL imports at the top of each section's first code cell
- Use relative paths: `"data/industrial_dataset.csv"`, `"weights/pesos_modelo.npy"`
- No `plt.show()` — use `plt.savefig()` and `display(fig)` for output visibility
- All text (titles, labels, legends) in Spanish
- Do NOT clear outputs before committing the final version
