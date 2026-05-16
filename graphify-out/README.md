# graphify-out/

Carpeta generada por la herramienta **graphify**. Contiene el grafo de conocimiento
del proyecto `micro-stop-detection` y los archivos de estado del proceso de construcción.

## Estructura

### outputs/
Entregables principales del grafo:
- `graph.json` — Grafo completo en formato NetworkX JSON (78 nodos, 135 aristas, 10 comunidades)
- `graph.html` — Visualización interactiva del grafo (abrir en navegador)
- `GRAPH_TREE.html` — Visualización alternativa en forma de árbol

### reports/
Informes legibles generados tras cada build:
- `GRAPH_REPORT.md` — Resumen del grafo: nodos "dios", comunidades, gaps de conocimiento y preguntas sugeridas

### Archivos de estado (raíz — no mover)
Usados internamente por graphify para builds incrementales:

| Archivo | Propósito |
|---|---|
| `manifest.json` | Registro de archivos fuente procesados (MD5 + mtime) |
| `cost.json` | Historial de tokens usados por run |
| `.graphify_labels.json` | Nombres de las 10 comunidades detectadas |
| `.graphify_root` | Directorio raíz del proyecto (contiene: `src`) |
| `.graphify_python` | Ruta al intérprete Python de graphify |
| `.rebuild.lock` | Timestamp del último build completo |

### cache/
Caché de extracción para builds incrementales (regenerable, se puede borrar sin perder datos):
- `cache/semantic/` — Extracción semántica de imágenes PNG (9 archivos)
- `cache/ast/` — ASTs de archivos Python en `src/` (10 archivos)
