---
name: wsl-linux
description: WSL Debian environment commands and conventions for this project. Use before running any bash command, creating directories, managing paths, or installing packages. Critical for avoiding Windows/Linux path confusion.
---

# WSL Debian — Environment Guide

## Path conventions
- Project root: always use relative paths from repo root, never hardcode Windows paths
- Home: `~` = `/home/<user>` inside WSL
- Skills: `~/.claude/skills/micro-stop-detection/`
- Dataset output: `data/industrial_dataset.csv` (relative to repo root)
- Weights output: `weights/pesos_modelo.npy` (relative to repo root)

## Python environment
Claude Code's Bash tool runs on Windows. Always prefix every command with:
```
wsl -d Debian bash -c "..."
```
Never use bare `python3`, `pip3`, or `pytest` directly.

## Command patterns — no exceptions
Python execution:
```bash
wsl -d Debian bash -c "PYTHONPATH=src .venv/bin/python3 <file>"
```
Pytest:
```bash
wsl -d Debian bash -c "PYTHONPATH=src .venv/bin/python3 -m pytest tests/ -v"
```
Pip install:
```bash
wsl -d Debian bash -c ".venv/bin/pip install <package>"
```
Git:
```bash
wsl -d Debian bash -c "git <command>"
```
Jupyter execute:
```bash
wsl -d Debian bash -c ".venv/bin/jupyter nbconvert --to notebook --execute notebook/dataset.ipynb --output notebook/dataset.ipynb"
```

## Virtual environment
Located at `.venv/` in repo root (gitignored).
Created with: `wsl -d Debian bash -c "python3 -m venv .venv"`
Dependencies: `numpy scikit-learn pandas matplotlib seaborn jupyter nbformat pytest`

## Directory creation
```bash
wsl -d Debian bash -c "mkdir -p src/neural_engine tests weights data docs"
wsl -d Debian bash -c "touch src/__init__.py src/neural_engine/__init__.py"
wsl -d Debian bash -c "touch weights/.gitkeep data/.gitkeep"
```

## File permissions / encoding
- Always use UTF-8 for Python files
- Notebooks: save as UTF-8, no BOM
- Line endings: LF (Linux), never CRLF — git should handle this automatically

## Common WSL gotchas
- If `jupyter` not found: `python3 -m jupyter notebook`
- If module not found: check you're running from repo root, not a subdirectory
- If path issues with numpy savetxt: use `os.path.join` or pathlib, never hardcoded `/mnt/c/...`
