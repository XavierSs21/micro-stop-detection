---
name: git-workflow
description: Git conventions and commands for the micro-stop-detection project. Use this skill before ANY git operation: committing, branching, pushing, tagging, or resolving conflicts. Always read this before touching git in this project.
---

# Git Workflow — micro-stop-detection

## Branch strategy
- `main` — always runnable, never commit directly
- `feat/lstm-cell` — done, merged
- `feat/attention` — done, merged
- `feat/prediction-head` — done, merged
- `feat/tests` — done, merged
- `feat/notebook-integration` — done, merged
- Future branches follow the same pattern: `feat/<scope>`

Always work on a feature branch. Merge to main only when tests pass.

## Commit convention (Conventional Commits)
Format: `<type>(<scope>): <short description>`

Types:
- `feat` — new code/functionality
- `fix` — bug fix
- `refactor` — restructure without changing behavior
- `test` — adding or fixing tests
- `docs` — docstrings, markdown, comments
- `chore` — dependency, config, non-code changes

Scopes for this project: `lstm`, `attention`, `prediction`, `notebook`, `tests`, `data`

Examples:
- `feat(lstm): implement LSTMCell forward pass with numpy`
- `feat(attention): add softmax attention over hidden states`
- `fix(lstm): clip gradients to [-5, 5] in BPTT`
- `test(lstm): verify loss decreases over 20 epochs`
- `docs(attention): add Google-style docstrings to Attention class`

## Standard workflow
```bash
# Start new work
git checkout main && git pull origin main
git checkout -b feat/<scope>

# During work — small atomic commits
git add src/neural_engine/lstm_cell.py
git commit -m "feat(lstm): implement forget/input/output gates"

# Push
git push -u origin feat/<scope>

# After all tests pass — merge to main
git checkout main
git merge feat/<scope> --no-ff -m "feat(<scope>): <description>"
git push origin main
```

## What to NEVER commit
- `industrial_dataset.csv` (generated, heavy)
- `*.npy` weight files (generated at runtime)
- `__pycache__/`, `.ipynb_checkpoints/`
- Any file with API keys or credentials

Always verify `.gitignore` covers these before first commit.

## .gitignore required entries
```
industrial_dataset.csv
*.npy
__pycache__/
.ipynb_checkpoints/
*.pyc
.DS_Store
data/*.csv
weights/*.npy
```
