# Repo cleanup (reduce root clutter)

The codebase now *prefers* loading inference assets from `artifacts/` and keeps tooling under `scripts/`.

If you want the repo root to be minimal, you can run these moves locally (recommended: use `git mv` so history is preserved).

## Suggested moves

### Docs

```bash
git mv IMPLEMENTATION_SUMMARY.md docs/IMPLEMENTATION_SUMMARY.md
git mv MASTER_TEST_PLAN.md docs/MASTER_TEST_PLAN.md
git mv SETUP.md docs/SETUP.md
```

### Notebook

```bash
git mv main.ipynb notebooks/main.ipynb
```

### Model artifacts

```bash
git mv columns.json artifacts/columns.json
git mv model_metadata.json artifacts/model_metadata.json
git mv tunisia_home_prices_model.safetensors artifacts/tunisia_home_prices_model.safetensors
```

### Data & images (optional)

```bash
mkdir -p data images
# Windows PowerShell equivalents may differ

git mv data.csv data/data.csv
git mv heatmap.png images/heatmap.png
```

## Notes

- Backend: loads `artifacts/*` first; falls back to repo root if missing.
- Tests: vectorizer test prefers `artifacts/columns.json`.
- Tooling: parity verifier is `scripts/verify_export.py`.
