# New Zealand Migration Session Log

## What Changed In This Session

- Cloned the Tunisia repository into a separate workspace at `NewZealand-House-Price-Predictor` so the original project stayed untouched.
- Rebranded the clone from Tunisia to New Zealand in the main runtime, docs, package metadata, and frontend UI.
- Updated the backend startup path to prefer `nz_home_prices_model.safetensors` first, while keeping a temporary fallback to the legacy Tunisia artifact name.
- Updated API examples, health/service labels, and frontend price formatting to New Zealand conventions.
- Refreshed the visible story panels and UI copy so the app reads as a New Zealand-focused project.
- Added a New Zealand training scaffold at `Machine Learning/scripts/train_nz.py` that exports NZ-named artifacts.
- Validated the clone successfully with `pytest` and a production frontend build.

## Files Most Relevant To The Migration

- `backend/main.py`
- `backend/adapters/api/routes.py`
- `backend/adapters/api/schemas.py`
- `frontend/src/components/form/PredictionForm.tsx`
- `frontend/src/components/story/MachineLearningStory.tsx`
- `frontend/src/components/story/BackendStory.tsx`
- `frontend/src/App.css`
- `frontend/index.html`
- `README.md`
- `docs/SETUP.md`
- `Machine Learning/scripts/train_nz.py`

## Moving Forward Plan For Future Agents

1. Replace the legacy Tunisia data and model artifacts with real New Zealand training data.
2. Regenerate `columns.json`, `model_metadata.json`, and `nz_home_prices_model.safetensors` from the NZ pipeline.
3. Remove the temporary Tunisia fallback from `backend/main.py` once NZ artifacts are the only supported source of truth.
4. Update any remaining dataset/history documentation so it describes the NZ contract instead of the old Tunisia project.
5. Tighten the feature contract for the NZ market if the final dataset uses different fields than the current Tunisia-style schema.
6. Add or adjust tests for the NZ model contract, artifact loading, and metadata response.
7. Revisit the frontend copy and formatting after the NZ dataset is finalized so labels, currency, and examples match the real data.

## Validation Completed

- `pytest` in the clone: passed.
- `frontend` production build: passed.

## Notes

- The original Tunisia repository was not modified.
- The clone is NZ-focused now, but the training data itself still needs to be swapped before the project is fully NZ-complete.