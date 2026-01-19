# Model artifacts

This folder is intended to store exported inference artifacts used by the FastAPI backend.

Expected files:
- `columns.json` (feature contract / ordering)
- `model_metadata.json` (training metadata)
- `tunisia_home_prices_model.safetensors` (weights)

The backend will look in `artifacts/` first, and fall back to the repository root for backward compatibility.
