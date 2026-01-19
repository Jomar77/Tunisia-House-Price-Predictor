# Master Test Plan — Tunisia House Price Predictor

Owner: MLOps / QA

System under test: Linear Regression (scikit-learn) trained offline and exported to Safetensors, served by a FastAPI backend that re-implements inference via NumPy (`np.dot`).

Non‑negotiables (per repo constraints):
- No `pickle` / `joblib` model deserialization in production paths.
- `columns.json` is the single source of truth for feature ordering.
- Model loads once at startup (FastAPI lifespan).

Key decisions (from stakeholder):
1) Unknown `location` should map to the `other` bucket when available.
2) Parity testing uses *relative tolerance* (plus a small `atol` for near-zero stability).
3) Directional expectations are **warn-only** (do not gate releases).

---

## Phase 1 — Model Evaluation (Data Science Phase)

### Test 1.1 — Holdout metrics: $R^2$ and RMSE
- **Goal**: Prove the model has baseline predictive signal and quantify generalization.
- **Method / Code Logic**:
  - Deterministic split (`train_test_split(random_state=...)`).
  - Fit `LinearRegression` with the exact feature contract used for export (`columns.json` ordering).
  - Compute:
    - $R^2$ on train and test
    - RMSE on train and test
  - Persist metrics + dataset summary into `model_metadata.json`.
- **Success Criteria**:
  - Test RMSE is finite and better than a naive baseline (mean predictor).
  - Train/test metrics are not wildly divergent (or divergence is explained/documented).

### Test 1.2 — Residual diagnostics (visual)
- **Goal**: Detect systematic errors and data issues (nonlinearity, heteroscedasticity, location bias).
- **Method / Code Logic**:
  - Residual plots:
    - residual vs predicted
    - residual vs each numeric feature
    - residual histogram + QQ plot
  - Slice residuals by `location` buckets (including `other`).
- **Success Criteria**:
  - No strong, unexplained bias pattern; known limitations are documented as backlog items.

### Test 1.3 — Feature contract verification
- **Goal**: Prevent silent feature reordering and training/serving skew.
- **Method / Code Logic**:
  - Assert training feature list matches `columns.json:data_columns` exactly.
  - Assert required numeric features exist and are placed correctly.
  - Confirm `other` exists if training uses rare-location bucketing.
- **Success Criteria**:
  - Feature list is identical and stable; any changes require regenerating artifacts.

---

## Phase 2 — Inference Verification (Engineering Phase — Parity Testing)

Parity testing proves the backend NumPy implementation matches a scikit-learn reference to **5 decimals** using relative tolerance.

### Test 2.1 — Artifact integrity: shapes, dtype, and contract
- **Goal**: Catch export/import issues early.
- **Method / Code Logic**:
  - Load `columns.json` and assert `len(data_columns)` matches `coef.shape[0]`.
  - Assert the Safetensors file includes `coef` and `intercept`.
  - Assert intercept is scalar (shape `(1,)`).
- **Success Criteria**:
  - All assertions pass; otherwise block release.

### Test 2.2 — Parity script (`scripts/verify_export.py`)
- **Goal**: Prove `np.dot` inference equals scikit-learn `predict()` within tolerance.
- **Method / Code Logic**:
  - Load artifacts (`columns.json`, `tunisia_home_prices_model.safetensors`).
  - Build a scikit-learn `LinearRegression` object **without training** by setting:
    - `coef_`, `intercept_`, `n_features_in_`
  - Use the real production vectorizer (`backend.domain.vectorizer.FeatureVectorizer`) to build the feature vectors.
  - Compare per-case:
    - `y_numpy = np.dot(x, coef) + intercept`
    - `y_sklearn = lr.predict(x.reshape(1,-1))[0]`
  - Use `np.isclose(..., rtol=1e-5, atol=1e-8)` and report worst-case diffs.
- **Success Criteria**:
  - All cases pass `rtol`/`atol`.
  - Script exits `0` on success, non-zero on failure.

### Test 2.3 — Unknown location fallback parity (`other`)
- **Goal**: Ensure unknown locations do not break inference and map to `other`.
- **Method / Code Logic**:
  - Parity suite includes a sample with `location="__UNKNOWN_LOCATION__"`.
  - Vectorizer should set the `other` one-hot if present.
- **Success Criteria**:
  - Parity passes and the vector has `other == 1.0`.

---

## Phase 3 — Behavioral Testing (Sanity Checks — API Unit/Integration Tests)

### Test 3.1 — Strict range validation (Pydantic)
- **Goal**: Reject nonsensical inputs early and safely.
- **Method / Code Logic**:
  - POST `/api/v1/predict` with:
    - `area <= 0`, `rooms <= 0`, `bathrooms <= 0`, `age < 0`
  - Assert FastAPI responds 422 (validation).
- **Success Criteria**:
  - All invalid inputs return 422 with consistent error schema.

### Test 3.2 — Unknown location behavior (maps to `other`)
- **Goal**: Ensure inference accepts novel locations safely.
- **Method / Code Logic**:
  - POST `/api/v1/predict` using an unknown `location` string.
  - Expect a 200 response and a numeric `predicted_price`.
- **Success Criteria**:
  - 200 OK and prediction returned; no 400 for unknown location if `other` exists.

### Test 3.3 — Directional expectations (warn-only)
- **Goal**: Catch severe regressions like sign flips without blocking releases.
- **Method / Code Logic**:
  - Hold all fields constant; increase `area` and compare predictions.
  - If prediction decreases, emit `warnings.warn(...)` with input payload.
- **Success Criteria**:
  - Test suite does not fail on direction violations; warnings are visible in CI logs.

### Test 3.4 — Determinism
- **Goal**: Same input yields same output (no hidden randomness, stable artifact load).
- **Method / Code Logic**:
  - Call `/api/v1/predict` twice with the same payload.
- **Success Criteria**:
  - Outputs match exactly (or match within a tight tolerance if formatting changes).

---

## Phase 4 — Production Readiness (Stress & Monitoring)

### Test 4.1 — Load testing (K6)
- **Goal**: Validate latency and error rates under concurrency.
- **Method / Code Logic**:
  - Run a scripted load test against `/api/v1/predict` with realistic payloads.
  - Track p50/p95/p99 latency and error rates.
- **Success Criteria**:
  - Error rate below agreed threshold and p95 latency meets SLA.

### Test 4.2 — “Load once at startup” verification
- **Goal**: Ensure model is not reloaded per request.
- **Method / Code Logic**:
  - Validate logs contain one “model loaded” event per process.
- **Success Criteria**:
  - No repeated load events during steady request traffic.

### Test 4.3 — Data drift monitoring (SQL checks)
- **Goal**: Detect feature distribution shifts and prediction shifts.
- **Method / Code Logic**:
  - Persist request features + predictions to Postgres.
  - Run scheduled SQL checks comparing last 7 days vs baseline window.
- **Success Criteria**:
  - Drift report runs successfully; alerts trigger on threshold breaches.

---

## Summary Checklist (Release Gate)

| Phase | Test | Goal | Success Criteria |
|---|---|---|---|
| 1 | 1.1 | Metrics | RMSE/R² recorded; beats baseline |
| 1 | 1.2 | Residuals | No unexplained systemic patterns |
| 1 | 1.3 | Contract | Training features == `columns.json` |
| 2 | 2.1 | Artifacts | Shapes/tensors correct |
| 2 | 2.2 | Parity | `rtol=1e-5` (`atol=1e-8`) passes |
| 2 | 2.3 | Unknown location | Unknown → `other` works |
| 3 | 3.1 | Validation | 422 on invalid numerics |
| 3 | 3.2 | Location behavior | Unknown location returns 200 |
| 3 | 3.3 | Directional checks | Warn-only (no gating) |
| 3 | 3.4 | Determinism | Same input → same output |
| 4 | 4.1 | Load | SLA met; low error rate |
| 4 | 4.2 | Startup | Model loads once |
| 4 | 4.3 | Drift | SQL checks scheduled + alerts |
