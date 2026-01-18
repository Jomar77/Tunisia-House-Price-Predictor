# Tunisia House Price Predictor — AI Agent Instructions

These instructions guide changes across:
- The existing ML workflow (scraping → training in notebook → exporting artifacts)
- The target production system described in `.github/Architecture/architecture.instructions.md` and `.github/instructions/architecture-blueprint-generator.prompt.md`

If there is a conflict, follow this priority:
1) Safety + security requirements (no pickle; avoid arbitrary code execution paths)
2) Architecture blueprint constraints (React/Vite + FastAPI + Safetensors + Hexagonal)
3) Existing repo conventions (notebook sequencing, artifact formats)

## 1) Project Goal
Predict house prices in Tunisia from a small set of features (area/rooms/bathrooms/age + location) using a Linear Regression model, and expose predictions via an API consumed by a web UI.

## 2) Current Repository Reality (Today)
This repo currently contains:
- Data scraping script producing raw listing data
- A notebook training pipeline using scikit-learn
- Exported inference assets (`columns.json` and a Safetensors weight file if present)

Keep notebook execution deterministic (no hidden state), and keep inference compatibility with `columns.json`.

## 3) Target Production Architecture (Planned)
The architecture to converge towards:
- **Frontend**: React + Vite SPA
- **Backend**: FastAPI (async), serving both API and static frontend assets (single container)
- **Model artifacts**: Safetensors + JSON metadata (no Pickle)
- **Architecture pattern**: Hexagonal (Ports & Adapters)
- **Deployment**: Containerized, serverless (Google Cloud Run)
- **Database**: Postgres (Neon recommended), used primarily for logging predictions/feedback

The system should remain stateless at the HTTP layer; any persistent data (logs/feedback/config) goes to Postgres.

## 4) Non-Negotiable Constraints
- **Do not use Pickle/joblib for model persistence** in production paths; use Safetensors + explicit metadata.
- **`columns.json` is the source of truth** for feature ordering during inference.
- **Model loads once at startup** (FastAPI lifespan), not per request.
- Prefer **async + background tasks** for non-critical path DB writes to keep inference latency predictable.

## 5) Domain Inputs & Feature Contract
The canonical input features (inference contract) are:
- Numeric: `area`, `rooms`, `bathrooms`, `age`
- Categorical: `location`

Inference feature vector rules:
1) Load feature names from `columns.json`.
2) Initialize a zero vector of length `len(columns)`.
3) Set numeric features at their defined indices.
4) One-hot set the location column to `1` if present; otherwise use the `other`/fallback location if defined.

Never silently reorder features; if the input schema changes, regenerate `columns.json` and keep backward compatibility at the API layer.

## 6) Hexagonal Architecture Guidance (Ports & Adapters)
When adding backend application code, keep boundaries clear:

**Core (Domain / Application)**
- Pure Python logic for feature vectorization and prediction orchestration
- Interfaces (ports) for:
    - Model inference
    - Persistence/logging
    - Configuration (supported locations, etc.)

**Primary Adapters (Driving)**
- FastAPI routes + Pydantic request/response models

**Secondary Adapters (Driven)**
- Safetensors-based model loader/inference adapter
- Postgres adapter (SQLAlchemy/AsyncPG) implementing repository ports

Avoid importing FastAPI/SQLAlchemy into the domain layer.

## 7) Backend API Expectations (When Implemented)
When adding endpoints, follow these conventions:
- `POST /api/predict`:
    - Request: `area, rooms, bathrooms, age, location`
    - Response: predicted price + optional explanation fields
- `POST /api/feedback`:
    - Stores user feedback linked to a prediction id (write via background task)
- `GET /api/metadata`:
    - Returns supported locations and model metadata (e.g., training date, feature list)

Input validation must be strict (Pydantic) and error responses should be consistent and user-safe.

## 8) Frontend Architecture Expectations (When Added)
- Vite + React SPA
- Prefer TanStack Query (React Query) for server state (predictions/metadata)
- Use React Context for UI-only state (theme, wizard step)
- If the location list is large, use virtualization (e.g., `react-window`) inside a combobox-style selector

## 9) Persistence & Database Logging
Postgres is for:
- Prediction request/response logging
- User feedback
- Optional dynamic configuration (supported locations/zip codes)

Guidelines:
- Writes should not block inference; use FastAPI BackgroundTasks.
- For Cloud Run + Neon: connect to a pooling endpoint (often port `6432`) and tune pool sizes modestly.

## 10) Model Training & Artifact Export Rules
Training lives in the notebook pipeline.

Model selection:
- Prefer Linear Regression as the baseline.
- You may compare Lasso/DecisionTree via `GridSearchCV` but keep the exported inference contract stable.

Artifact export:
- Save weights with `safetensors.numpy.save_file` (or equivalent safe API).
- Save columns/features to `columns.json`.
- If you add metadata, prefer `model_metadata.json` (explicit, versioned schema).

## 11) Outlier Handling & Data Cleaning
Use domain-specific heuristics (per-location rules, price-per-area sanity checks, room size thresholds) rather than only global thresholds.

Any new cleaning rule must:
- Be deterministic
- Be documented in the notebook
- Not leak target information into features

## 12) Testing & Quality Gates (As the System Grows)
If/when tests are introduced:
- Backend: `pytest` unit tests for feature vectorization and model adapter; integration tests can mock DB
- Lint: `ruff check`
- Type check: `mypy` where feasible

Do not add large frameworks if unnecessary; keep changes minimal and aligned with the repo’s scope.

## 13) Security & Operational Notes
- Never deserialize untrusted content.
- Treat model files as read-only runtime assets.
- Keep secrets out of the repo; use environment variables/secret managers.
- Log safely (avoid logging full payloads if they can contain sensitive data).

## 14) How to Extend This Repo Safely
When adding new features:
1) Update the notebook/export artifacts first if the model inputs change.
2) Update backend domain vectorization to match `columns.json`.
3) Update API schema with backward compatibility.
4) Update frontend to use `/api/metadata` to populate selectors.

Keep architecture documentation and Copilot instructions in sync with any major refactor.
