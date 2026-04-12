# ── Stage 1: Build the React/Vite frontend ─────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app

# Install dependencies (cache layer)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy source and build
COPY frontend/ ./
RUN npm run build


# ── Stage 2: Production Python image ────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Prevent .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install production Python dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy backend source
COPY backend/ ./backend/

# Copy model artifacts
COPY artifacts/ ./artifacts/

# Copy built frontend bundle
COPY --from=frontend-build /app/dist ./frontend/dist

EXPOSE 8000

# Railway injects $PORT at runtime; fall back to 8000 for local use
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
