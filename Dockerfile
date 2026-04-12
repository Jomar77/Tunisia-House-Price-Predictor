# ── Backend-only production image (served on Railway) ────────────────────────
# The React/Vite frontend is deployed separately to Vercel and is NOT included
# in this image. The frontend communicates with this container via the Railway
# public URL using the VITE_API_URL environment variable set in Vercel.
FROM python:3.11-slim

WORKDIR /app

# Prevent .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install production Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/

# Copy model artifacts
COPY artifacts/ ./artifacts/

EXPOSE 8000

# Railway injects $PORT at runtime; fall back to 8000 for local use
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
