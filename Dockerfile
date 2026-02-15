FROM python:3.11-slim AS base

WORKDIR /app

# System deps for SQLite and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.4 && \
    poetry config virtualenvs.create false

# Copy dependency files first for layer caching
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-root --only main

# Copy application code
COPY abba/ abba/
COPY concepts.yaml* ./
COPY .env.example .env

# Create data directory
RUN mkdir -p bible_data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/ || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "abba.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

# ---------- Frontend build stage ----------
FROM node:20-alpine AS frontend-build

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---------- Full image with frontend ----------
FROM base AS full

COPY --from=frontend-build /frontend/dist /app/static

# Serve static files via FastAPI's StaticFiles mount
ENV ABBA_STATIC_DIR=/app/static
