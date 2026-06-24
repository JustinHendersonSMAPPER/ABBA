FROM python:3.11-slim AS base

WORKDIR /app

# System deps for SQLite and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv (copy the static binary from the official image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for layer caching, then install runtime deps into a venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Put the project venv on PATH so the ENTRYPOINT's `python -m uvicorn ...` resolves
ENV PATH="/app/.venv/bin:$PATH"

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
FROM node:22-alpine AS frontend-build

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
