# Dockerfile
# Multi-stage build for DriftCatcher microservice

FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies with Python 3.12 explicitly
RUN uv sync --frozen --no-dev --python 3.12

# Copy application code (only what exists)
COPY agent/ ./agent/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY monitoring/ ./monitoring/
COPY training/ ./training/
COPY preprocessing/ ./preprocessing/
COPY config/ ./config/
COPY tools/ ./tools/

# Create necessary directories
RUN mkdir -p artifacts data/uploads data/processed mlruns agent/logs monitoring/logs

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI with uvicorn
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
