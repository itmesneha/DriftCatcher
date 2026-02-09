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

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --python 3.12

# Copy all necessary code and config
COPY agent/ ./agent/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY monitoring/ ./monitoring/
COPY training/ ./training/
COPY preprocessing/ ./preprocessing/
COPY config/ ./config/
COPY tools/ ./tools/
COPY artifacts/ ./artifacts/
COPY mlruns/ ./mlruns/
COPY main.py ./main.py
COPY drift_feature_psi.json ./drift_feature_psi.json
COPY sample_baseline.py ./sample_baseline.py
COPY regenerate_stats.py ./regenerate_stats.py
COPY README.md ./README.md
COPY docs/ ./docs/
COPY screenshots/ ./screenshots/
RUN mkdir -p data/uploads data/processed agent/logs monitoring/logs

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI with uvicorn
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
