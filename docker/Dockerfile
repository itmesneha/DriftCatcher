# DriftCatcher Base Image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY training/ ./training/
COPY monitoring/ ./monitoring/
COPY agent/ ./agent/
COPY inference/ ./inference/

# Install dependencies
RUN uv sync --frozen

# Create necessary directories
RUN mkdir -p data/raw data/processed mlruns artifacts agent/logs monitoring

# Expose ports
EXPOSE 8000 5000

# Default command (override in docker-compose)
CMD ["uv", "run", "python", "-m", "http.server", "8000"]
