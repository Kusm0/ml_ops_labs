# MLOps Lab 5 — multi-stage build: minimal final image for ML/DVC/MLflow
# Stage 1: install heavy Python dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: minimal runtime image
FROM python:3.11-slim

# Git needed for DVC (dvc repro, git add/commit from container) and MLflow run names
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
ENV GIT_PYTHON_REFRESH=quiet

WORKDIR /app

# Copy only installed packages from builder (no build tools in final image)
COPY --from=builder /root/.local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /root/.local/bin /usr/local/bin
ENV PATH="/usr/local/bin:${PATH}"

# Copy project code, scripts, config (data/ and .dvc/ typically mounted at run time for DVC)
COPY src/ src/
COPY scripts/ scripts/
COPY config/ config/
COPY dvc.yaml ./
COPY .dvc/ .dvc/
COPY data/ data/

WORKDIR /app
CMD ["python", "src/train.py"]
