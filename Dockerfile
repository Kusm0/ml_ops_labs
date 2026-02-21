# MLOps Lab 1 & 2 — run training and DVC pipeline inside container
FROM python:3.11-slim

# Git needed for DVC (dvc repro, git add/commit from container) and MLflow run names
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ENV GIT_PYTHON_REFRESH=quiet

WORKDIR /app

# Install dependencies (includes DVC)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code, scripts, config (data/ and .dvc/ typically mounted at run time for DVC)
COPY src/ src/
COPY scripts/ scripts/
COPY config/ config/
COPY dvc.yaml ./
COPY .dvc/ .dvc/
COPY data/ data/

# Default: run training from project root so paths in train.py resolve correctly
WORKDIR /app
CMD ["python", "src/train.py"]
