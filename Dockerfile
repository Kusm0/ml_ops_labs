# MLOps Lab 1 — run training (and optional MLflow UI) inside container
FROM python:3.11-slim

# Silence MLflow warning when Git is not installed in the container
ENV GIT_PYTHON_REFRESH=quiet

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code, scripts, and data
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/

# Default: run training from project root so paths in train.py resolve correctly
WORKDIR /app
CMD ["python", "src/train.py"]
