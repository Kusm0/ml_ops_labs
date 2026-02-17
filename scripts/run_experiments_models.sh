#!/usr/bin/env bash
# Run experiments with 4 different models (RF, GBM, HistGBM, Ridge) for comparison.
# Execute from project root. Requires data/processed/dataset.csv (run preprocess first).

set -e
cd "$(dirname "$0")/.."

python scripts/run_experiments_models.py
