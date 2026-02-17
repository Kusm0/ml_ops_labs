#!/usr/bin/env bash
# Run 5 experiments with different hyperparameters (baseline + 4 tuned).
# Runs are written to ./mlruns/ and visible in MLflow UI (local or Docker with volume).
# Execute from project root: bash scripts/run_experiments.sh
# Requires: data/raw/dataset.csv, venv activated.

set -e
cd "$(dirname "$0")/.."

echo "Running 5 experiments (spotify_popularityReg_v1) -> ./mlruns/"

# 1. Baseline
python src/train.py --run_type baseline --run_name "baseline_depth10_est100"

# 2. Tuned: shallower trees, fewer estimators
python src/train.py --run_type tuned --max_depth 5 --n_estimators 50 --run_name "tuned_depth5_est50"

# 3. Tuned: deeper trees, more estimators
python src/train.py --run_type tuned --max_depth 15 --n_estimators 150 --run_name "tuned_depth15_est150"

# 4. Tuned: high depth, higher min_samples_split
python src/train.py --run_type tuned --max_depth 20 --min_samples_split 5 --n_estimators 100 --run_name "tuned_depth20_min_split5"

# 5. Tuned: medium depth, more trees
python src/train.py --run_type tuned --max_depth 8 --n_estimators 200 --min_samples_split 4 --run_name "tuned_depth8_est200_min_split4"

echo "Done. View in UI: mlflow ui --backend-store-uri mlruns/  (or docker compose up mlflow-ui)"
