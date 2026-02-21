#!/usr/bin/env bash
# Run HPO for TPE and Random samplers (Lab 3, step 8: compare samplers).
# Execute from project root: bash scripts/run_hpo_samplers.sh
# Ensure data/prepared exists (dvc repro) and deps are installed (pip install -r requirements.txt).

set -e
cd "$(dirname "$0")/.."
N_TRIALS="${N_TRIALS:-20}"

echo "=== HPO with TPE sampler (n_trials=$N_TRIALS) ==="
python src/optimize.py hpo.n_trials="$N_TRIALS"

echo ""
echo "=== HPO with Random sampler (n_trials=$N_TRIALS) ==="
python src/optimize.py hpo=random hpo.n_trials="$N_TRIALS"

echo ""
echo "Done. Compare in MLflow UI: mlflow ui (then open experiment HPO_Lab3)."
