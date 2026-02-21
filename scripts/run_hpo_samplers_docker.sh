#!/usr/bin/env bash
# Run Lab 3 HPO for TPE and Random samplers inside Docker.
# Execute from project root: bash scripts/run_hpo_samplers_docker.sh
# Prerequisites: data/prepared exists (run step 1 below).

set -e
N_TRIALS="${N_TRIALS:-20}"

echo "=== HPO with TPE sampler (n_trials=$N_TRIALS) ==="
docker compose run --rm train python src/optimize.py hpo.n_trials="$N_TRIALS"

echo ""
echo "=== HPO with Random sampler (n_trials=$N_TRIALS) ==="
docker compose run --rm train python src/optimize.py hpo=random hpo.n_trials="$N_TRIALS"

echo ""
echo "Done. Start MLflow UI: docker compose up mlflow-ui"
echo "Then open http://localhost:5001 and select experiment HPO_Lab3."
