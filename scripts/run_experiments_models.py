#!/usr/bin/env python3
"""
Run MLflow experiments with different models (RF, GBM, HistGBM, Ridge).

Compares RandomForest, GradientBoosting, HistGradientBoosting, Ridge on the same data.
Run from project root: python scripts/run_experiments_models.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# One run per model with sensible defaults for comparison
EXPERIMENTS = [
    {
        "model": "rf",
        "run_name": "rf_baseline",
        "run_type": "baseline",
        "n_estimators": 100,
        "max_depth": 10,
    },
    {
        "model": "gbm",
        "run_name": "gbm_baseline",
        "run_type": "baseline",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
    },
    {
        "model": "hist_gbm",
        "run_name": "hist_gbm_baseline",
        "run_type": "baseline",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
    },
    {
        "model": "ridge",
        "run_name": "ridge_baseline",
        "run_type": "baseline",
        "alpha": 1.0,
    },
]


def main() -> int:
    train_script = PROJECT_ROOT / "src" / "train.py"
    if not train_script.exists():
        print(f"Not found: {train_script}. Run from project root.", file=sys.stderr)
        return 1

    print("Running experiments with 4 models -> ./mlruns/")
    for i, cfg in enumerate(EXPERIMENTS, 1):
        run_name = cfg.pop("run_name")
        model = cfg.pop("model")
        run_type = cfg.pop("run_type")
        cmd = [
            sys.executable,
            str(train_script),
            "--model",
            model,
            "--run_type",
            run_type,
            "--run_name",
            run_name,
        ]
        for k, v in cfg.items():
            cmd.extend([f"--{k}", str(v)])
        print(f"  [{i}/4] {run_name} ({model})")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"Failed: {' '.join(cmd)}", file=sys.stderr)
            return result.returncode

    print("Done. Compare in UI: mlflow ui --backend-store-uri mlruns/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
