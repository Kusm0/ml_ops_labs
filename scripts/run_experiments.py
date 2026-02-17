#!/usr/bin/env python3
"""
Run 5 MLflow experiments with different hyperparameters.

Uses the same tracking URI as train.py (project mlruns/), so runs are visible
in MLflow UI (mlflow ui --backend-store-uri mlruns/ or docker compose up mlflow-ui).
Run from project root: python scripts/run_experiments.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS = [
    {"run_type": "baseline", "run_name": "baseline_depth10_est100"},
    {
        "run_type": "tuned",
        "run_name": "tuned_depth5_est50",
        "max_depth": 5,
        "n_estimators": 50,
    },
    {
        "run_type": "tuned",
        "run_name": "tuned_depth15_est150",
        "max_depth": 15,
        "n_estimators": 150,
    },
    {
        "run_type": "tuned",
        "run_name": "tuned_depth20_min_split5",
        "max_depth": 20,
        "min_samples_split": 5,
        "n_estimators": 100,
    },
    {
        "run_type": "tuned",
        "run_name": "tuned_depth8_est200_min_split4",
        "max_depth": 8,
        "n_estimators": 200,
        "min_samples_split": 4,
    },
]


def main() -> int:
    train_script = PROJECT_ROOT / "src" / "train.py"
    if not train_script.exists():
        print(f"Not found: {train_script}. Run from project root.", file=sys.stderr)
        return 1

    print("Running 5 experiments (spotify_popularityReg_v1) -> ./mlruns/")
    for i, cfg in enumerate(EXPERIMENTS, 1):
        run_name = cfg.pop("run_name")
        run_type = cfg.pop("run_type")
        cmd = [
            sys.executable,
            str(train_script),
            "--run_type",
            run_type,
            "--run_name",
            run_name,
        ]
        for k, v in cfg.items():
            cmd.extend([f"--{k}", str(v)])
        print(f"  [{i}/5] {run_name}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"Failed: {' '.join(cmd)}", file=sys.stderr)
            return result.returncode

    print("Done. View in UI: mlflow ui --backend-store-uri mlruns/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
