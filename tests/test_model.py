"""Post-train tests: artifacts existence and Quality Gate (Lab 4)."""

import json
import os
from pathlib import Path

from conftest import get_artifact_dir


def test_artifacts_exist() -> None:
    """Check that model.pkl, metrics.json, and confusion_matrix.png exist after training."""
    root = get_artifact_dir()
    assert (root / "model.pkl").exists(), "model.pkl not found"
    assert (root / "metrics.json").exists(), "metrics.json not found"
    assert (root / "confusion_matrix.png").exists(), "confusion_matrix.png not found"


def test_quality_gate_f1() -> None:
    """Quality Gate: F1 must be >= threshold (default 0.70)."""
    threshold = float(os.environ.get("F1_THRESHOLD", "0.70"))
    root = get_artifact_dir()
    metrics_path = root / "metrics.json"
    assert metrics_path.exists(), "metrics.json not found (run train_ci first)"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    f1 = float(metrics["f1"])
    assert f1 >= threshold, (
        f"Quality Gate not passed: f1={f1:.4f} < {threshold:.2f}"
    )
