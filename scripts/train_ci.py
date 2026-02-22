"""
CI training script for Lab 4: single classifier run producing model.pkl, metrics.json,
and confusion_matrix.png in project root for GitHub Actions and CML report.

Uses data/prepared/ (train.csv, test.csv), binary target popularity >= 50,
fixed Random Forest. Optional --max-rows for fast CI.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_COLUMNS = [
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "explicit_numeric",
    "track_genre_encoded",
]
POPULARITY_COLUMN = "popularity"
POPULARITY_THRESHOLD = 50
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "prepared"
OUTPUT_MODEL = PROJECT_ROOT / "model.pkl"
OUTPUT_METRICS = PROJECT_ROOT / "metrics.json"
OUTPUT_CM = PROJECT_ROOT / "confusion_matrix.png"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train classifier for CI; write model.pkl, metrics.json, confusion_matrix.png"
    )
    default_path = os.environ.get("DATA_PATH", str(DEFAULT_DATA_DIR))
    data_dir_default = default_path
    if default_path and Path(default_path).is_file():
        data_dir_default = str(Path(default_path).parent)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=data_dir_default,
        help="Directory with train.csv and test.csv (or DATA_PATH env)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Max rows per split in CI (e.g. 5000) for faster run",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_data(
    data_dir: str, max_rows: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train.csv and test.csv; binary target popularity >= threshold. Return X_train, y_train, X_test, y_test."""
    path = Path(data_dir)
    train_path = path / "train.csv"
    test_path = path / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {test_path}. Run prepare first (e.g. dvc repro)."
        )
    required = FEATURE_COLUMNS + [POPULARITY_COLUMN]
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    for name, df in [("train", train_df), ("test", test_df)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name}.csv missing columns: {missing}")
    if max_rows is not None:
        train_df = train_df.sample(
            min(max_rows, len(train_df)), random_state=42
        ).reset_index(drop=True)
        test_df = test_df.sample(
            min(max(max_rows // 4, 100), len(test_df)), random_state=42
        ).reset_index(drop=True)
    y_train = (train_df[POPULARITY_COLUMN].values >= POPULARITY_THRESHOLD).astype(
        np.int64
    )
    y_test = (test_df[POPULARITY_COLUMN].values >= POPULARITY_THRESHOLD).astype(
        np.int64
    )
    X_train = train_df[FEATURE_COLUMNS].values
    X_test = test_df[FEATURE_COLUMNS].values
    logger.info(
        "Loaded train=%d test=%d, binary target (popularity >= %d)",
        len(X_train),
        len(X_test),
        POPULARITY_THRESHOLD,
    )
    return X_train, y_train, X_test, y_test


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: str) -> None:
    """Save confusion matrix plot to path."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["low", "high"],
        yticklabels=["low", "high"],
        title="Confusion matrix",
        ylabel="True",
        xlabel="Predicted",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def main() -> int:
    """Train classifier, write model.pkl, metrics.json, confusion_matrix.png to project root."""
    args = parse_args()
    max_rows = args.max_rows
    if os.environ.get("CI") == "true" and max_rows is None:
        max_rows = 5000
    X_train, y_train, X_test, y_test = load_data(args.data_dir, max_rows)
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        random_state=args.random_state,
        class_weight="balanced",
    )
    logger.info("Training RandomForest...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="binary"))
    metrics = {"accuracy": accuracy, "f1": f1}
    joblib.dump(model, OUTPUT_MODEL)
    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    plot_confusion_matrix(y_test, y_pred, str(OUTPUT_CM))
    logger.info(
        "CI artifacts: model.pkl, metrics.json (f1=%.4f), confusion_matrix.png", f1
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
