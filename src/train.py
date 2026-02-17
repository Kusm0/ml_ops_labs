"""
Training script for Spotify popularity regression with MLflow tracking.

Supports multiple models: RandomForest, GradientBoosting, HistGradientBoosting, Ridge.
Loads data from data/processed/, logs params, metrics, model, and feature importance (if available).
"""

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default path relative to project root (script is in src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"

# Naming: domain_objective_stage (e.g. spotify_popularityReg_v1)
EXPERIMENT_NAME = "spotify_popularityReg_v1"

# Must match columns produced by src/preprocess.py (normalized + track_genre_encoded)
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
TARGET_COLUMN = "popularity"


MODEL_CHOICES = ("rf", "gbm", "hist_gbm", "ridge")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train regression model for Spotify popularity")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_CHOICES,
        default="rf",
        help="Model type: rf=RandomForest, gbm=GradientBoosting, hist_gbm=HistGradientBoosting, ridge=Ridge",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for test set",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees (rf, gbm, hist_gbm)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Max tree depth (rf, gbm, hist_gbm)",
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="Min samples to split node (rf, gbm)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate (gbm, hist_gbm)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength (ridge only)",
    )
    parser.add_argument(
        "--run_type",
        type=str,
        choices=("baseline", "tuned"),
        default="baseline",
        help="Run type tag: baseline or tuned",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name (visible in UI); default: model + key params",
    )
    return parser.parse_args()


def get_data_hash(data_path: str, chunk_size: int = 65536) -> str:
    """Compute SHA256 hash of dataset file for data_version (reproducibility)."""
    path = Path(data_path)
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_code_version() -> str:
    """Return git commit hash if in repo, else 'dev' (for reproducibility)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=PROJECT_ROOT,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()[:12]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "dev"


def get_model(args: argparse.Namespace):
    """Build model instance and return (model, display_name)."""
    seed = args.random_state
    if args.model == "rf":
        return (
            RandomForestRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                random_state=seed,
            ),
            "RandomForest",
        )
    if args.model == "gbm":
        return (
            GradientBoostingRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                learning_rate=args.learning_rate,
                random_state=seed,
            ),
            "GradientBoosting",
        )
    if args.model == "hist_gbm":
        return (
            HistGradientBoostingRegressor(
                max_iter=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                random_state=seed,
            ),
            "HistGradientBoosting",
        )
    if args.model == "ridge":
        return (
            Ridge(alpha=args.alpha, random_state=seed),
            "Ridge",
        )
    raise ValueError(f"Unknown model: {args.model}")


def load_and_preprocess(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed CSV and return X, y. Expects data from preprocess.py (data/processed/)."""
    logger.info("Reading CSV: %s", data_path)
    df = pd.read_csv(data_path)

    # If raw format (has 'explicit'), encode; processed data already has explicit_numeric
    if "explicit" in df.columns and "explicit_numeric" not in df.columns:
        df["explicit_numeric"] = (
            df["explicit"].astype(str).str.lower() == "true"
        ).astype(int)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    logger.info("Data loaded: n_samples=%d, n_features=%d", X.shape[0], X.shape[1])
    return X, y


def get_feature_importance(model: Any, feature_names: list[str]) -> np.ndarray | None:
    """Return importance array if model supports it (trees: feature_importances_, Ridge: |coef_|)."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "coef_"):
        return np.abs(model.coef_)
    return None


def save_feature_importance_plot(model: Any, feature_names: list[str], path: str) -> None:
    """Plot and save feature importance bar chart (trees or Ridge)."""
    import matplotlib.pyplot as plt

    importances = get_feature_importance(model, feature_names)
    if importances is None:
        return
    indices = np.argsort(importances)[::-1]  # most important first
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), importances[indices], align="center")
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
    plt.xlabel("Feature importance")
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def build_dataset_info(
    data_path: str,
    data_version: str,
    n_samples: int,
    n_features: int,
) -> dict:
    """Build dataset reference dict (path + hash, no raw data) for artifact."""
    return {
        "data_path": str(Path(data_path).resolve()),
        "data_version": data_version,
        "n_samples": n_samples,
        "n_features": n_features,
        "target": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
    }


def main() -> None:
    args = parse_args()

    model_instance, model_name = get_model(args)
    logger.info(
        "Experiment=%s model=%s run_type=%s seed=%d",
        EXPERIMENT_NAME,
        args.model,
        args.run_type,
        args.random_state,
    )
    logger.info("Loading data from %s", args.data_path)
    X, y = load_and_preprocess(args.data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    logger.info(
        "Split: train=%d test=%d (test_size=%.2f)",
        len(X_train),
        len(X_test),
        args.test_size,
    )

    data_version = get_data_hash(args.data_path)
    code_version = get_code_version()
    logger.info("data_version=%s code_version=%s", data_version, code_version)

    # Explicit tracking URI so runs are always in project mlruns/ (visible in UI)
    tracking_uri = (PROJECT_ROOT / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking_uri=%s", tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    if args.run_name:
        run_name = args.run_name
    elif args.model == "ridge":
        run_name = f"{args.model}_{args.run_type}_alpha{args.alpha}"
    else:
        run_name = f"{args.model}_{args.run_type}_depth{args.max_depth}_est{args.n_estimators}"
    with mlflow.start_run(run_name=run_name):
        # Key hyperparams + reproducibility
        mlflow.log_param("seed", args.random_state)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("model", args.model)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("code_version", code_version)
        if args.model in ("rf", "gbm", "hist_gbm"):
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
            mlflow.log_param("learning_rate", args.learning_rate)
            if args.model != "hist_gbm":
                mlflow.log_param("min_samples_split", args.min_samples_split)
        if args.model == "ridge":
            mlflow.log_param("alpha", args.alpha)

        mlflow.set_tag("run_type", args.run_type)
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("target", "popularity")

        # Dataset reference artifact (path + version + shape, no raw data)
        dataset_info = build_dataset_info(
            args.data_path, data_version, X.shape[0], X.shape[1]
        )
        dataset_info_path = PROJECT_ROOT / "dataset_info.json"
        with open(dataset_info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)
        mlflow.log_artifact(str(dataset_info_path), artifact_path="dataset")
        dataset_info_path.unlink(missing_ok=True)

        logger.info("Training %s...", model_name)
        model_instance.fit(X_train, y_train)
        model = model_instance

        metrics_log = []
        for name, X_data, y_data in [("train", X_train, y_train), ("test", X_test, y_test)]:
            pred = model.predict(X_data)
            rmse = np.sqrt(mean_squared_error(y_data, pred))
            mae = mean_absolute_error(y_data, pred)
            r2 = r2_score(y_data, pred)
            mlflow.log_metric(f"{name}_rmse", rmse)
            mlflow.log_metric(f"{name}_mae", mae)
            mlflow.log_metric(f"{name}_r2", r2)
            metrics_log.append(f"{name}_rmse={rmse:.4f} r2={r2:.4f}")
        logger.info("Metrics: %s", " | ".join(metrics_log))

        mlflow.sklearn.log_model(model, name="model")

        artifact_path = PROJECT_ROOT / "feature_importance.png"
        if get_feature_importance(model, FEATURE_COLUMNS) is not None:
            save_feature_importance_plot(model, FEATURE_COLUMNS, str(artifact_path))
            mlflow.log_artifact(str(artifact_path))
            artifact_path.unlink(missing_ok=True)

    logger.info("Run finished. View with: mlflow ui")


if __name__ == "__main__":
    main()
