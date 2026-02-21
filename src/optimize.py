"""
HPO (Hyperparameter Optimization) with Optuna, Hydra, and MLflow nested runs.

Lab 3: Loads data from data/prepared/, creates binary target from popularity
(popularity >= threshold), optimizes Random Forest or Logistic Regression,
logs each trial as nested MLflow run, retrains best model and saves to models/.
"""

import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import joblib
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
import numpy as np
import optuna
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Must match columns from src/prepare.py (data/prepared output)
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


def set_global_seed(seed: int) -> None:
    """Fix seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_code_version() -> str:
    """Return git commit hash if in repo, else 'dev' (for reproducibility, Lab 3 sect. 0)."""
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


def load_prepared_data(
    prepared_dir: str,
    target_type: str = "quartiles",
    popularity_threshold: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train.csv and test.csv from prepared_dir.

    target_type:
      - "quartiles": 4 classes (0=Q1, 1=Q2, 2=Q3, 3=Q4) by popularity quartiles on train.
      - "binary": target = (popularity >= popularity_threshold).
    Returns X_train, X_test, y_train, y_test as numpy arrays.
    """
    abs_dir = to_absolute_path(prepared_dir)
    train_path = Path(abs_dir) / "train.csv"
    test_path = Path(abs_dir) / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {test_path}. Run DVC prepare stage first."
        )

    required = FEATURE_COLUMNS + [POPULARITY_COLUMN]
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    for name, df in [("train", train_df), ("test", test_df)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name}.csv missing columns: {missing}")

    pop_train = train_df[POPULARITY_COLUMN].values
    pop_test = test_df[POPULARITY_COLUMN].values

    if target_type == "quartiles":
        q25, q50, q75 = np.percentile(pop_train, [25, 50, 75])
        # classes 0, 1, 2, 3: [0,q25), [q25,q50), [q50,q75), [q75,100]
        def to_quartile(pop: np.ndarray) -> np.ndarray:
            y = np.zeros(len(pop), dtype=np.int64)
            y[pop >= q25] = 1
            y[pop >= q50] = 2
            y[pop >= q75] = 3
            return y
        y_train = to_quartile(pop_train)
        y_test = to_quartile(pop_test)
        logger.info(
            "Quartile boundaries (from train): Q25=%.1f Q50=%.1f Q75=%.1f",
            q25, q50, q75,
        )
    else:
        y_train = (pop_train >= popularity_threshold).astype(np.int64)
        y_test = (pop_test >= popularity_threshold).astype(np.int64)

    X_train = train_df[FEATURE_COLUMNS].values
    X_test = test_df[FEATURE_COLUMNS].values

    n_train, n_test = len(X_train), len(X_test)
    n_classes = len(np.unique(y_train))
    if n_classes <= 2:
        logger.info(
            "Loaded prepared data: train=%d test=%d, binary target: train %.2f%% positive",
            n_train, n_test, 100 * y_train.mean(),
        )
    else:
        train_pct = [100 * (y_train == k).mean() for k in range(n_classes)]
        logger.info(
            "Loaded prepared data: train=%d test=%d, %d classes (train %%): %s",
            n_train, n_test, n_classes, " ".join(f"Q{k+1}={train_pct[k]:.1f}%%" for k in range(n_classes)),
        )

    min_train, min_test = 50, 10
    if n_train < min_train or n_test < min_test:
        raise ValueError(
            f"Too few samples for HPO: train={n_train}, test={n_test}. "
            f"Need at least train>={min_train}, test>={min_test}. "
            "Ensure full data: run 'dvc pull' then 'dvc repro', or prepare from a full data/raw/dataset.csv."
        )
    return X_train, X_test, y_train, y_test


def build_model(model_type: str, params: dict[str, Any], seed: int) -> Any:
    """Build classifier from model_type and hyperparameters.
    Uses class_weight='balanced' for imbalanced target (e.g. popularity >= 50).
    """
    if model_type == "random_forest":
        return RandomForestClassifier(
            random_state=seed, n_jobs=-1, class_weight="balanced", **params
        )
    if model_type == "logistic_regression":
        clf = LogisticRegression(
            random_state=seed, max_iter=500, class_weight="balanced", **params
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    raise ValueError(
        f"Unknown model.type='{model_type}'. Expecting 'random_forest' or 'logistic_regression'."
    )


def evaluate(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric: str,
) -> float:
    """Train model and return metric on test set."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if metric == "f1":
        avg = "binary" if len(np.unique(y_test)) == 2 else "weighted"
        return float(f1_score(y_test, y_pred, average=avg))
    if metric == "roc_auc":
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        if len(np.unique(y_test)) > 2:
            return float(
                roc_auc_score(
                    y_test,
                    model.predict_proba(X_test),
                    multi_class="ovr",
                    average="weighted",
                )
            )
        return float(roc_auc_score(y_test, y_score))
    raise ValueError("Unsupported metrics. Use 'f1' or 'roc_auc'.")


def evaluate_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric: str,
    seed: int,
    n_splits: int = 5,
) -> float:
    """Evaluate model with stratified K-fold CV."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        m = clone(model)
        scores.append(evaluate(m, X_tr, y_tr, X_te, y_te, metric))
    return float(np.mean(scores))


def make_sampler(
    sampler_name: str, seed: int, grid_space: dict[str, Any] | None = None
) -> optuna.samplers.BaseSampler:
    """Create Optuna sampler from config."""
    sampler_name = sampler_name.lower()
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler_name == "grid":
        if not grid_space:
            raise ValueError("For sampler='grid' need to set grid_space.")
        return optuna.samplers.GridSampler(search_space=grid_space)
    raise ValueError("sampler should be: tpe, random, grid")


def suggest_params(
    trial: optuna.Trial, model_type: str, cfg: DictConfig
) -> dict[str, Any]:
    """Suggest hyperparameters for trial from search space."""
    if model_type == "random_forest":
        space = cfg.hpo.random_forest
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", space.n_estimators.low, space.n_estimators.high
            ),
            "max_depth": trial.suggest_int(
                "max_depth", space.max_depth.low, space.max_depth.high
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                space.min_samples_split.low,
                space.min_samples_split.high,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                space.min_samples_leaf.low,
                space.min_samples_leaf.high,
            ),
        }
    if model_type == "logistic_regression":
        space = cfg.hpo.logistic_regression
        return {
            "C": trial.suggest_float("C", space.C.low, space.C.high, log=True),
            "solver": trial.suggest_categorical("solver", list(space.solver)),
            "penalty": trial.suggest_categorical("penalty", list(space.penalty)),
        }
    raise ValueError(f"Unknown model.type='{model_type}'.")


def objective_factory(
    cfg: DictConfig,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
):
    """Return objective function for Optuna that logs each trial in MLflow."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg.model.type, cfg)
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", str(cfg.seed))
            mlflow.log_params(params)

            model = build_model(cfg.model.type, params=params, seed=cfg.seed)
            if cfg.hpo.use_cv:
                X = np.concatenate([X_train, X_test], axis=0)
                y = np.concatenate([y_train, y_test], axis=0)
                score = evaluate_cv(
                    model, X, y, metric=cfg.hpo.metric, seed=cfg.seed, n_splits=cfg.hpo.cv_folds
                )
            else:
                score = evaluate(model, X_train, y_train, X_test, y_test, cfg.hpo.metric)

            mlflow.log_metric(cfg.hpo.metric, score)
            return score

    return objective


def register_model_if_enabled(
    model_uri: str, model_name: str, stage: str
) -> None:
    """Register model in MLflow Model Registry and transition to stage."""
    client = mlflow.tracking.MlflowClient()
    mv = mlflow.register_model(model_uri, model_name)
    client.transition_model_version_stage(
        name=model_name, version=mv.version, stage=stage
    )
    client.set_model_version_tag(model_name, mv.version, "registered_by", "lab3")
    client.set_model_version_tag(model_name, mv.version, "stage", stage)
    logger.info("Registered model '%s' version %s -> %s", model_name, mv.version, stage)


def _ensure_mlflow_experiment(experiment_name: str) -> None:
    """Set active experiment; restore it if it was deleted (e.g. from UI)."""
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp and getattr(exp, "lifecycle_stage", None) == "deleted":
        client.restore_experiment(exp.experiment_id)
        logger.info("Restored deleted experiment '%s'", experiment_name)
    mlflow.set_experiment(experiment_name)


def main(cfg: DictConfig) -> None:
    """Run HPO with Optuna, log to MLflow as nested runs, retrain and save best model."""
    set_global_seed(cfg.seed)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    try:
        mlflow.set_experiment(cfg.mlflow.experiment_name)
    except MlflowException as e:
        if "deleted" in str(e).lower():
            _ensure_mlflow_experiment(cfg.mlflow.experiment_name)
        else:
            raise

    X_train, X_test, y_train, y_test = load_prepared_data(
        cfg.data.prepared_dir,
        target_type=cfg.data.target_type,
        popularity_threshold=cfg.data.popularity_threshold,
    )

    grid_space: dict[str, Any] | None = None
    if cfg.hpo.sampler.lower() == "grid":
        if cfg.model.type == "random_forest":
            grid_space = {
                "n_estimators": list(cfg.hpo.grid.random_forest.n_estimators),
                "max_depth": list(cfg.hpo.grid.random_forest.max_depth),
                "min_samples_split": list(cfg.hpo.grid.random_forest.min_samples_split),
                "min_samples_leaf": list(cfg.hpo.grid.random_forest.min_samples_leaf),
            }
        else:
            grid_space = {
                "C": list(cfg.hpo.grid.logistic_regression.C),
                "solver": list(cfg.hpo.grid.logistic_regression.solver),
                "penalty": list(cfg.hpo.grid.logistic_regression.penalty),
            }

    sampler = make_sampler(cfg.hpo.sampler, seed=cfg.seed, grid_space=grid_space)

    with mlflow.start_run(run_name=f"HPO_{cfg.hpo.sampler}") as parent_run:
        mlflow.set_tag("model_type", cfg.model.type)
        mlflow.set_tag("sampler", cfg.hpo.sampler)
        mlflow.set_tag("seed", str(cfg.seed))
        mlflow.set_tag("code_version", get_code_version())
        mlflow.log_dict(
            OmegaConf.to_container(cfg, resolve=True), "config_resolved.json"
        )

        study = optuna.create_study(
            direction=cfg.hpo.direction,
            sampler=sampler,
        )
        objective = objective_factory(cfg, X_train, X_test, y_train, y_test)
        study.optimize(objective, n_trials=cfg.hpo.n_trials)

        best_trial = study.best_trial
        mlflow.log_metric(f"best_{cfg.hpo.metric}", float(best_trial.value))
        mlflow.log_dict(dict(best_trial.params), "best_params.json")

        best_model = build_model(
            cfg.model.type, params=best_trial.params, seed=cfg.seed
        )
        best_score = evaluate(
            best_model, X_train, y_train, X_test, y_test, metric=cfg.hpo.metric
        )
        mlflow.log_metric(f"final_{cfg.hpo.metric}", best_score)

        os.makedirs("models", exist_ok=True)
        model_path = "models/best_model.pkl"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, artifact_path="model")

        if cfg.mlflow.register_model:
            model_uri = f"runs:/{parent_run.info.run_id}/model"
            register_model_if_enabled(
                model_uri, cfg.mlflow.model_name, stage=cfg.mlflow.stage
            )

        logger.info(
            "HPO complete. Best %s=%.4f, final (retrain) %s=%.4f",
            cfg.hpo.metric,
            best_trial.value,
            cfg.hpo.metric,
            best_score,
        )
        logger.info("Best params: %s", best_trial.params)


def hydra_entry(cfg: DictConfig) -> None:
    """Hydra entry point."""
    main(cfg)


if __name__ == "__main__":
    import hydra

    # config_path relative to this file: src/ -> config is ../config
    config_path = Path(__file__).resolve().parent.parent / "config"
    hydra.main(
        version_base=None,
        config_path=str(config_path),
        config_name="config",
    )(hydra_entry)()
