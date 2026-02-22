"""
Lab 5: ML training pipeline DAG — Sensor → Prepare → Train → Evaluate → Branch → Register or Stop.

Runs in Airflow with ML project mounted at ML_PROJECT_ROOT (default /opt/airflow/ml_project).
Produces model.pkl, metrics.json, confusion_matrix.png; registers model in MLflow if accuracy > threshold.
"""

from datetime import datetime
import json
import os
from pathlib import Path

from airflow.models import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.empty import EmptyOperator

# Project root when running in Airflow container (mount point)
ML_PROJECT_ROOT = os.environ.get("ML_PROJECT_ROOT", "/opt/airflow/ml_project")
ACCURACY_THRESHOLD = float(os.environ.get("ML_ACCURACY_THRESHOLD", "0.85"))
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "spotify_ci_classifier")

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
}

with DAG(
    dag_id="ml_training_pipeline",
    default_args=DEFAULT_ARGS,
    description="Lab 5: Prepare data (DVC) → Train (train_ci) → Evaluate → Register in MLflow if accuracy > threshold",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["lab5", "mlops", "training"],
) as dag:
    # 1. Sensor: check raw data is available
    check_data = FileSensor(
        task_id="check_data",
        filepath=str(Path(ML_PROJECT_ROOT) / "data" / "raw" / "dataset.csv"),
        poke_interval=30,
        timeout=300,
    )

    # 2. Data preparation: DVC repro prepare only
    data_prepare = BashOperator(
        task_id="data_prepare",
        bash_command=f"cd {ML_PROJECT_ROOT} && dvc repro prepare",
    )

    # 3. Model training: produces model.pkl, metrics.json, confusion_matrix.png
    model_train = BashOperator(
        task_id="model_train",
        bash_command=f"cd {ML_PROJECT_ROOT} && python scripts/train_ci.py",
    )

    def read_metrics_and_push(**kwargs) -> dict:
        """Read metrics.json from project root and return for XCom (branching)."""
        metrics_path = Path(ML_PROJECT_ROOT) / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics.json not found at {metrics_path}")
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        return metrics

    # 4. Evaluation: push metrics to XCom
    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=read_metrics_and_push,
    )

    def check_accuracy(**kwargs) -> str:
        """Branch: register model if accuracy > threshold, else stop."""
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="evaluate_model")
        if metrics is None:
            return "stop_pipeline"
        accuracy = metrics.get("accuracy", 0.0)
        if accuracy > ACCURACY_THRESHOLD:
            return "register_model"
        return "stop_pipeline"

    # 5. Branching
    branch = BranchPythonOperator(
        task_id="branch_on_accuracy",
        python_callable=check_accuracy,
    )

    def register_model_mlflow(**kwargs) -> None:
        """Load model.pkl, log to MLflow, register and set stage to Staging."""
        import joblib
        import mlflow

        model_path = Path(ML_PROJECT_ROOT) / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"model.pkl not found at {model_path}")

        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI", "file:///opt/airflow/ml_project/mlruns"
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlruns_dir = tracking_uri.replace("file://", "")
        os.makedirs(mlruns_dir, exist_ok=True)

        with mlflow.start_run(run_name="airflow_register"):
            model = joblib.load(model_path)
            mlflow.sklearn.log_model(model, artifact_path="model")
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"

        client = mlflow.tracking.MlflowClient()
        try:
            client.create_registered_model(MLFLOW_MODEL_NAME)
        except Exception:
            pass  # model may already exist
        mv = client.register_model(model_uri=model_uri, name=MLFLOW_MODEL_NAME)
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME, version=mv.version, stage="Staging"
        )

    # 6a. Register model in MLflow (Staging)
    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_model_mlflow,
    )

    # 6b. Stop pipeline (model below threshold)
    stop_pipeline = EmptyOperator(task_id="stop_pipeline")

    # Dependencies
    check_data >> data_prepare >> model_train >> evaluate_model >> branch
    branch >> register_model
    branch >> stop_pipeline
