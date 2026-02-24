"""
Lab 5: DAG integrity test for CI — ensure DAGs load without import errors.
Run with: pytest tests/test_dag_integrity.py (requires apache-airflow in CI).
"""

import os
from pathlib import Path

import pytest


def test_dag_import() -> None:
    """Ensure all DAGs in dags/ load without import or syntax errors."""
    # Avoid Airflow looking for config in system paths during test
    airflow_home = Path(__file__).resolve().parent.parent
    os.environ.setdefault("AIRFLOW_HOME", str(airflow_home))

    from airflow.models import DagBag

    dag_folder = airflow_home / "dags"
    assert dag_folder.is_dir(), f"dags folder not found: {dag_folder}"

    dag_bag = DagBag(dag_folder=str(dag_folder), include_examples=False)
    assert (
        len(dag_bag.import_errors) == 0
    ), f"DAG import errors: {dag_bag.import_errors}"
    assert len(dag_bag.dags) >= 1, "Expected at least one DAG in dags/"
    # Lab 5 pipeline must be present
    assert (
        "ml_training_pipeline" in dag_bag.dags
    ), "Expected DAG 'ml_training_pipeline' in dags/"
