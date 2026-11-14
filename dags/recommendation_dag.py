"""
Apache Airflow DAG for the recommendation pipeline (Airflow 2.x).

Setup:
    pip install "apache-airflow==2.9.3" --constraint \
      "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt"
    export AIRFLOW_HOME=~/airflow
    export RECSYS_HOME=/absolute/path/to/recommendation-pipeline
    cp dags/recommendation_dag.py $AIRFLOW_HOME/dags/
    airflow standalone            # UI at http://localhost:8080
    # trigger "recommendation_pipeline" from the UI, or:
    airflow dags test recommendation_pipeline 2026-11-14

Each task calls the same stage function used by run_pipeline.py, so the
Airflow run and the local run are identical. Stages exchange data through
files in data/ (not XCom), which keeps tasks idempotent and re-runnable.
"""
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT = os.environ.get("RECSYS_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT)


def _stage(module: str, func: str = "run", **kwargs):
    def _call():
        import importlib
        from src.utils import load_config
        mod = importlib.import_module(module)
        getattr(mod, func)(load_config(), **kwargs)
    return _call


default_args = {
    "owner": "data-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=30),
}

with DAG(
    dag_id="recommendation_pipeline",
    description="Ingest -> validate -> prepare -> transform/features -> train -> evaluate",
    default_args=default_args,
    start_date=datetime(2026, 11, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["recsys", "assignment-2"],
) as dag:
    ingest = PythonOperator(task_id="ingest_data", python_callable=_stage("src.ingestion.ingest"))
    validate = PythonOperator(task_id="validate_data", python_callable=_stage("src.validation.validate"))
    prepare = PythonOperator(task_id="prepare_data", python_callable=_stage("src.preparation.prepare"))
    transform = PythonOperator(task_id="transform_data",
                               python_callable=_stage("src.transformation.transform"))
    features = PythonOperator(task_id="build_features",
                              python_callable=_stage("src.features.build_features", mode="train"))
    serving = PythonOperator(task_id="build_serving_features",
                             python_callable=_stage("src.features.build_features", mode="serve"))
    train = PythonOperator(task_id="train_models", python_callable=_stage("src.training.train"))
    evaluate = PythonOperator(task_id="evaluate_models", python_callable=_stage("src.evaluation.evaluate"))

    ingest >> validate >> prepare >> [transform, features, serving]
    [transform, features] >> train >> evaluate
