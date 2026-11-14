"""
Prefect 2/3 flow — an alternative orchestrator with the same task graph.

    pip install prefect
    python dags/prefect_flow.py                 # one run
    prefect server start                        # optional UI at http://127.0.0.1:4200
    # schedule daily: python dags/prefect_flow.py --serve
"""
import os
import sys

from prefect import flow, task

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate  # noqa: E402
from src.features import build_features  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.preparation import prepare  # noqa: E402
from src.training import train  # noqa: E402
from src.transformation import transform  # noqa: E402
from src.utils import load_config  # noqa: E402
from src.validation import validate  # noqa: E402

RETRY = dict(retries=2, retry_delay_seconds=30)


@task(name="ingest_data", **RETRY)
def t_ingest(cfg): return ingest.run(cfg)


@task(name="validate_data", **RETRY)
def t_validate(cfg, _): return validate.run(cfg)


@task(name="prepare_data", **RETRY)
def t_prepare(cfg, _): return prepare.run(cfg)


@task(name="transform_data", **RETRY)
def t_transform(cfg, _): return transform.run(cfg)


@task(name="build_features", **RETRY)
def t_features(cfg, _): return build_features.run(cfg, "train")


@task(name="build_serving_features", **RETRY)
def t_serving(cfg, _): return build_features.run(cfg, "serve")


@task(name="train_models", **RETRY)
def t_train(cfg, *_): return train.run(cfg)


@task(name="evaluate_models", **RETRY)
def t_evaluate(cfg, _): return evaluate.run(cfg).to_dict("records")


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str | None = None):
    cfg = load_config(config_path)
    a = t_ingest(cfg)
    b = t_validate(cfg, a)
    c = t_prepare(cfg, b)
    d = t_transform(cfg, c)
    e = t_features(cfg, c)
    t_serving(cfg, c)
    f = t_train(cfg, d, e)
    return t_evaluate(cfg, f)


if __name__ == "__main__":
    if "--serve" in sys.argv:
        recommendation_pipeline.serve(name="daily", cron="0 2 * * *")
    else:
        recommendation_pipeline()
