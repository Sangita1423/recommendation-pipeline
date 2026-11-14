"""
Pipeline definition + a dependency-aware task runner.

The DAG below is the single definition of the workflow. It is executed by:
  * run_pipeline.py                 (this runner: no extra installs, cron-schedulable)
  * dags/recommendation_dag.py      (Apache Airflow, same tasks + dependencies)
  * dags/prefect_flow.py            (Prefect, same tasks + dependencies)

Runner features: topological ordering, upstream-failure propagation (tasks are
"upstream_failed" instead of running on bad inputs), retries, per-task timing,
a JSON run record in reports/runs/ and a rendered run graph PNG.
"""
from __future__ import annotations

import time
import traceback
import uuid
from datetime import datetime
from graphlib import TopologicalSorter

from src.evaluation import evaluate
from src.features import build_features
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.utils import get_logger, load_config, p, write_json
from src.validation import validate

log = get_logger("orchestrator")

# task_name: (callable(cfg), [upstream tasks])
DAG = {
    "ingest_data":         (ingest.run,                                   []),
    "validate_data":       (validate.run,                                 ["ingest_data"]),
    "prepare_data":        (prepare.run,                                  ["validate_data"]),
    "transform_data":      (transform.run,                                ["prepare_data"]),
    "build_features":      (lambda c: build_features.run(c, "train"),     ["prepare_data"]),
    "build_serving_features": (lambda c: build_features.run(c, "serve"),  ["prepare_data"]),
    "train_models":        (train.run,                                    ["transform_data", "build_features"]),
    "evaluate_models":     (evaluate.run,                                 ["train_models"]),
}


def run_pipeline(cfg: dict | None = None, retries: int = 1, only: list[str] | None = None) -> dict:
    cfg = cfg or load_config()
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
    order = list(TopologicalSorter({t: set(d[1]) for t, d in DAG.items()}).static_order())
    if only:
        order = [t for t in order if t in only]
    state = {t: {"status": "pending", "upstream": DAG[t][1]} for t in order}
    log.info("=== Pipeline run %s | %d tasks: %s", run_id, len(order), " -> ".join(order))
    t_start = time.time()

    for task in order:
        fn, upstream = DAG[task]
        bad = [u for u in upstream if u in state and state[u]["status"] != "success"]
        if bad:
            state[task]["status"] = "upstream_failed"
            log.error("Skipping %s: upstream failed %s", task, bad)
            continue
        for attempt in range(1, retries + 2):
            st = time.time()
            state[task].update(start=datetime.now().isoformat(timespec="seconds"),
                               start_offset=round(st - t_start, 2), attempt=attempt)
            try:
                log.info(">>> %s (attempt %d)", task, attempt)
                fn(cfg)
                state[task].update(status="success", duration_sec=round(time.time() - st, 2))
                log.info("<<< %s succeeded in %.1fs", task, time.time() - st)
                break
            except Exception as e:  # noqa: BLE001
                state[task].update(status="failed", error=repr(e),
                                   traceback=traceback.format_exc(),
                                   duration_sec=round(time.time() - st, 2))
                log.exception("Task %s failed on attempt %d", task, attempt)

    overall = "success" if all(s["status"] == "success" for s in state.values()) else "failed"
    record = {"run_id": run_id, "status": overall,
              "started_at": datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
              "duration_sec": round(time.time() - t_start, 2), "tasks": state}
    runs = p(cfg, "reports") / "runs"
    runs.mkdir(exist_ok=True)
    write_json(record, runs / f"run_{run_id}.json")
    write_json(record, runs / "latest_run.json")
    try:
        from src.orchestration.render import render_run
        render_run(record, runs / "latest_run.png")
    except Exception as e:  # noqa: BLE001
        log.warning("Could not render run graph: %s", e)
    log.info("=== Pipeline %s finished: %s in %.1fs", run_id, overall.upper(), record["duration_sec"])
    return record
