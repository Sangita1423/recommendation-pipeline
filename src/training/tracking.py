"""
Experiment tracking.

Uses MLflow when it is installed. Runs are stored in a local SQLite database
(mlflow.db) and artifacts in mlruns/ — MLflow 3.x no longer accepts the plain
file store. View the runs with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

If MLflow is not installed OR fails to initialise, the same API falls back to
a file-based tracker with an MLflow-like layout, so training never breaks:

  mlruns/<experiment>/<run_id>/{meta.json, params.json, metrics.json, artifacts/}
"""
from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from src.utils import ROOT, get_logger, read_json, write_json

log = get_logger("tracking")

try:
    import mlflow  # type: ignore
    from mlflow.tracking import MlflowClient  # type: ignore
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class _LocalRun:
    def __init__(self, exp_dir: Path, run_name: str):
        self.run_id = uuid.uuid4().hex[:12]
        self.dir = exp_dir / self.run_id
        (self.dir / "artifacts").mkdir(parents=True, exist_ok=True)
        self.params, self.metrics = {}, {}
        self.meta = {"run_id": self.run_id, "run_name": run_name,
                     "start_time": datetime.now().isoformat(timespec="seconds"),
                     "tracker": "local-file"}

    def flush(self):
        write_json(self.params, self.dir / "params.json")
        write_json(self.metrics, self.dir / "metrics.json")
        write_json(self.meta, self.dir / "meta.json")


class Tracker:
    def __init__(self, experiment: str, uri: str = "mlruns"):
        self.uri = ROOT / uri                        # artifact folder (and local fallback store)
        self.uri.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self._run = None
        self.use_mlflow = False
        if HAS_MLFLOW:
            try:
                db_path = (ROOT / "mlflow.db").resolve().as_posix()   # works on Windows too
                mlflow.set_tracking_uri(f"sqlite:///{db_path}")
                client = MlflowClient()
                if client.get_experiment_by_name(experiment) is None:
                    client.create_experiment(experiment,
                                             artifact_location=self.uri.resolve().as_uri())
                mlflow.set_experiment(experiment)
                self.use_mlflow = True
            except Exception as e:  # noqa: BLE001
                log.warning("MLflow could not start (%s). Using the local file tracker instead.", e)
        self.backend = "mlflow (sqlite:///mlflow.db)" if self.use_mlflow else "local-file"
        log.info("Experiment tracking backend: %s", self.backend)

    @contextmanager
    def start_run(self, run_name: str, tags: dict | None = None):
        if self.use_mlflow:
            with mlflow.start_run(run_name=run_name) as r:
                if tags:
                    mlflow.set_tags(tags)
                self._run = r
                yield r.info.run_id
        else:
            self._run = _LocalRun(self.uri / self.experiment, run_name)
            self._run.meta["tags"] = tags or {}
            try:
                yield self._run.run_id
                self._run.meta["status"] = "FINISHED"
            except Exception:
                self._run.meta["status"] = "FAILED"
                raise
            finally:
                self._run.meta["end_time"] = datetime.now().isoformat(timespec="seconds")
                self._run.flush()
        self._run = None

    @contextmanager
    def resume_run(self, run_id: str):
        """Re-open a training run so evaluation metrics land on the same run."""
        if self.use_mlflow:
            with mlflow.start_run(run_id=run_id) as r:
                self._run = r
                yield run_id
        else:
            run = _LocalRun.__new__(_LocalRun)
            run.run_id = run_id
            run.dir = self.uri / self.experiment / run_id
            run.params = read_json(run.dir / "params.json")
            run.metrics = read_json(run.dir / "metrics.json")
            run.meta = read_json(run.dir / "meta.json")
            self._run = run
            try:
                yield run_id
            finally:
                run.flush()
        self._run = None

    def log_params(self, params: dict):
        if self.use_mlflow:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        else:
            self._run.params.update(params)

    def log_metrics(self, metrics: dict):
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if self.use_mlflow:
            mlflow.log_metrics(clean)
        else:
            self._run.metrics.update(clean)

    def log_artifact(self, path: Path):
        if self.use_mlflow:
            mlflow.log_artifact(str(path))
        else:
            shutil.copy(path, self._run.dir / "artifacts" / Path(path).name)
