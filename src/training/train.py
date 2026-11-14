"""
Model training stage: fits every configured model on the train split,
saves artifacts to models/ and logs params / training metrics / artifacts
to the experiment tracker. A small model registry (models/registry.json)
records the run_id and artifact path of each model for the evaluation stage.
"""
from __future__ import annotations

import time

import pandas as pd

from src.features.build_features import FeatureStore
from src.training.models import (HybridRecommender, ImplicitALS, ItemKNNRecommender,
                                 MatrixFactorization, PopularityRecommender, TrainingContext)
from src.training.tracking import Tracker
from src.utils import ROOT, get_logger, load_config, p, read_json, read_table, write_json
from src.evaluation.metrics import mae, rmse

log = get_logger("training")


def build_context(cfg: dict) -> tuple[TrainingContext, pd.DataFrame]:
    d = p(cfg, "processed")
    train = read_table(d, "train", parse_dates=["timestamp"])
    uidx = pd.read_csv(d / "user_index.csv").set_index("user_id")["row"]
    iidx = pd.read_csv(d / "item_index.csv").set_index("item_id")["col"]
    items = read_table(d, "items_clean")
    as_of = pd.Timestamp(read_json(d / "split_info.json")["cutoff_timestamp"])
    return TrainingContext(uidx, iidx, items, as_of, FeatureStore(cfg)), train


def make_models(cfg: dict) -> list:
    m = cfg["models"]
    mf = MatrixFactorization(**m["matrix_factorization"], seed=cfg["seed"])
    als = ImplicitALS(**m["implicit_als"], seed=cfg["seed"])
    # order matters: the hybrid re-uses the already-fitted ALS and MF objects
    return [PopularityRecommender(**m["popularity"]),
            ItemKNNRecommender(**m["item_knn"]),
            mf,
            als,
            HybridRecommender(als, mf, **m["hybrid"])]


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ctx, train = build_context(cfg)
    tracker = Tracker(cfg["tracking"]["experiment_name"], cfg["paths"]["mlruns"])
    models_dir = p(cfg, "models")
    registry = {"tracker": tracker.backend, "models": {}}

    for model in make_models(cfg):
        with tracker.start_run(run_name=model.name,
                               tags={"stage": "training", "model_type": model.name}) as run_id:
            t0 = time.time()
            model.fit(train, ctx)
            secs = time.time() - t0
            yhat = model.predict(train.user_id.values, train.item_id.values)
            tracker.log_params({"model": model.name, **model.params(), "seed": cfg["seed"],
                                "train_rows": len(train), "split_cutoff": str(ctx.as_of)})
            tracker.log_metrics({"train_time_sec": secs, "train_rmse": rmse(train.rating, yhat),
                                 "train_mae": mae(train.rating, yhat)})
            path = models_dir / f"{model.name}.pkl"
            model.save(path)
            tracker.log_artifact(path)
            tracker.log_artifact(ROOT / "configs" / "config.yaml")
            registry["models"][model.name] = {"run_id": run_id, "artifact": str(path.relative_to(ROOT)),
                                              "params": model.params(), "train_time_sec": round(secs, 2)}
            log.info("Trained %-22s in %.1fs (run %s)", model.name, secs, run_id)

    write_json(registry, models_dir / "registry.json")
    return registry


if __name__ == "__main__":
    run()
