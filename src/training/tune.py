"""
Hyper-parameter search on a VALIDATION slice of the training period
(last 20% of train by time). The test split is never touched, so the
reported test metrics stay unbiased.

    python -m src.training.tune      -> reports/tuning_results.csv
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.training.models import ImplicitALS, ItemKNNRecommender
from src.training.train import build_context
from src.utils import get_logger, load_config, p

log = get_logger("tuning")

GRID = {
    "implicit_als": (ImplicitALS, {"n_factors": [16, 32], "regularization": [1, 10, 50],
                                   "alpha": [5, 20], "n_iterations": [15]}),
    "item_knn_cf": (ItemKNNRecommender, {"n_neighbors": [20, 50, 100], "shrinkage": [10, 50]}),
}


def run(cfg: dict | None = None, k: int = 10) -> pd.DataFrame:
    cfg = cfg or load_config()
    ctx, train = build_context(cfg)
    cut = train.timestamp.quantile(0.8)
    tr, va = train[train.timestamp <= cut], train[train.timestamp > cut]
    rel = va[va.rating >= cfg["evaluation"]["relevance_threshold"]].groupby("user_id").item_id.apply(set)
    users = [u for u in rel.index if u in set(tr.user_id)]
    rows = []
    for name, (cls, grid) in GRID.items():
        keys = list(grid)
        for vals in itertools.product(*grid.values()):
            params = dict(zip(keys, vals))
            recs = cls(**params).fit(tr, ctx).recommend(users, k)
            ndcg = np.mean([M.ndcg_at_k(recs[u], rel[u], k) for u in users])
            rows.append({"model": name, **params, f"val_ndcg_at_{k}": round(float(ndcg), 4)})
            log.info("%s %s -> NDCG@%d=%.4f", name, params, k, ndcg)
    df = pd.DataFrame(rows).sort_values(["model", f"val_ndcg_at_{k}"], ascending=[True, False])
    df.to_csv(p(cfg, "reports") / "tuning_results.csv", index=False)
    return df


if __name__ == "__main__":
    print(run().groupby("model").head(1).to_string(index=False))
