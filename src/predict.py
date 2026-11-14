"""
Prediction / serving: recommend items for given users with a trained model,
re-using the SAME feature definitions (serving snapshot in data/features/serving).

    python -m src.predict --users U00001 U00042 --model hybrid_als_content --k 10
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.features.build_features import FeatureStore
from src.training.models import BaseRecommender, HybridRecommender
from src.utils import ROOT, load_config, p, read_json, read_table


def recommend(user_ids, model_name="hybrid_als_content", k=10, cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    reg = read_json(p(cfg, "models") / "registry.json")
    model = BaseRecommender.load(ROOT / reg["models"][model_name]["artifact"])

    # serving-time features (as_of = latest data), same code path as training
    serving_cfg = {**cfg, "paths": {**cfg["paths"], "features": f"{cfg['paths']['features']}/serving"}}
    fs = FeatureStore(serving_cfg)
    model.ctx.feature_store = fs
    if isinstance(model, HybridRecommender):
        G, genres = model.ctx.genre_matrix()
        A = fs.genre_affinity_matrix(model.ctx.user_index, genres)
        model.content = A @ (G / np.maximum(G.sum(1, keepdims=True), 1)).T

    # also exclude anything the user interacted with after the training cutoff
    allx = read_table(p(cfg, "processed"), "interactions_clean")
    seen = allx.groupby("user_id").item_id.apply(set).to_dict()
    recs = model.recommend(user_ids, k + 50)
    items = model.ctx.items.set_index("item_id")
    feats = fs.get_item_features(model.ctx.item_ids, ["item_bayesian_avg_rating", "item_rating_count"])
    rows = []
    for u in user_ids:
        lst = [i for i in recs[u] if i not in seen.get(u, set())][:k]
        uf = fs.get_user_features([u]).iloc[0]
        for rank, i in enumerate(lst, 1):
            rows.append({"user_id": u, "rank": rank, "item_id": i, "title": items.title[i],
                         "genres": items.genres[i],
                         "item_avg_rating": round(feats.item_bayesian_avg_rating[i], 2),
                         "user_preferred_genre": uf.user_preferred_genre})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", nargs="+", default=["U00001"])
    ap.add_argument("--model", default="hybrid_als_content")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    print(recommend(a.users, a.model, a.k).to_string(index=False))
