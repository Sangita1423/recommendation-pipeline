"""
Feature computation + a lightweight feature store.

`compute_features(interactions, users, items, as_of)` is a PURE function: the
training run calls it with the train split and as_of = split cutoff; the
prediction run calls it with all cleaned interactions and as_of = now.
Same code, same definitions -> no training/serving skew.

Tables written to data/features/ (point-in-time stamped):
  user_features, item_features, user_genre_features, feature_metadata.json
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.features import registry
from src.utils import get_logger, load_config, p, read_json, read_table, save_table, write_json

log = get_logger("features")


def compute_features(inter: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame,
                     as_of: pd.Timestamp) -> dict[str, pd.DataFrame]:
    inter = inter[inter.timestamp <= as_of]
    recent = inter[inter.timestamp > as_of - pd.Timedelta(days=30)]
    n_genres = items.genres.str.split("|").explode().nunique()

    # ---- item ----
    g = inter.groupby("item_id")
    it = g.agg(item_rating_count=("rating", "size"), item_unique_users=("user_id", "nunique"),
               item_avg_rating=("rating", "mean"), _sum=("rating", "sum"),
               _last=("timestamp", "max"))
    it = items.set_index("item_id").join(it, how="left")
    it["item_rating_count"] = it.item_rating_count.fillna(0).astype(int)
    it["item_unique_users"] = it.item_unique_users.fillna(0).astype(int)
    m, C = inter.rating.mean(), max(1.0, it.item_rating_count[it.item_rating_count > 0].median())
    it["item_bayesian_avg_rating"] = (C * m + it._sum.fillna(0)) / (C + it.item_rating_count)
    it["item_popularity_percentile"] = it.item_rating_count.rank(pct=True)
    it["item_ratings_last_30d"] = recent.groupby("item_id").size().reindex(it.index).fillna(0).astype(int)
    it["item_days_since_last_interaction"] = (as_of - it._last).dt.days
    it["item_primary_genre"] = it.genres.str.split("|").str[0]
    it["item_genre_count"] = it.genres.str.split("|").str.len()
    it["item_age_years"] = as_of.year - it.release_year
    it["item_duration_min"] = it.duration_min
    item_f = it.reset_index()[["item_id"] + registry.names("item")]

    # ---- user x genre ----
    x = inter.merge(items[["item_id", "genres"]], on="item_id")
    x = x.assign(genre=x.genres.str.split("|")).explode("genre")
    ug = x.groupby(["user_id", "genre"]).agg(_n=("rating", "size"), ug_avg_rating=("rating", "mean"))
    ug["ug_share"] = ug._n / inter.groupby("user_id").size().reindex(ug.index.get_level_values(0)).values
    u_mean = inter.groupby("user_id").rating.mean()
    ug["ug_affinity"] = ug.ug_share * (ug.ug_avg_rating - u_mean.reindex(
        ug.index.get_level_values(0)).values + 1).clip(lower=0)
    ug_f = ug.reset_index()[["user_id", "genre"] + registry.names("user_genre")]

    # ---- user ----
    g = inter.groupby("user_id")
    us = g.agg(user_rating_count=("rating", "size"), user_avg_rating=("rating", "mean"),
               user_rating_std=("rating", "std"), _first=("timestamp", "min"),
               _last=("timestamp", "max"))
    us = users.set_index("user_id").join(us, how="left")
    us["user_rating_count"] = us.user_rating_count.fillna(0).astype(int)
    us["user_rating_std"] = us.user_rating_std.fillna(0)
    us["user_ratings_last_30d"] = recent.groupby("user_id").size().reindex(us.index).fillna(0).astype(int)
    us["user_days_since_last_interaction"] = (as_of - us._last).dt.days
    us["user_active_span_days"] = (us._last - us._first).dt.days + 1
    pref = (x.groupby(["user_id", "genre"]).size().reset_index(name="n")
            .sort_values(["user_id", "n"], ascending=[True, False]).drop_duplicates("user_id"))
    us["user_preferred_genre"] = pref.set_index("user_id").genre.reindex(us.index)
    us["user_genre_diversity"] = (x.groupby("user_id").genre.nunique() / n_genres).reindex(us.index)
    pop = item_f.set_index("item_id").item_popularity_percentile
    us["user_mainstreamness"] = inter.assign(pp=inter.item_id.map(pop)).groupby("user_id").pp.mean()
    us["user_age"] = us.age
    us["user_gender"] = us.gender
    us["user_tenure_days"] = (as_of - pd.to_datetime(us.signup_date)).dt.days
    user_f = us.reset_index()[["user_id"] + registry.names("user")]
    return {"user_features": user_f, "item_features": item_f, "user_genre_features": ug_f}


class FeatureStore:
    """Minimal offline feature store: point-in-time tables + lookup API."""

    def __init__(self, cfg: dict):
        self.dir = p(cfg, "features")
        self.meta = read_json(self.dir / "feature_metadata.json")
        self.user = read_table(self.dir, "user_features").set_index("user_id")
        self.item = read_table(self.dir, "item_features").set_index("item_id")
        self.user_genre = read_table(self.dir, "user_genre_features")

    def get_user_features(self, user_ids, cols=None) -> pd.DataFrame:
        return self.user.reindex(user_ids)[cols or self.user.columns]

    def get_item_features(self, item_ids, cols=None) -> pd.DataFrame:
        return self.item.reindex(item_ids)[cols or self.item.columns]

    def genre_affinity_matrix(self, user_index: pd.Series, genres: list[str]) -> np.ndarray:
        piv = self.user_genre.pivot_table(index="user_id", columns="genre",
                                          values="ug_affinity", fill_value=0)
        return piv.reindex(index=user_index.index, columns=genres, fill_value=0).values


def write_catalog(cfg: dict, feats: dict[str, pd.DataFrame]) -> None:
    rows = []
    tables = {"user": feats["user_features"], "item": feats["item_features"],
              "user_genre": feats["user_genre_features"]}
    for f in registry.FEATURES:
        s = tables[f.entity][f.name]
        if s.dtype.kind in "if":
            stats = f"mean={s.mean():.2f}, min={s.min():.2f}, max={s.max():.2f}"
        else:
            stats = f"{s.nunique()} distinct, top='{s.mode().iat[0] if s.notna().any() else ''}'"
        rows.append({**registry.asdict(f), "null_pct": round(s.isna().mean() * 100, 2), "stats": stats})
    pd.DataFrame(rows).to_csv(p(cfg, "reports") / "feature_catalog.csv", index=False)
    md = ["# Feature Catalog", "",
          f"Feature version: `{registry.FEATURE_VERSION}`. Definitions live in "
          "`src/features/registry.py`; computation in `src/features/build_features.py`. "
          "Statistics below are from the training snapshot (as_of = split cutoff).", "",
          "| # | Feature | Entity | Type | Description | Computation | Source | Null % | Training stats |",
          "|---:|---|---|---|---|---|---|---:|---|"]
    for k, r in enumerate(rows, 1):
        md.append(f"| {k} | `{r['name']}` | {r['entity']} | {r['dtype']} | {r['description']} | "
                  f"`{r['logic']}` | {r['source']} | {r['null_pct']} | {r['stats']} |")
    (p(cfg, "reports") / "feature_catalog.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def run(cfg: dict | None = None, mode: str = "train") -> dict:
    """mode='train' -> features from train split at cutoff (used for model training)
       mode='serve' -> features from all clean data at now (used for live prediction)"""
    cfg = cfg or load_config()
    d = p(cfg, "processed")
    users = read_table(d, "users_clean", parse_dates=["signup_date"])
    items = read_table(d, "items_clean")
    if mode == "train":
        inter = read_table(d, "train", parse_dates=["timestamp"])
        as_of = pd.Timestamp(read_json(d / "split_info.json")["cutoff_timestamp"])
        out = p(cfg, "features")
    else:
        inter = read_table(d, "interactions_clean", parse_dates=["timestamp"])
        as_of = inter.timestamp.max()
        out = p(cfg, "features") / "serving"
        out.mkdir(exist_ok=True)

    feats = compute_features(inter, users, items, as_of)
    for name, df in feats.items():
        save_table(df, out, name, cfg)
    meta = {"feature_version": registry.FEATURE_VERSION, "mode": mode, "as_of": str(as_of),
            "computed_at": datetime.now().isoformat(timespec="seconds"),
            "row_counts": {k: len(v) for k, v in feats.items()},
            "features": registry.catalog()}
    write_json(meta, out / "feature_metadata.json")
    if mode == "train":
        write_catalog(cfg, feats)
    log.info("Features (%s, as_of=%s): %s", mode, as_of, meta["row_counts"])
    return meta


if __name__ == "__main__":
    run()
