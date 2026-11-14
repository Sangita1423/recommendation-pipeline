"""
Data transformation: derived datasets built from the TRAIN split only
(so nothing from the test period leaks into what the models see).

Outputs (data/processed/):
  user_activity_summary   one row per user: counts, mean rating, first/last seen
  item_popularity         one row per item: counts, mean rating, popularity rank
  genre_summary           one row per genre
  user_item_matrix.npz    sparse CSR rating matrix  (+ user_index.csv, item_index.csv)
  user_item_matrix_sample a small dense slice, human-readable
"""
from __future__ import annotations

import pandas as pd
from scipy import sparse

from src.utils import get_logger, load_config, p, read_table, save_table

log = get_logger("transformation")


def user_activity_summary(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = g.agg(n_ratings=("rating", "size"), mean_rating=("rating", "mean"),
                std_rating=("rating", "std"), first_interaction=("timestamp", "min"),
                last_interaction=("timestamp", "max")).reset_index()
    out["active_days"] = (out.last_interaction - out.first_interaction).dt.days + 1
    out["std_rating"] = out.std_rating.fillna(0)
    return out


def item_popularity(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("item_id")
    out = g.agg(n_ratings=("rating", "size"), mean_rating=("rating", "mean"),
                n_users=("user_id", "nunique"), last_interaction=("timestamp", "max")).reset_index()
    out = items[["item_id", "title", "genres"]].merge(out, on="item_id", how="left")
    out["n_ratings"] = out.n_ratings.fillna(0).astype(int)
    out["popularity_rank"] = out.n_ratings.rank(ascending=False, method="min").astype(int)
    return out.sort_values("popularity_rank")


def genre_summary(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    x = train.merge(items[["item_id", "genres"]], on="item_id")
    x = x.assign(genre=x.genres.str.split("|")).explode("genre")
    return (x.groupby("genre").agg(n_ratings=("rating", "size"), mean_rating=("rating", "mean"),
                                   n_users=("user_id", "nunique"))
             .reset_index().sort_values("n_ratings", ascending=False))


def interaction_matrix(train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame):
    uidx = pd.Series(range(len(users)), index=users.user_id.values)
    iidx = pd.Series(range(len(items)), index=items.item_id.values)
    m = sparse.csr_matrix((train.rating.values,
                           (uidx[train.user_id].values, iidx[train.item_id].values)),
                          shape=(len(users), len(items)))
    return m, uidx, iidx


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    d = p(cfg, "processed")
    train = read_table(d, "train", parse_dates=["timestamp"])
    users = read_table(d, "users_clean")
    items = read_table(d, "items_clean")

    ua = user_activity_summary(train)
    ip = item_popularity(train, items)
    gs = genre_summary(train, items)
    m, uidx, iidx = interaction_matrix(train, users, items)

    save_table(ua, d, "user_activity_summary", cfg)
    save_table(ip, d, "item_popularity", cfg)
    save_table(gs, d, "genre_summary", cfg)
    sparse.save_npz(d / "user_item_matrix.npz", m)
    uidx.rename_axis("user_id").reset_index(name="row").to_csv(d / "user_index.csv", index=False)
    iidx.rename_axis("item_id").reset_index(name="col").to_csv(d / "item_index.csv", index=False)
    top_u = ua.nlargest(15, "n_ratings").user_id
    top_i = ip.head(15).item_id
    sample = (train[train.user_id.isin(top_u) & train.item_id.isin(top_i)]
              .pivot_table(index="user_id", columns="item_id", values="rating"))
    sample.to_csv(d / "user_item_matrix_sample.csv")

    density = m.nnz / (m.shape[0] * m.shape[1])
    log.info("Interaction matrix %s, nnz=%d, density=%.4f", m.shape, m.nnz, density)
    return {"matrix_shape": list(m.shape), "nnz": int(m.nnz), "density": round(density, 5)}


if __name__ == "__main__":
    run()
