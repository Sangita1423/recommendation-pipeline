"""
Data preparation: cleaning, standardisation, imputation, de-duplication and
the time-based train/test split.

Outputs (data/processed/):
  users_clean, items_clean, interactions_clean, train, test, split_info.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, p, read_table, save_table, write_json

log = get_logger("preparation")
GENDER_MAP = {"m": "M", "male": "M", "f": "F", "female": "F", "other": "Other", "o": "Other"}


def clean_users(users: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    v = cfg["validation"]
    u = users[["user_id", "age", "gender", "country", "signup_date"]].copy()
    u["age"] = pd.to_numeric(u.age, errors="coerce")
    u.loc[~u.age.between(v["age_min"], v["age_max"]), "age"] = np.nan
    u["age_imputed"] = u.age.isna().astype(int)
    u["age"] = u.age.fillna(u.age.median()).round().astype(int)
    u["gender"] = u.gender.astype(str).str.strip().str.lower().map(GENDER_MAP).fillna("Unknown")
    u["country"] = u.country.fillna("Unknown").astype(str).str.strip().str.title()
    u["signup_date"] = pd.to_datetime(u.signup_date, format="%Y-%m-%d", errors="coerce")
    return u


def clean_items(items: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    i = items[["item_id", "title", "genres", "release_year", "duration_min"]].copy()
    i["title"] = i.title.astype(str).str.strip()
    i["genres"] = (i.genres.fillna("Unknown").astype(str).str.split("|")
                   .map(lambda gs: "|".join(sorted({g.strip().title() for g in gs if g.strip()}))))
    yr = pd.to_numeric(i.release_year, errors="coerce")
    yr[~yr.between(cfg["validation"]["release_year_min"], pd.Timestamp.now().year)] = np.nan
    i["release_year"] = yr.fillna(yr.median()).round().astype(int)
    dur = pd.to_numeric(i.duration_min, errors="coerce")
    i["duration_min"] = dur.fillna(dur.median() if dur.notna().any() else 0).round().astype(int)
    i["primary_genre"] = i.genres.str.split("|").str[0]
    return i


def clean_interactions(inter: pd.DataFrame) -> pd.DataFrame:
    x = inter[["user_id", "item_id", "rating", "timestamp"]].copy()
    x["rating"] = pd.to_numeric(x.rating).astype(float)
    x["timestamp"] = pd.to_datetime(x.timestamp, format="%Y-%m-%d %H:%M:%S")
    before = len(x)
    # A user may re-rate an item: keep the most recent opinion only.
    x = (x.sort_values("timestamp")
           .drop_duplicates(["user_id", "item_id"], keep="last")
           .reset_index(drop=True))
    log.info("Removed %d repeat user-item ratings (kept latest)", before - len(x))
    return x


def time_split(inter: pd.DataFrame, test_fraction: float):
    """Global temporal split: everything after the cutoff is test.

    This mimics production (train on the past, predict the future) and
    prevents leakage of future behaviour into training features.
    """
    cutoff = inter.timestamp.quantile(1 - test_fraction)
    train = inter[inter.timestamp <= cutoff].copy()
    test = inter[inter.timestamp > cutoff].copy()
    return train, test, cutoff


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    vdir, out = p(cfg, "validated"), p(cfg, "processed")
    users = clean_users(read_table(vdir, "users", dtype=str), cfg)
    items = clean_items(read_table(vdir, "items", dtype=str), cfg)
    inter = clean_interactions(read_table(vdir, "interactions", dtype=str))
    train, test, cutoff = time_split(inter, cfg["split"]["test_fraction"])

    for name, df in {"users_clean": users, "items_clean": items,
                     "interactions_clean": inter, "train": train, "test": test}.items():
        save_table(df, out, name, cfg)
    info = {
        "cutoff_timestamp": str(cutoff), "train_rows": len(train), "test_rows": len(test),
        "train_period": [str(train.timestamp.min()), str(train.timestamp.max())],
        "test_period": [str(test.timestamp.min()), str(test.timestamp.max())],
        "test_users": int(test.user_id.nunique()),
        "test_users_cold_start": int((~test.user_id.isin(train.user_id)).groupby(test.user_id).any().sum()),
        "test_items_cold_start": int(test.loc[~test.item_id.isin(train.item_id), "item_id"].nunique()),
    }
    write_json(info, out / "split_info.json")
    log.info("Time split at %s -> train=%d test=%d", cutoff, len(train), len(test))
    return info


if __name__ == "__main__":
    run()
