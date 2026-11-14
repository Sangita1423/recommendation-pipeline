"""
Synthetic data generator for a movie-streaming platform.

Why synthetic?  It lets us (1) control the ground-truth structure so a
personalised model *should* beat popularity, and (2) inject known data
quality problems so the validation layer can be tested end-to-end.

Generative story
----------------
* Every item has 1-3 genres, a latent "quality" and a Zipf-like popularity.
* Every user has a Dirichlet genre-affinity vector and a personal rating bias.
* A user picks items with probability  ~ popularity * exp(6 * affinity . genres)
  and rates them  rating = 3 + quality + user_bias + 2.2 * (affinity match) + noise,
  rounded and clipped to 1..5.
* Interaction timestamps start after the user's signup date.

Dirty records (``dirty_fraction`` of each table) are then injected:
missing values, exact duplicates, inconsistent formats, out-of-range
ratings, non-numeric ratings, unparseable / future timestamps and
orphan foreign keys.  All are documented in docs/synthetic_data.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, p

log = get_logger("ingestion.synthetic")

GENRES = ["Action", "Comedy", "Drama", "Thriller", "Romance", "Sci-Fi",
          "Horror", "Documentary", "Animation", "Fantasy", "Crime", "Adventure"]
COUNTRIES = ["India", "USA", "UK", "Germany", "Brazil", "Japan", "Canada", "Australia"]
ADJ = ["Silent", "Last", "Hidden", "Golden", "Broken", "Midnight", "Electric",
       "Lost", "Crimson", "Frozen", "Wild", "Secret", "Iron", "Paper", "Velvet"]
POP_EXPONENT = 0.7      # Zipf exponent of item popularity (long tail)
TASTE_STRENGTH = 6.0    # how strongly genre taste drives which items a user picks
NOUN = ["River", "Empire", "Signal", "Garden", "Promise", "Horizon", "Mirror",
        "Station", "Kingdom", "Echo", "Voyage", "Code", "Harbor", "Storm", "Letter"]


def generate(cfg: dict) -> dict[str, pd.DataFrame]:
    s = cfg["synthetic"]
    rng = np.random.default_rng(cfg["seed"])
    n_u, n_i, n_g = s["n_users"], s["n_items"], s["n_genres"]
    genres = GENRES[:n_g]
    start, end = pd.Timestamp(s["start_date"]), pd.Timestamp(s["end_date"])
    span_days = (end - start).days

    # ---------------- users ----------------
    user_ids = [f"U{i:05d}" for i in range(1, n_u + 1)]
    signup = start + pd.to_timedelta(rng.integers(0, int(span_days * 0.7), n_u), unit="D")
    users = pd.DataFrame({
        "user_id": user_ids,
        "age": rng.integers(16, 70, n_u),
        "gender": rng.choice(["M", "F", "Other"], n_u, p=[0.5, 0.46, 0.04]),
        "country": rng.choice(COUNTRIES, n_u),
        "signup_date": signup.strftime("%Y-%m-%d"),
    })
    affinity = rng.dirichlet(np.full(n_g, 0.35), n_u)          # sparse tastes
    user_bias = rng.normal(0, 0.35, n_u)
    activity = rng.lognormal(0, 0.9, n_u)                       # heavy vs light users

    # ---------------- items ----------------
    item_ids = [f"I{i:05d}" for i in range(1, n_i + 1)]
    item_genre = np.zeros((n_i, n_g))
    genre_str = []
    for j in range(n_i):
        k = rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        g = rng.choice(n_g, k, replace=False)
        item_genre[j, g] = 1
        genre_str.append("|".join(genres[x] for x in sorted(g)))
    titles = [f"The {rng.choice(ADJ)} {rng.choice(NOUN)}" for _ in range(n_i)]
    titles = [f"{t} ({k})" if titles.count(t) > 1 else t for k, t in enumerate(titles)]
    items = pd.DataFrame({
        "item_id": item_ids,
        "title": titles,
        "genres": genre_str,
        "release_year": rng.integers(1970, 2026, n_i),
        "duration_min": rng.integers(80, 180, n_i),
    })
    quality = rng.normal(0, 0.45, n_i)
    popularity = 1.0 / np.arange(1, n_i + 1) ** POP_EXPONENT
    rng.shuffle(popularity)

    # ---------------- interactions ----------------
    n_per_user = np.maximum(5, (activity / activity.sum() * s["n_interactions"]).astype(int))
    genre_norm = item_genre / item_genre.sum(1, keepdims=True)
    rows = []
    for u in range(n_u):
        match = genre_norm @ affinity[u]                        # 0..1 taste match
        prob = popularity * np.exp(TASTE_STRENGTH * match)
        prob /= prob.sum()
        n = min(n_per_user[u], n_i // 2)
        chosen = rng.choice(n_i, n, replace=False, p=prob)
        r = 3.0 + quality[chosen] + user_bias[u] + 2.2 * (match[chosen] - match.mean()) \
            + rng.normal(0, 0.55, n)
        r = np.clip(np.rint(r), 1, 5)
        u_start = signup[u]
        offs = rng.uniform(0, max(1, (end - u_start).total_seconds()), n)
        ts = u_start + pd.to_timedelta(offs, unit="s")
        for it, rt, t in zip(chosen, r, ts):
            rows.append((user_ids[u], item_ids[it], float(rt), t.strftime("%Y-%m-%d %H:%M:%S")))
    inter = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    inter = inter.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    log.info("Clean synthetic data: %d users, %d items, %d interactions",
             len(users), len(items), len(inter))

    users, items, inter = _inject_dirt(users, items, inter, s["dirty_fraction"], rng)
    return {"users": users, "items": items, "interactions": inter}


def _inject_dirt(users, items, inter, frac, rng):
    """Corrupt a small, documented share of records so validation has work to do."""
    users = users.astype(object).copy()
    items = items.astype(object).copy()
    inter = inter.astype(object).copy()

    def pick(df, f=frac):
        return rng.choice(df.index, max(1, int(len(df) * f)), replace=False)

    # users: missing age/country, impossible ages, inconsistent gender labels, duplicates
    users.loc[pick(users), "age"] = np.nan
    idx = pick(users, frac / 2)
    users.loc[idx, "age"] = rng.choice([-3, 0, 142, 230, "unknown"], len(idx))
    users.loc[pick(users), "country"] = None
    gidx = pick(users, 0.1)
    users.loc[gidx, "gender"] = users.loc[gidx, "gender"].map(
        {"M": " male", "F": "female ", "Other": "other"})
    users.loc[pick(users, frac / 4), "signup_date"] = "31/02/2024"   # bad date format
    users = pd.concat([users, users.loc[pick(users, frac / 2)]], ignore_index=True)

    # items: missing genres, bad release year, lower-case genres, duplicates
    items.loc[pick(items), "genres"] = None
    idx = pick(items, frac / 2)
    items.loc[idx, "release_year"] = rng.choice([1066, 3021], len(idx))
    gi = pick(items, 0.05)
    items.loc[gi, "genres"] = items.loc[gi, "genres"].str.lower()
    items = pd.concat([items, items.loc[pick(items, frac / 2)]], ignore_index=True)

    # interactions
    idx = pick(inter, frac / 2)
    inter.loc[idx, "rating"] = rng.choice([0, 6, 7, -1, 10], len(idx))
    idx = pick(inter, frac / 4)
    inter.loc[idx, "rating"] = rng.choice(["five", "N/A", "4 stars"], len(idx))
    inter.loc[pick(inter, frac / 4), "rating"] = np.nan
    idx = pick(inter, frac / 4)
    inter.loc[idx, "timestamp"] = rng.choice(["not_a_date", "2024-13-45 10:00:00", "", "12/05/2024"], len(idx))
    inter.loc[pick(inter, frac / 4), "timestamp"] = "2031-06-01 12:00:00"      # future
    inter.loc[pick(inter, frac / 8), "timestamp"] = "1999-01-01 00:00:00"      # before platform
    idx = pick(inter, frac / 4)
    inter.loc[idx, "user_id"] = rng.choice(["U99999", "U88888", "UXXXXX"], len(idx))
    idx = pick(inter, frac / 4)
    inter.loc[idx, "item_id"] = rng.choice(["I99999", "I77777"], len(idx))
    inter.loc[pick(inter, frac / 8), "user_id"] = None
    dup = inter.loc[pick(inter, frac / 2)]
    inter = pd.concat([inter, dup], ignore_index=True)
    # repeat ratings: same user-item re-rated later (not an error -> dedup keeps latest)
    rr = inter.loc[pick(inter, frac / 2)].copy()
    rr["timestamp"] = pd.to_datetime(rr["timestamp"], errors="coerce") + pd.Timedelta(days=3)
    rr = rr.dropna(subset=["timestamp"])
    rr["timestamp"] = rr["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    inter = pd.concat([inter, rr], ignore_index=True)
    inter = inter.sample(frac=1, random_state=7).reset_index(drop=True)
    return users, items, inter


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    raw = p(cfg, "raw")
    for name, df in generate(cfg).items():
        df.to_csv(raw / f"{name}.csv", index=False)
        log.info("Wrote %s (%d rows) -> %s", name, len(df), raw / f"{name}.csv")


if __name__ == "__main__":
    main()
