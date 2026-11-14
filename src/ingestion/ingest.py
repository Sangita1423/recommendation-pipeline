"""
Data ingestion.

Supported sources (configs/config.yaml -> ingestion.source):
  * synthetic  : generates documented synthetic CSVs (default, works offline)
  * movielens  : reads MovieLens ml-latest-small (ratings.csv, movies.csv) and
                 maps it to the pipeline schema (user details are synthesised
                 because MovieLens small has no user table)

Ingestion is deliberately "dumb": every column is read as string so that no
data is silently coerced or lost before validation. Each run writes
  data/raw/ingested/<table>.csv   (bronze copy + lineage columns)
  reports/ingestion_manifest.json (row counts, checksums, timestamps)
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.ingestion import generate_synthetic
from src.utils import ROOT, get_logger, load_config, p, write_json

log = get_logger("ingestion")
TABLES = ["users", "items", "interactions"]


def _md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare_movielens(cfg: dict, source_dir):
    ml = ROOT / cfg["ingestion"]["movielens_dir"]
    ratings = pd.read_csv(ml / "ratings.csv")
    movies = pd.read_csv(ml / "movies.csv")
    rng = np.random.default_rng(cfg["seed"])
    inter = pd.DataFrame({
        "user_id": "U" + ratings.userId.astype(str).str.zfill(5),
        "item_id": "I" + ratings.movieId.astype(str).str.zfill(5),
        "rating": ratings.rating.round().clip(1, 5),
        "timestamp": pd.to_datetime(ratings.timestamp, unit="s").dt.strftime("%Y-%m-%d %H:%M:%S"),
    })
    years = movies.title.str.extract(r"\((\d{4})\)\s*$")[0]
    items = pd.DataFrame({
        "item_id": "I" + movies.movieId.astype(str).str.zfill(5),
        "title": movies.title, "genres": movies.genres.replace("(no genres listed)", None),
        "release_year": years, "duration_min": None,
    })
    uids = inter.user_id.unique()
    users = pd.DataFrame({"user_id": uids, "age": rng.integers(16, 70, len(uids)),
                          "gender": rng.choice(["M", "F"], len(uids)),
                          "country": "unknown", "signup_date": "2000-01-01"})
    for name, df in {"users": users, "items": items, "interactions": inter}.items():
        df.to_csv(source_dir / f"{name}.csv", index=False)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    raw = p(cfg, "raw")
    source_dir = raw / "source"
    source_dir.mkdir(exist_ok=True)
    out_dir = raw / "ingested"
    out_dir.mkdir(exist_ok=True)

    source = cfg["ingestion"]["source"]
    if source == "synthetic":
        cfg_local = {**cfg, "paths": {**cfg["paths"], "raw": f"{cfg['paths']['raw']}/source"}}
        generate_synthetic.main(cfg_local)
    elif source == "movielens":
        _prepare_movielens(cfg, source_dir)
    else:
        raise ValueError(f"Unknown ingestion source: {source}")

    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {"source": source, "ingested_at": ingested_at, "tables": {}}
    for t in TABLES:
        src_file = source_dir / f"{t}.csv"
        df = pd.read_csv(src_file, dtype=str, keep_default_na=False, na_values=[""])
        missing = [c for c in _expected_columns(t) if c not in df.columns]
        if missing:
            raise ValueError(f"{t}: schema mismatch, missing columns {missing}")
        df["_source_file"] = src_file.name
        df["_ingested_at"] = ingested_at
        df.to_csv(out_dir / f"{t}.csv", index=False)
        manifest["tables"][t] = {"rows": int(len(df)), "columns": list(df.columns),
                                 "md5": _md5(src_file)}
        log.info("Ingested %-12s rows=%d", t, len(df))
    write_json(manifest, p(cfg, "reports") / "ingestion_manifest.json")
    return manifest


def _expected_columns(table: str) -> list[str]:
    return {"users": ["user_id", "age", "gender", "country", "signup_date"],
            "items": ["item_id", "title", "genres", "release_year", "duration_min"],
            "interactions": ["user_id", "item_id", "rating", "timestamp"]}[table]


if __name__ == "__main__":
    run()
