"""Shared helpers: config loading, logging, path resolution and table I/O."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def p(cfg: dict, key: str) -> Path:
    """Resolve a configured directory relative to the project root and create it."""
    path = ROOT / cfg["paths"][key]
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    (ROOT / "logs").mkdir(exist_ok=True)
    fh = logging.FileHandler(ROOT / "logs" / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def save_table(df: pd.DataFrame, directory: Path, name: str, cfg: dict) -> Path:
    fmt = cfg.get("storage", {}).get("format", "csv")
    if fmt == "parquet" and _parquet_available():
        out = directory / f"{name}.parquet"
        df.to_parquet(out, index=False)
    else:
        out = directory / f"{name}.csv"
        df.to_csv(out, index=False)
    return out


def read_table(directory: Path, name: str, **kwargs) -> pd.DataFrame:
    pq = directory / f"{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(directory / f"{name}.csv", **kwargs)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
