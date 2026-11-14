"""
Declarative validation rules.

Each rule is a small object with a name, a severity and a function that
returns a boolean Series marking the FAILING rows.

  severity = "error"   -> row is quarantined to data/errors/<table>_errors.csv
  severity = "warning" -> row is kept, issue is logged and fixed in preparation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

TS_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"


@dataclass
class Rule:
    name: str
    severity: str           # "error" | "warning"
    description: str
    check: Callable[[pd.DataFrame], pd.Series]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _missing(col):
    return lambda df: df[col].isna() | (df[col].astype(str).str.strip() == "")


def _not_numeric(col):
    return lambda df: df[col].notna() & _num(df[col]).isna()


def user_rules(v: dict) -> list[Rule]:
    return [
        Rule("missing_user_id", "error", "user_id is null/blank", _missing("user_id")),
        Rule("invalid_user_id_format", "error", "user_id must match U#####",
             lambda df: df.user_id.notna() & ~df.user_id.astype(str).str.fullmatch(r"U\d{5}")),
        Rule("duplicate_user_id", "error", "user_id already seen (duplicate record)",
             lambda df: df.user_id.notna() & df.duplicated("user_id", keep="first")),
        Rule("age_not_numeric", "warning", "age is not a number", _not_numeric("age")),
        Rule("age_out_of_range", "warning", f"age outside [{v['age_min']}, {v['age_max']}]",
             lambda df: _num(df.age).notna() & ~_num(df.age).between(v["age_min"], v["age_max"])),
        Rule("missing_age", "warning", "age is null", _missing("age")),
        Rule("missing_country", "warning", "country is null", _missing("country")),
        Rule("invalid_signup_date", "warning", "signup_date not parseable as YYYY-MM-DD",
             lambda df: df.signup_date.notna() &
             pd.to_datetime(df.signup_date, format=DATE_FORMAT, errors="coerce").isna()),
        Rule("nonstandard_gender", "warning", "gender not in {M, F, Other}",
             lambda df: ~df.gender.isin(["M", "F", "Other"])),
    ]


def item_rules(v: dict) -> list[Rule]:
    return [
        Rule("missing_item_id", "error", "item_id is null/blank", _missing("item_id")),
        Rule("invalid_item_id_format", "error", "item_id must match I#####",
             lambda df: df.item_id.notna() & ~df.item_id.astype(str).str.fullmatch(r"I\d{5}")),
        Rule("duplicate_item_id", "error", "item_id already seen (duplicate record)",
             lambda df: df.item_id.notna() & df.duplicated("item_id", keep="first")),
        Rule("missing_title", "error", "title is null", _missing("title")),
        Rule("missing_genres", "warning", "genres is null", _missing("genres")),
        Rule("release_year_out_of_range", "warning",
             f"release_year outside [{v['release_year_min']}, current year]",
             lambda df: _num(df.release_year).notna() &
             ~_num(df.release_year).between(v["release_year_min"], pd.Timestamp.now().year)),
        Rule("nonstandard_genre_case", "warning", "genres not Title-Case",
             lambda df: df.genres.notna() &
             (df.genres.astype(str) != df.genres.astype(str).str.title())),
    ]


def interaction_rules(v: dict, valid_users: set, valid_items: set) -> list[Rule]:
    max_ts = pd.Timestamp.now() if v["max_timestamp"] == "now" else pd.Timestamp(v["max_timestamp"])
    min_ts = pd.Timestamp(v["min_timestamp"])
    ts = lambda df: pd.to_datetime(df.timestamp, format=TS_FORMAT, errors="coerce")  # noqa: E731
    rating = lambda df: _num(df.rating)  # noqa: E731
    return [
        Rule("missing_user_id", "error", "user_id is null", _missing("user_id")),
        Rule("missing_item_id", "error", "item_id is null", _missing("item_id")),
        Rule("missing_rating", "error", "rating is null", _missing("rating")),
        Rule("missing_timestamp", "error", "timestamp is null", _missing("timestamp")),
        Rule("rating_wrong_type", "error", "rating is not numeric", _not_numeric("rating")),
        Rule("rating_out_of_range", "error", f"rating outside [{v['rating_min']}, {v['rating_max']}]",
             lambda df: rating(df).notna() & ~rating(df).between(v["rating_min"], v["rating_max"])),
        Rule("timestamp_unparseable", "error", f"timestamp not in format {TS_FORMAT}",
             lambda df: df.timestamp.notna() & (df.timestamp.astype(str).str.strip() != "") & ts(df).isna()),
        Rule("timestamp_in_future", "error", "timestamp later than pipeline run time",
             lambda df: ts(df) > max_ts),
        Rule("timestamp_too_old", "error", f"timestamp earlier than {v['min_timestamp']}",
             lambda df: ts(df) < min_ts),
        Rule("orphan_user_id", "error", "user_id not present in validated users (referential integrity)",
             lambda df: df.user_id.notna() & ~df.user_id.isin(valid_users)),
        Rule("orphan_item_id", "error", "item_id not present in validated items (referential integrity)",
             lambda df: df.item_id.notna() & ~df.item_id.isin(valid_items)),
        Rule("exact_duplicate", "error", "row is an exact duplicate of an earlier row",
             lambda df: df.duplicated(["user_id", "item_id", "rating", "timestamp"], keep="first")),
    ]
