"""
Feature registry: the single source of truth for every feature.

Each feature is declared once with its entity, type, description and the
logic used to compute it. The same definitions drive:
  * the offline feature tables in data/features/   (training)
  * the online-style lookup in FeatureStore          (prediction)
  * the auto-generated reports/feature_catalog.md   (documentation)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

FEATURE_VERSION = "v1"


@dataclass(frozen=True)
class FeatureDef:
    name: str
    entity: str          # user | item | user_genre
    dtype: str
    description: str
    logic: str
    source: str


FEATURES: list[FeatureDef] = [
    # ---------------- user features ----------------
    FeatureDef("user_rating_count", "user", "int", "Number of ratings the user has given",
               "count(rating) by user_id", "interactions"),
    FeatureDef("user_avg_rating", "user", "float", "User's mean rating (rating generosity)",
               "mean(rating) by user_id", "interactions"),
    FeatureDef("user_rating_std", "user", "float", "Spread of the user's ratings",
               "std(rating) by user_id, 0 if single rating", "interactions"),
    FeatureDef("user_ratings_last_30d", "user", "int", "Ratings in the 30 days before as_of",
               "count(rating) where timestamp > as_of - 30d", "interactions"),
    FeatureDef("user_days_since_last_interaction", "user", "int", "Recency of the user's activity",
               "as_of - max(timestamp), in days", "interactions"),
    FeatureDef("user_active_span_days", "user", "int", "Days between first and last interaction",
               "max(timestamp) - min(timestamp) + 1", "interactions"),
    FeatureDef("user_preferred_genre", "user", "category", "Genre the user rates most often",
               "argmax(count) over exploded item genres", "interactions + items"),
    FeatureDef("user_genre_diversity", "user", "float",
               "Distinct genres rated / total genres (breadth of taste)",
               "nunique(genre) / n_genres", "interactions + items"),
    FeatureDef("user_mainstreamness", "user", "float",
               "Mean popularity percentile of items the user rated",
               "mean(item_popularity_percentile) over user's items", "interactions"),
    FeatureDef("user_age", "user", "int", "Age (median-imputed when missing/invalid)",
               "users.age", "users"),
    FeatureDef("user_gender", "user", "category", "Standardised gender (M/F/Other/Unknown)",
               "users.gender", "users"),
    FeatureDef("user_tenure_days", "user", "int", "Days since sign-up",
               "as_of - signup_date", "users"),
    # ---------------- item features ----------------
    FeatureDef("item_rating_count", "item", "int", "Number of ratings (raw popularity)",
               "count(rating) by item_id", "interactions"),
    FeatureDef("item_unique_users", "item", "int", "Distinct users who rated the item",
               "nunique(user_id) by item_id", "interactions"),
    FeatureDef("item_avg_rating", "item", "float", "Mean rating of the item",
               "mean(rating) by item_id", "interactions"),
    FeatureDef("item_bayesian_avg_rating", "item", "float",
               "Mean rating shrunk to global mean (robust for few ratings)",
               "(C*m + sum(r)) / (C + n), C = median item count, m = global mean", "interactions"),
    FeatureDef("item_popularity_percentile", "item", "float", "Popularity percentile (1 = most popular)",
               "rank_pct(item_rating_count)", "interactions"),
    FeatureDef("item_ratings_last_30d", "item", "int", "Ratings in the 30 days before as_of (trending)",
               "count(rating) where timestamp > as_of - 30d", "interactions"),
    FeatureDef("item_days_since_last_interaction", "item", "int", "Recency of item activity",
               "as_of - max(timestamp), in days", "interactions"),
    FeatureDef("item_primary_genre", "item", "category", "First listed genre",
               "items.genres.split('|')[0]", "items"),
    FeatureDef("item_genre_count", "item", "int", "Number of genres tagged", "len(genres)", "items"),
    FeatureDef("item_age_years", "item", "int", "Years since release", "year(as_of) - release_year", "items"),
    FeatureDef("item_duration_min", "item", "int", "Runtime in minutes", "items.duration_min", "items"),
    # ---------------- user x genre (interaction) features ----------------
    FeatureDef("ug_share", "user_genre", "float", "Share of the user's ratings that fall in the genre",
               "count(user, genre) / count(user)", "interactions + items"),
    FeatureDef("ug_avg_rating", "user_genre", "float", "User's mean rating within the genre",
               "mean(rating) by (user_id, genre)", "interactions + items"),
    FeatureDef("ug_affinity", "user_genre", "float",
               "Combined preference score used by the hybrid model",
               "ug_share * (ug_avg_rating - user_avg_rating + 1)", "interactions + items"),
]


def catalog() -> list[dict]:
    return [asdict(f) for f in FEATURES]


def names(entity: str) -> list[str]:
    return [f.name for f in FEATURES if f.entity == entity]
