"""
Ranking and rating metrics.

Ranking metrics are computed per user over the top-K list and averaged over
evaluated users (users who exist in train AND have >= 1 relevant test item).

  Precision@K  = |rec ∩ rel| / K
  Recall@K     = |rec ∩ rel| / |rel|
  NDCG@K       = DCG / IDCG with binary gains, DCG = Σ 1/log2(rank+1)
  HitRate@K    = share of users with at least one hit
  Coverage     = |∪ recommended items| / |catalogue|        (catalogue coverage)
  Diversity    = mean intra-list diversity, 1 - Jaccard(genres_a, genres_b)
                 averaged over all item pairs within a user's list
  Novelty      = mean self-information  -log2(popularity share) of recommended items
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def precision_at_k(rec: list, rel: set, k: int) -> float:
    return len(set(rec[:k]) & rel) / k


def recall_at_k(rec: list, rel: set, k: int) -> float:
    return len(set(rec[:k]) & rel) / len(rel) if rel else 0.0


def ndcg_at_k(rec: list, rel: set, k: int) -> float:
    dcg = sum(1 / np.log2(i + 2) for i, it in enumerate(rec[:k]) if it in rel)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0


def catalog_coverage(recs: dict, n_items: int) -> float:
    return len({i for lst in recs.values() for i in lst}) / n_items


def intra_list_diversity(recs: dict, item_genres: dict, k: int) -> float:
    vals = []
    for lst in recs.values():
        gs = [item_genres.get(i, set()) for i in lst[:k]]
        pair = [1 - len(a & b) / len(a | b) if (a | b) else 0 for a, b in combinations(gs, 2)]
        if pair:
            vals.append(np.mean(pair))
    return float(np.mean(vals)) if vals else 0.0


def novelty(recs: dict, item_pop_share: dict, k: int) -> float:
    vals = [-np.log2(item_pop_share.get(i, 1e-6) or 1e-6) for lst in recs.values() for i in lst[:k]]
    return float(np.mean(vals)) if vals else 0.0


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))
