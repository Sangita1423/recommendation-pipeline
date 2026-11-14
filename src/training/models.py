"""
Recommender models. All share one interface:

    fit(train_df, ctx)            -> self
    score_all()                   -> (n_users, n_items) score matrix
    recommend(user_ids, k)        -> {user_id: [item_id, ...]}   (seen items excluded)
    predict(user_ids, item_ids)   -> predicted ratings (for RMSE / MAE)

`ctx` is a TrainingContext holding the user/item index, item metadata and the
feature store, so every model sees exactly the same id space.

Models
------
1. PopularityRecommender   (baseline)  time-decayed rating counts
2. ItemKNNRecommender      (personalised) item-item CF on implicit co-occurrence
3. MatrixFactorization     (personalised) explicit biased MF (Funk-SVD, SGD) -> rating prediction
4. ImplicitALS             (personalised) implicit-feedback MF (Hu, Koren & Volinsky 2008)
5. HybridRecommender       (personalised) ALS + content (genre affinity features) + popularity
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.features.build_features import FeatureStore


@dataclass
class TrainingContext:
    user_index: pd.Series          # user_id -> row
    item_index: pd.Series          # item_id -> col
    items: pd.DataFrame            # cleaned item table (genres)
    as_of: pd.Timestamp
    feature_store: FeatureStore | None = None

    @property
    def item_ids(self) -> np.ndarray:
        return self.item_index.index.values

    def train_matrix(self, train: pd.DataFrame) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            (train.rating.values.astype(float),
             (self.user_index[train.user_id].values, self.item_index[train.item_id].values)),
            shape=(len(self.user_index), len(self.item_index)))

    def genre_matrix(self) -> tuple[np.ndarray, list[str]]:
        g = self.items.set_index("item_id").reindex(self.item_ids).genres.str.get_dummies("|")
        return g.values.astype(float), list(g.columns)


class BaseRecommender:
    name = "base"

    def fit(self, train: pd.DataFrame, ctx: TrainingContext):
        self.ctx = ctx
        self.R = ctx.train_matrix(train)
        self.global_mean = float(train.rating.mean())
        return self

    def score_all(self) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_ids, k: int = 10) -> dict:
        rows = self.ctx.user_index.reindex(user_ids)
        known = rows.notna()
        scores = self.score_all()
        scores = np.where(self.R.toarray() > 0, -np.inf, scores)       # exclude seen
        out = {}
        fallback = np.argsort(-PopularityRecommender.raw_popularity(self.R))[:k * 3]
        for uid, r in rows.items():
            if pd.isna(r):                                              # cold-start user
                out[uid] = list(self.ctx.item_ids[fallback[:k]])
                continue
            s = scores[int(r)]
            top = np.argpartition(-s, k)[:k]
            out[uid] = list(self.ctx.item_ids[top[np.argsort(-s[top])]])
        return out

    def predict(self, user_ids, item_ids) -> np.ndarray:
        return np.full(len(user_ids), self.global_mean)

    def params(self) -> dict:
        return {}

    def save(self, path: Path):
        ctx_fs, self.ctx.feature_store = self.ctx.feature_store, None
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        self.ctx.feature_store = ctx_fs

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ------------------------------------------------------------------ baseline
class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def __init__(self, min_ratings: int = 5, recency_half_life_days: float | None = 180):
        self.min_ratings = min_ratings
        self.half_life = recency_half_life_days

    @staticmethod
    def raw_popularity(R) -> np.ndarray:
        return np.asarray((R > 0).sum(0)).ravel()

    def fit(self, train, ctx):
        super().fit(train, ctx)
        if self.half_life:
            age = (ctx.as_of - train.timestamp).dt.total_seconds() / 86400
            w = np.power(0.5, age / self.half_life)
        else:
            w = np.ones(len(train))
        pop = pd.Series(w.values if hasattr(w, "values") else w).groupby(train.item_id.values).sum()
        counts = train.groupby("item_id").size()
        pop[counts.reindex(pop.index) < self.min_ratings] = 0
        self.item_scores = pop.reindex(ctx.item_ids).fillna(0).values
        n = counts.reindex(ctx.item_ids).fillna(0).values
        sums = train.groupby("item_id").rating.sum().reindex(ctx.item_ids).fillna(0).values
        C = np.median(n[n > 0])
        self.item_mean = (C * self.global_mean + sums) / (C + n)         # Bayesian average
        return self

    def score_all(self):
        return np.tile(self.item_scores, (self.R.shape[0], 1))

    def predict(self, user_ids, item_ids):
        cols = self.ctx.item_index.reindex(item_ids)
        return np.array([self.item_mean[int(c)] if pd.notna(c) else self.global_mean for c in cols])

    def params(self):
        return {"min_ratings": self.min_ratings, "recency_half_life_days": self.half_life}


# ------------------------------------------------------------------ item-kNN CF
class ItemKNNRecommender(BaseRecommender):
    name = "item_knn_cf"

    def __init__(self, n_neighbors: int = 50, shrinkage: float = 10):
        self.n_neighbors, self.shrinkage = n_neighbors, shrinkage

    def fit(self, train, ctx):
        super().fit(train, ctx)
        R = self.R.tocsr().astype(float)
        counts = np.diff(R.indptr)
        self.user_mean = np.divide(np.asarray(R.sum(1)).ravel(), counts,
                                   out=np.full(R.shape[0], self.global_mean), where=counts > 0)
        C = R.copy()
        C.data -= np.repeat(self.user_mean, counts)                    # mean-centred (for ratings)
        self.C = C
        B = (R > 0).astype(float)                                      # implicit: who watched what
        co = (B.T @ B).toarray()
        deg = np.diag(co).copy()
        sim = co / (np.sqrt(np.outer(deg, deg)) + 1e-9)                # cosine on co-occurrence
        sim *= co / (co + self.shrinkage)                              # shrink low-support sims
        np.fill_diagonal(sim, 0)
        k = min(self.n_neighbors, sim.shape[0] - 1)
        thresh = -np.sort(-sim, axis=1)[:, k - 1:k]
        sim[sim < thresh] = 0                                          # keep top-k neighbours
        self.S = sim
        return self

    def score_all(self):
        B = (self.R > 0).astype(float)
        return np.asarray(B @ self.S.T)

    def predict(self, user_ids, item_ids):
        rows = self.ctx.user_index.reindex(user_ids).values
        cols = self.ctx.item_index.reindex(item_ids).values
        out = np.full(len(rows), self.global_mean)
        for n, (r, c) in enumerate(zip(rows, cols)):
            if pd.isna(r) or pd.isna(c):
                continue
            r, c = int(r), int(c)
            row = self.C.getrow(r)
            w = self.S[c, row.indices]
            denom = np.abs(w).sum()
            dev = (w @ row.data) / denom if denom > 0 else 0.0
            out[n] = np.clip(self.user_mean[r] + dev, 1, 5)
        return out

    def params(self):
        return {"n_neighbors": self.n_neighbors, "shrinkage": self.shrinkage}


# ------------------------------------------------------------------ matrix factorisation
class MatrixFactorization(BaseRecommender):
    name = "matrix_factorization"

    def __init__(self, n_factors=32, n_epochs=30, learning_rate=0.01, regularization=0.05, seed=42):
        self.f, self.epochs, self.lr, self.reg, self.seed = (n_factors, n_epochs, learning_rate,
                                                             regularization, seed)

    def fit(self, train, ctx):
        super().fit(train, ctx)
        rng = np.random.default_rng(self.seed)
        nu, ni = self.R.shape
        u = ctx.user_index[train.user_id].values
        i = ctx.item_index[train.item_id].values
        r = train.rating.values.astype(float)
        P = rng.normal(0, 0.1, (nu, self.f))
        Q = rng.normal(0, 0.1, (ni, self.f))
        bu, bi, mu = np.zeros(nu), np.zeros(ni), self.global_mean
        lr, reg = self.lr, self.reg
        self.loss_history = []
        for _ in range(self.epochs):
            order = rng.permutation(len(r))
            sq = 0.0
            for idx in order:
                uu, ii = u[idx], i[idx]
                pu, qi = P[uu], Q[ii]
                e = r[idx] - (mu + bu[uu] + bi[ii] + pu @ qi)
                sq += e * e
                bu[uu] += lr * (e - reg * bu[uu])
                bi[ii] += lr * (e - reg * bi[ii])
                P[uu] = pu + lr * (e * qi - reg * pu)
                Q[ii] = qi + lr * (e * pu - reg * qi)
            self.loss_history.append(float(np.sqrt(sq / len(r))))
        self.P, self.Q, self.bu, self.bi, self.mu = P, Q, bu, bi, mu
        return self

    def score_all(self):
        return self.mu + self.bu[:, None] + self.bi[None, :] + self.P @ self.Q.T

    def predict(self, user_ids, item_ids):
        rows = self.ctx.user_index.reindex(user_ids).values
        cols = self.ctx.item_index.reindex(item_ids).values
        out = np.full(len(rows), self.mu)
        ok_u, ok_i = ~pd.isna(rows), ~pd.isna(cols)
        both = ok_u & ok_i
        ru, ci = rows[both].astype(int), cols[both].astype(int)
        out[both] = self.mu + self.bu[ru] + self.bi[ci] + np.einsum("ij,ij->i", self.P[ru], self.Q[ci])
        only_i = ~ok_u & ok_i
        out[only_i] = self.mu + self.bi[cols[only_i].astype(int)]
        return np.clip(out, 1, 5)

    def params(self):
        return {"n_factors": self.f, "n_epochs": self.epochs,
                "learning_rate": self.lr, "regularization": self.reg}


# ------------------------------------------------------------------ implicit ALS
class ImplicitALS(BaseRecommender):
    """Weighted matrix factorisation for implicit feedback.
    Preference p_ui = 1 if rated, confidence c_ui = 1 + alpha * rating.
    Optimised for ranking ("will the user engage?") rather than rating error."""
    name = "implicit_als"

    def __init__(self, n_factors=64, n_iterations=15, regularization=0.1, alpha=10.0, seed=42):
        self.f, self.iters, self.reg, self.alpha, self.seed = (n_factors, n_iterations,
                                                               regularization, alpha, seed)

    @staticmethod
    def _solve(Cui: sparse.csr_matrix, Y: np.ndarray, reg: float) -> np.ndarray:
        YtY = Y.T @ Y
        X = np.zeros((Cui.shape[0], Y.shape[1]))
        lam = reg * np.eye(Y.shape[1])
        for u in range(Cui.shape[0]):
            s, e = Cui.indptr[u], Cui.indptr[u + 1]
            if s == e:
                continue
            idx, conf = Cui.indices[s:e], Cui.data[s:e]
            Yu = Y[idx]
            A = YtY + (Yu.T * (conf - 1)) @ Yu + lam
            X[u] = np.linalg.solve(A, (Yu.T * conf).sum(1))
        return X

    def fit(self, train, ctx):
        super().fit(train, ctx)
        rng = np.random.default_rng(self.seed)
        Cui = self.R.copy().tocsr()
        Cui.data = 1 + self.alpha * Cui.data / 5.0
        Ciu = Cui.T.tocsr()
        self.X = rng.normal(0, 0.01, (Cui.shape[0], self.f))
        self.Y = rng.normal(0, 0.01, (Cui.shape[1], self.f))
        for _ in range(self.iters):
            self.X = self._solve(Cui, self.Y, self.reg)
            self.Y = self._solve(Ciu, self.X, self.reg)
        return self

    def score_all(self):
        return self.X @ self.Y.T

    def params(self):
        return {"n_factors": self.f, "n_iterations": self.iters,
                "regularization": self.reg, "alpha": self.alpha}


# ------------------------------------------------------------------ hybrid
class HybridRecommender(BaseRecommender):
    """Weighted blend of z-scored collaborative scores (implicit ALS), content
    genre affinity (feature store: ug_affinity) and log-popularity. Rating
    predictions (RMSE/MAE) come from the explicit MF model."""
    name = "hybrid_als_content"

    def __init__(self, cf: ImplicitALS, mf: MatrixFactorization, w_cf=0.6, w_content=0.3, w_pop=0.1):
        self.cf, self.mf = cf, mf
        self.w_cf, self.w_content, self.w_pop = w_cf, w_content, w_pop

    def fit(self, train, ctx):
        super().fit(train, ctx)
        G, genres = ctx.genre_matrix()
        A = ctx.feature_store.genre_affinity_matrix(ctx.user_index, genres)
        Gn = G / np.maximum(G.sum(1, keepdims=True), 1)
        self.content = A @ Gn.T
        self.logpop = np.log1p(PopularityRecommender.raw_popularity(self.R))
        return self

    @staticmethod
    def _z(x, axis=1):
        return (x - x.mean(axis, keepdims=True)) / (x.std(axis, keepdims=True) + 1e-9)

    def score_all(self):
        return (self.w_cf * self._z(self.cf.score_all()) + self.w_content * self._z(self.content)
                + self.w_pop * self._z(np.tile(self.logpop, (self.R.shape[0], 1))))

    def predict(self, user_ids, item_ids):
        return self.mf.predict(user_ids, item_ids)

    def params(self):
        return {"w_cf": self.w_cf, "w_content": self.w_content, "w_pop": self.w_pop,
                **{f"cf_{k}": v for k, v in self.cf.params().items()}}
