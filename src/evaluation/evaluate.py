"""
Model evaluation stage.

For each model in models/registry.json:
  * top-K recommendations for every evaluable test user (seen train items excluded)
  * Precision@K, Recall@K, NDCG@K, HitRate@K, Coverage, Diversity, Novelty
  * RMSE / MAE on all test ratings (explicit-rating prediction)
Metrics are logged back to the model's training run and written to
reports/model_comparison.{csv,md} plus a comparison chart.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation import metrics as M  # noqa: E402
from src.training.models import BaseRecommender  # noqa: E402
from src.training.train import build_context  # noqa: E402
from src.training.tracking import Tracker  # noqa: E402
from src.utils import ROOT, get_logger, load_config, p, read_json, read_table, write_json  # noqa: E402

log = get_logger("evaluation")


def evaluate_model(model, test, eval_users, relevant, item_genres, pop_share, k, n_items):
    recs = model.recommend(eval_users, k)
    P = [M.precision_at_k(recs[u], relevant[u], k) for u in eval_users]
    R = [M.recall_at_k(recs[u], relevant[u], k) for u in eval_users]
    N = [M.ndcg_at_k(recs[u], relevant[u], k) for u in eval_users]
    H = [float(len(set(recs[u]) & relevant[u]) > 0) for u in eval_users]
    yhat = model.predict(test.user_id.values, test.item_id.values)
    return recs, {
        f"precision_at_{k}": float(np.mean(P)), f"recall_at_{k}": float(np.mean(R)),
        f"ndcg_at_{k}": float(np.mean(N)), f"hit_rate_at_{k}": float(np.mean(H)),
        "coverage": M.catalog_coverage(recs, n_items),
        "diversity": M.intra_list_diversity(recs, item_genres, k),
        "novelty": M.novelty(recs, pop_share, k),
        "rmse": M.rmse(test.rating, yhat), "mae": M.mae(test.rating, yhat),
    }


def run(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    k, thr = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    d, reports = p(cfg, "processed"), p(cfg, "reports")
    ctx, train = build_context(cfg)
    test = read_table(d, "test", parse_dates=["timestamp"])

    rel = test[test.rating >= thr].groupby("user_id").item_id.apply(set)
    eval_users = [u for u in rel.index if u in set(train.user_id)]
    relevant = rel.to_dict()
    item_genres = ctx.items.set_index("item_id").genres.str.split("|").map(set).to_dict()
    cnt = train.item_id.value_counts()
    pop_share = (cnt / cnt.sum()).to_dict()
    registry = read_json(p(cfg, "models") / "registry.json")
    tracker = Tracker(cfg["tracking"]["experiment_name"], cfg["paths"]["mlruns"])

    rows, all_recs = [], {}
    for name, entry in registry["models"].items():
        model = BaseRecommender.load(ROOT / entry["artifact"])
        model.ctx.feature_store = ctx.feature_store
        recs, res = evaluate_model(model, test, eval_users, relevant, item_genres,
                                   pop_share, k, len(ctx.item_ids))
        all_recs[name] = recs
        with tracker.resume_run(entry["run_id"]):
            tracker.log_metrics({f"test_{m}": v for m, v in res.items()})
        rows.append({"model": name, **res})
        log.info("%-22s P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f cov=%.3f div=%.3f rmse=%.3f",
                 name, k, res[f"precision_at_{k}"], k, res[f"recall_at_{k}"], k,
                 res[f"ndcg_at_{k}"], res["coverage"], res["diversity"], res["rmse"])

    df = pd.DataFrame(rows)
    df.to_csv(reports / "model_comparison.csv", index=False)
    meta = {"k": k, "relevance_threshold": thr, "evaluated_users": len(eval_users),
            "test_ratings": len(test), "avg_relevant_per_user": float(rel.map(len).mean())}
    write_json({"meta": meta, "results": rows}, reports / "model_comparison.json")
    _write_markdown(df, meta, reports)
    _plot(df, k, reports)
    _sample_recs(all_recs, ctx, train, relevant, reports)
    return df


def _write_markdown(df, meta, reports):
    k = meta["k"]
    cols = ["model", f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", f"hit_rate_at_{k}",
            "coverage", "diversity", "novelty", "rmse", "mae"]
    head = ["Model", f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}", f"HitRate@{k}",
            "Coverage", "Diversity", "Novelty", "RMSE", "MAE"]
    lines = ["# Model Evaluation Results", "",
             f"K = {k}; relevant = test rating >= {meta['relevance_threshold']}; "
             f"evaluated users = {meta['evaluated_users']:,}; test ratings = {meta['test_ratings']:,}; "
             f"avg relevant items per user = {meta['avg_relevant_per_user']:.2f}.", "",
             "| " + " | ".join(head) + " |", "|---" + "|---:" * (len(head) - 1) + "|"]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join([f"`{r['model']}`"] + [f"{r[c]:.4f}" for c in cols[1:]]) + " |")
    best = df.loc[df[f"ndcg_at_{k}"].idxmax(), "model"]
    lines += ["", f"Best model by NDCG@{k}: **{best}**.", "",
              "Higher is better for all metrics except RMSE and MAE. RMSE/MAE for the popularity "
              "baseline use the item's Bayesian-average rating as the prediction."]
    (reports / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(df, k, reports):
    figdir = reports / "figures"
    figdir.mkdir(exist_ok=True)
    mets = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", "coverage", "diversity"]
    labels = [f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}", "Coverage", "Diversity"]
    colors = ["#9AA5B1", "#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    fig, axes = plt.subplots(1, len(mets), figsize=(16, 3.8))
    for ax, m, lab in zip(axes, mets, labels):
        ax.bar(range(len(df)), df[m], color=colors[:len(df)])
        ax.set_title(lab, fontsize=11)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([s.replace("_", "\n") for s in df.model], fontsize=7)
        for i, v in enumerate(df[m]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model comparison on the time-based test split", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "model_comparison.png", dpi=150)
    plt.close(fig)


def _sample_recs(all_recs, ctx, train, relevant, reports, n_users=3):
    titles = ctx.items.set_index("item_id")[["title", "genres"]]
    users = sorted(relevant, key=lambda u: -len(relevant[u]))[:n_users]
    lines = ["# Sample recommendations (top 5)", ""]
    for u in users:
        hist = train[train.user_id == u].nlargest(5, "rating").item_id
        lines += [f"## {u}", "", "Top-rated in train: " +
                  "; ".join(f"{titles.title[i]} [{titles.genres[i]}]" for i in hist), ""]
        for name, recs in all_recs.items():
            lst = recs.get(u, [])[:5]
            marks = [f"{titles.title[i]} [{titles.genres[i]}]{' ✔' if i in relevant[u] else ''}"
                     for i in lst]
            lines.append(f"- **{name}**: " + "; ".join(marks))
        lines.append("")
    lines.append("✔ = item the user actually rated >= 4 in the test period.")
    (reports / "sample_recommendations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
