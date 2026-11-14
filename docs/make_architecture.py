"""Draws docs/architecture.png — run:  python docs/make_architecture.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).parent / "architecture.png"

STAGES = [  # (x, title, module, outputs, colour)
    (0.0, "1. Ingestion", "src/ingestion", "synthetic / MovieLens CSV\n→ data/raw/ingested\n+ manifest (md5, rows)", "#4C78A8"),
    (1.0, "2. Validation", "src/validation", "12 interaction rules,\n9 user, 7 item rules\n→ data/validated\n→ data/errors (quarantine)", "#E45756"),
    (2.0, "3. Preparation", "src/preparation", "clean, impute, standardise,\ndedupe (keep latest)\ntime-based 80/20 split", "#F58518"),
    (3.0, "4. Transformation", "src/transformation", "user activity summary\nitem popularity, genres\nsparse user-item matrix", "#72B7B2"),
    (4.0, "5. Feature store", "src/features", "registry (26 features)\nuser / item / user×genre\ntrain & serving snapshots", "#54A24B"),
    (5.0, "6. Training", "src/training", "popularity, item-kNN,\nexplicit MF, implicit ALS,\nhybrid ALS+content", "#B279A2"),
    (6.0, "7. Evaluation", "src/evaluation", "P@K, R@K, NDCG@K,\nHitRate, coverage,\ndiversity, novelty,\nRMSE, MAE", "#9D755D"),
]

fig, ax = plt.subplots(figsize=(17, 6.4))
ax.set_xlim(-0.55, 6.55)
ax.set_ylim(-1.75, 2.2)
ax.axis("off")
w, h = 0.86, 1.55
for x, title, mod, out, c in STAGES:
    ax.add_patch(FancyBboxPatch((x - w / 2, 0.05), w, h, boxstyle="round,pad=0.02", fc="white", ec=c, lw=2.2))
    ax.add_patch(FancyBboxPatch((x - w / 2, 1.27), w, 0.33, boxstyle="round,pad=0.02", fc=c, ec=c))
    ax.text(x, 1.435, title, ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
    ax.text(x, 1.12, mod, ha="center", va="center", fontsize=8, family="monospace", color="#444")
    ax.text(x, 0.6, out, ha="center", va="center", fontsize=8.2, linespacing=1.35)
    if x < 6:
        ax.annotate("", xy=(x + 1 - w / 2, 0.83), xytext=(x + w / 2, 0.83),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.6))

# orchestration band
ax.add_patch(FancyBboxPatch((-0.45, 1.8), 6.9, 0.32, boxstyle="round,pad=0.02", fc="#EEF2F7", ec="#4C78A8", lw=1.2))
ax.text(3.0, 1.96, "8. Orchestration — run_pipeline.py (dependency-aware runner, retries, run log)  |  "
        "dags/recommendation_dag.py (Airflow)  |  dags/prefect_flow.py (Prefect)",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#24476B")

# storage + tracking band
stores = [(-0.45, 2.35, "Storage (CSV / Parquet)",
           "data/raw · data/validated · data/errors · data/processed · data/features", "#F2F2F2"),
          (2.05, 2.15, "Experiment tracking", "MLflow (mlruns/) — params, metrics,\nartifacts; local fallback tracker", "#F6EEF5"),
          (4.35, 2.1, "Outputs", "models/*.pkl + registry.json\nreports/ (validation, catalog, results)", "#F3EEE9")]
for x0, wd, t, d, c in stores:
    ax.add_patch(FancyBboxPatch((x0, -1.62), wd, 1.05, boxstyle="round,pad=0.02", fc=c, ec="#999", lw=1))
    ax.text(x0 + wd / 2, -0.78, t, ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(x0 + wd / 2, -1.17, d, ha="center", va="center", fontsize=8.2, linespacing=1.35)
for x in range(7):
    ax.annotate("", xy=(x, -0.55), xytext=(x, 0.03),
                arrowprops=dict(arrowstyle="<->", color="#AAA", lw=0.9, ls="--"))
ax.text(3.0, 2.32, "End-to-End Data Management Pipeline for a Movie Recommendation System",
        ha="center", fontsize=14, fontweight="bold")
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)
