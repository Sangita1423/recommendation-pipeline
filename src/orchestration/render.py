"""Renders a pipeline run record as an image: DAG graph (status-coloured) + Gantt timeline."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

COLORS = {"success": "#2E9E44", "failed": "#D62728", "upstream_failed": "#FF9F1C",
          "pending": "#B0B0B0"}

# fixed layout (column, row) for the DAG view
POS = {"ingest_data": (0, 1), "validate_data": (1, 1), "prepare_data": (2, 1),
       "transform_data": (3, 2), "build_features": (3, 1), "build_serving_features": (3, 0),
       "train_models": (4, 1.5), "evaluate_models": (5, 1.5)}


def render_run(record: dict, out_path) -> None:
    tasks = record["tasks"]
    fig = plt.figure(figsize=(15, 8.2))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.02, 0.42, 0.96, 0.48])
    ax.set_xlim(-0.6, 5.7)
    ax.set_ylim(-0.6, 2.6)
    ax.axis("off")
    w, h = 0.85, 0.42
    for t, s in tasks.items():
        x, y = POS.get(t, (0, 0))
        for u in s["upstream"]:
            ux, uy = POS.get(u, (0, 0))
            ax.annotate("", xy=(x - w / 2, y), xytext=(ux + w / 2, uy),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=1.3))
    for t, s in tasks.items():
        x, y = POS.get(t, (0, 0))
        c = COLORS.get(s["status"], "#B0B0B0")
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02",
                                    fc="white", ec=c, lw=2.5))
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), 0.06, h, boxstyle="square,pad=0",
                                    fc=c, ec=c))
        ax.text(x + 0.03, y + 0.06, t, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x + 0.03, y - 0.11, f"{s['status']}  ·  {s.get('duration_sec', 0):.1f}s",
                ha="center", va="center", fontsize=8, color=c)
    fig.text(0.02, 0.95, f"Pipeline run  {record['run_id']}", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.915, f"Status: {record['status'].upper()}   |   started {record['started_at']}"
             f"   |   total {record['duration_sec']:.1f}s   |   tasks: {len(tasks)}",
             fontsize=10, color=COLORS.get(record["status"], "#333"))

    gx = fig.add_axes([0.17, 0.06, 0.78, 0.30])
    names = list(tasks)
    for i, t in enumerate(names):
        s = tasks[t]
        gx.barh(i, max(s.get("duration_sec", 0), 0.05), left=s.get("start_offset", 0),
                color=COLORS.get(s["status"]), height=0.6)
        gx.text(s.get("start_offset", 0) + s.get("duration_sec", 0) + 0.2, i,
                f"{s.get('duration_sec', 0):.1f}s", va="center", fontsize=8)
    gx.set_yticks(range(len(names)))
    gx.set_yticklabels(names, fontsize=8)
    gx.invert_yaxis()
    gx.set_xlabel("seconds since run start", fontsize=9)
    gx.set_title("Task timeline (Gantt)", fontsize=10, loc="left")
    gx.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
