"""Generator comparison plot across the four pipeline stages.

Reads:
    evaluation/summary/{pretrain,transfer,pretrain_rl,transfer_rl}_summary.json

Outputs:
    img/generator_comparison.png

Usage::

    uv run python -m src.scripts.plot_generator_comparison
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = ROOT / "evaluation" / "summary"
OUT = ROOT / "img"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)
# Config
STAGES = [
    ("Pretrain", "pretrain_summary.json"),
    ("Transfer", "transfer_summary.json"),
    ("Pretrain-RL", "pretrain_rl_summary.json"),
    ("Transfer-RL", "transfer_rl_summary.json"),
]

# Metrics to plot, in display order.
# (display_name, json_key, format_str, higher_is_better)
METRIC_SPEC = [
    ("Validity", "validity", ".3f", True),
    ("Uniqueness", "uniqueness", ".3f", True),
    ("Novelty", "novelty", ".3f", True),
    ("Diversity", "diversity", ".3f", True),
    ("SA Score", "sa_score_mean", ".2f", False),
    ("GNN P(act)", "hiv_mean_prob", ".3f", True),
    ("GNN Active %", "hiv_active_frac", ".1%", True),
    ("FP P(act)", "fp_mean_prob", ".3f", True),
    ("FP Active %", "fp_active_frac", ".1%", True),
    ("GNN-FP Agree", "gnn_fp_agreement", ".1%", True),
]


def _load_summaries() -> list[tuple[str, dict]]:
    """Load per-stage summary JSONs.

    Returns:
        List of ``(stage_name, metrics_dict)`` tuples.
    """
    rows = []
    for stage_name, filename in STAGES:
        path = SUMMARY_DIR / filename
        with open(path) as f:
            data = json.load(f)
        rows.append((stage_name, data["metrics"]))
    return rows


def plot() -> None:
    """Build a colour-coded comparison table and save as PNG."""
    rows = _load_summaries()
    n_stages = len(rows)
    n_metrics = len(METRIC_SPEC)

    # Build value matrix (stages × metrics) for the heatmap.
    cell_text = []
    cell_vals = []  # numeric values for colour mapping.
    for stage_name, metrics in rows:
        row_text = []
        row_vals = []
        for display, key, fmt, higher_better in METRIC_SPEC:
            if key in metrics:
                mean = metrics[key]["mean"]
                std = metrics[key]["std"]
                if "%" in fmt:
                    row_text.append(f"{mean:{fmt}} +- {std:{fmt}}")
                else:
                    row_text.append(f"{mean:{fmt}} +- {std:{fmt}}")
                row_vals.append(mean)
            else:
                row_text.append("-")
                row_vals.append(np.nan)
        cell_text.append(row_text)
        cell_vals.append(row_vals)

    cell_vals = np.array(cell_vals, dtype=float)

    # Create table plot with coloured cells based on metric values.
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.set_axis_off()

    col_labels = [spec[0] for spec in METRIC_SPEC]
    row_labels = [r[0] for r in rows]

    # Normalise each column to [0, 1] for colour mapping
    normed = np.full_like(cell_vals, np.nan)
    for j in range(n_metrics):
        col = cell_vals[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        lo, hi = np.nanmin(col), np.nanmax(col)
        if hi - lo < 1e-9:
            normed[valid, j] = 0.5
        else:
            normed[valid, j] = (col[valid] - lo) / (hi - lo)
        # Invert if lower is better (SA score)
        if not METRIC_SPEC[j][3]:
            normed[valid, j] = 1.0 - normed[valid, j]

    # Colour map: white -> green for better.
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "wg", ["#ffffff", "#c7e9c0", "#41ab5d"]
    )

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)

    # Apply colours.
    for i in range(n_stages):
        for j in range(n_metrics):
            cell = table[i + 1, j]  # +1 because row 0 is header
            val = normed[i, j]
            if np.isnan(val):
                cell.set_facecolor("#f0f0f0")
                cell.get_text().set_color("#999999")
            else:
                cell.set_facecolor(cmap(val))

    # Style header and row labels.
    for j in range(n_metrics):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].get_text().set_color("white")
        table[0, j].get_text().set_fontweight("bold")
    for i in range(n_stages):
        table[i + 1, -1].set_facecolor("#f7f7f7")
        table[i + 1, -1].get_text().set_fontweight("bold")

    ax.set_title(
        "Generator Evaluation Across Pipeline Stages  (mean +- std, 3 seeds)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    out_path = OUT / "generator_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot()
