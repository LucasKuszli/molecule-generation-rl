"""Cross-pipeline comparison report for the academic writeup.

Reads all summary JSONs and produces:
  - Grouped bar charts comparing generation quality across stages
  - HIV activity comparison (bar + violin)
  - GNN performance summary table
  - Aggregated LaTeX-ready table (printed to stdout)

Usage:
    uv run python -m src.evaluation.report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config.data import root_dir

logger = logging.getLogger(__name__)

SUMMARY_DIR = root_dir / "evaluation" / "summary"
OUT_DIR = root_dir / "evaluation" / "report"
OUT_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
    }
)

STAGE_LABELS = {
    "pretrain": "Pretrain",
    "transfer": "Transfer",
    "pretrain_rl": "PT + RL",
    "transfer_rl": "TL + RL",
}

STAGE_COLORS = {
    "pretrain": "#4C72B0",
    "transfer": "#DD8452",
    "pretrain_rl": "#55A868",
    "transfer_rl": "#C44E52",
}

STAGE_ORDER = ["pretrain", "transfer", "pretrain_rl", "transfer_rl"]


def _load_summaries() -> dict:
    """Load all summary JSONs into a dict keyed by stage name.

    Returns:
        dict mapping stage names to their parsed summary data.
    """
    summaries = {}
    for stage in ["gnn", "pretrain", "transfer", "pretrain_rl", "transfer_rl"]:
        path = SUMMARY_DIR / f"{stage}_summary.json"
        if path.exists():
            with open(path) as f:
                summaries[stage] = json.load(f)
    return summaries


def fig_generation_quality(summaries: dict) -> Path:
    """Grouped bar chart of validity, uniqueness, novelty, diversity.

    Args:
        summaries(dict): Stage-keyed summary data.

    Returns:
        Path to the saved PNG.
    """
    metrics = ["validity", "uniqueness", "novelty", "diversity"]
    metric_labels = ["Validity", "Uniqueness", "Novelty", "IntDiv"]
    stages = [s for s in STAGE_ORDER if s in summaries]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.8 / len(stages)

    for i, stage in enumerate(stages):
        means = [summaries[stage]["metrics"].get(m, {}).get("mean", 0) for m in metrics]
        stds = [summaries[stage]["metrics"].get(m, {}).get("std", 0) for m in metrics]
        offset = (i - len(stages) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=STAGE_LABELS[stage],
            color=STAGE_COLORS[stage],
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )
        # Value labels
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{mean:.1%}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Generation Quality Comparison")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = OUT_DIR / "generation_quality.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def fig_hiv_activity(summaries: dict) -> Path:
    """Bar charts of mean P(active) and active fraction across stages.

    Args:
        summaries(dict): Stage-keyed summary data.

    Returns:
        Path to the saved PNG.
    """
    stages = [s for s in STAGE_ORDER if s in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mean P(active)
    ax = axes[0]
    means = [
        summaries[s]["metrics"].get("hiv_mean_prob", {}).get("mean", 0) for s in stages
    ]
    stds = [
        summaries[s]["metrics"].get("hiv_mean_prob", {}).get("std", 0) for s in stages
    ]
    colors = [STAGE_COLORS[s] for s in stages]
    labels = [STAGE_LABELS[s] for s in stages]

    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
        width=0.6,
    )
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Mean P(active)")
    ax.set_title("GNN-Predicted HIV Activity")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Active fraction (P>0.5).
    ax = axes[1]
    means = [
        summaries[s]["metrics"].get("hiv_active_frac", {}).get("mean", 0)
        for s in stages
    ]
    stds = [
        summaries[s]["metrics"].get("hiv_active_frac", {}).get("std", 0) for s in stages
    ]

    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
        width=0.6,
    )
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{m:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Fraction P(active) > 0.5")
    ax.set_title("Predicted Active Fraction")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "hiv_activity_comparison.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def fig_sa_score(summaries: dict) -> Path:
    """Bar chart of mean SA score across stages.

    Lower SA score indicates easier synthesis.

    Args:
        summaries(dict): Stage-keyed summary data.

    Returns:
        Path to the saved PNG.
    """
    stages = [s for s in STAGE_ORDER if s in summaries]
    labels = [STAGE_LABELS[s] for s in stages]
    means = [
        summaries[s]["metrics"].get("sa_score_mean", {}).get("mean", 0) for s in stages
    ]
    stds = [
        summaries[s]["metrics"].get("sa_score_mean", {}).get("std", 0) for s in stages
    ]
    colors = [STAGE_COLORS[s] for s in stages]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
        width=0.5,
    )
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Mean SA Score")
    ax.set_title("Synthetic Accessibility (lower = easier)")
    ax.set_ylim(0, 7)
    ax.axhline(
        y=6.0, color="gray", linestyle="--", alpha=0.5, label="Hard to synthesize"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUT_DIR / "sa_score_comparison.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def fig_tradeoff_radar(summaries: dict) -> Path:
    """Radar plot showing multi-objective trade-offs per stage.

    Args:
        summaries(dict): Stage-keyed summary data.

    Returns:
        Path to the saved PNG.
    """
    metrics = ["validity", "uniqueness", "diversity", "hiv_active_frac"]
    metric_labels = ["Validity", "Uniqueness", "IntDiv", "Active Frac."]
    stages = [s for s in STAGE_ORDER if s in summaries]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1.05)

    for stage in stages:
        values = [
            summaries[stage]["metrics"].get(m, {}).get("mean", 0) for m in metrics
        ]
        values += values[:1]
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=STAGE_LABELS[stage],
            color=STAGE_COLORS[stage],
            markersize=5,
        )
        ax.fill(angles, values, alpha=0.1, color=STAGE_COLORS[stage])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_title("Generation Trade-offs", fontsize=14, fontweight="bold", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()

    path = OUT_DIR / "tradeoff_radar.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def fig_gnn_performance(summaries: dict) -> Path | None:
    """Bar chart of GNN classification metrics.

    Args:
        summaries(dict): Stage-keyed summary data (must contain ``"gnn"``).

    Returns:
        Path to the saved PNG, or *None* if no GNN data is available.
    """
    if "gnn" not in summaries:
        return None
    gnn = summaries["gnn"]["metrics"]
    metrics = ["roc_auc", "pr_auc", "mcc", "balanced_acc", "f1"]
    labels = ["ROC-AUC", "PR-AUC", "MCC", "Balanced Acc.", "F1"]

    means = [gnn[m]["mean"] for m in metrics]
    stds = [gnn[m]["std"] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
        width=0.5,
    )
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Score")
    ax.set_title("GNN (GINE) - HIV Classification on Test Set")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = OUT_DIR / "gnn_performance.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def log_summary_table(summaries: dict) -> None:
    """Log a formatted comparison table of all generation stages.

    Args:
        summaries(dict): Stage-keyed summary data.
    """
    gen_stages = [s for s in STAGE_ORDER if s in summaries]
    gen_metrics = [
        ("validity", "Validity", True),
        ("uniqueness", "Uniqueness", True),
        ("novelty", "Novelty", True),
        ("diversity", "IntDiv", False),
        ("sa_score_mean", "SA Score", False),
        ("hiv_mean_prob", "Mean P(act)", False),
        ("hiv_active_frac", "Active %", True),
    ]

    width = 15 + 16 * len(gen_stages)
    header = f"{'Metric':<15}"
    for s in gen_stages:
        header += f" {STAGE_LABELS[s]:>15}"

    logger.info("=" * width)
    logger.info("GENERATION PIPELINE COMPARISON (mean +- std, n=3 seeds)")
    logger.info("=" * width)
    logger.info(header)
    logger.info("-" * width)

    for key, label, as_pct in gen_metrics:
        row = f"{label:<15}"
        for stage in gen_stages:
            m = summaries[stage]["metrics"].get(key, {})
            mean = m.get("mean", float("nan"))
            std = m.get("std", float("nan"))
            if as_pct:
                row += f" {mean * 100:>6.1f}+-{std * 100:4.1f}%  "
            else:
                row += f" {mean:>6.3f}+-{std:.3f}  "
        logger.info(row)

    logger.info("=" * width)

    if "gnn" in summaries:
        gnn = summaries["gnn"]["metrics"]
        logger.info("GNN CLASSIFICATION (GINE, HIV test set)")
        logger.info("-" * 40)
        for key, label in [
            ("roc_auc", "ROC-AUC"),
            ("pr_auc", "PR-AUC"),
            ("mcc", "MCC"),
            ("balanced_acc", "Balanced Acc."),
            ("f1", "F1"),
        ]:
            m = gnn[key]
            logger.info("  %-16s: %.3f +- %.3f", label, m["mean"], m["std"])
        logger.info("-" * 40)


def main() -> None:
    """Load summaries, generate comparison figures, and save report data."""
    logger.info("=" * 60)
    logger.info("CROSS-PIPELINE REPORT")
    logger.info("=" * 60)

    summaries = _load_summaries()
    if not summaries:
        logger.error("No summary JSONs found in %s", SUMMARY_DIR)
        return

    logger.info("Loaded summaries: %s", list(summaries.keys()))

    log_summary_table(summaries)

    logger.info("Generating figures …")
    paths = [
        fig_generation_quality(summaries),
        fig_hiv_activity(summaries),
        fig_sa_score(summaries),
        fig_tradeoff_radar(summaries),
    ]
    p = fig_gnn_performance(summaries)
    if p:
        paths.append(p)

    for p in paths:
        if p:
            logger.info("  → %s", p)

    # Save combined report data.
    report_data = {}
    for stage, data in summaries.items():
        report_data[stage] = {
            k: {"mean": v["mean"], "std": v["std"]}
            for k, v in data.get("metrics", {}).items()
        }

    report_path = OUT_DIR / "report_data.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Report data → %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
