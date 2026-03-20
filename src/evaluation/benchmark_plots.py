"""Benchmark bar chart comparing GNN vs Random Forest on HIV classification.

Reads ``benchmark/summary.csv`` and produces a grouped bar chart of PR-AUC,
ROC-AUC, and balanced accuracy for each model.

Usage::

    uv run python -m src.evaluation.benchmark_plots
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.container import BarContainer

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmark"
IMG_DIR = RESULTS_DIR / "plots"

METRIC_COLUMNS: dict[str, tuple[str, str]] = {
    "pr_auc": ("gnn_pr_auc", "rf_pr_auc"),
    "roc_auc": ("gnn_roc_auc", "rf_roc_auc"),
    "balanced_acc": ("gnn_balanced_acc", "rf_balanced_acc"),
}


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model mean and std for each metric.

    Args:
        df(pd.DataFrame): Benchmark summary with per-seed rows.

    Returns:
        pd.DataFrame with columns [metric, model, mean, std].
    """
    rows = []
    for metric_name, (gnn_col, rf_col) in METRIC_COLUMNS.items():
        rows.append(
            {
                "metric": metric_name,
                "model": "GNN",
                "mean": df[gnn_col].mean(),
                "std": df[gnn_col].std(),
            }
        )
        rows.append(
            {
                "metric": metric_name,
                "model": "RF",
                "mean": df[rf_col].mean(),
                "std": df[rf_col].std(),
            }
        )
    return pd.DataFrame(rows)


def plot_benchmark(summary: pd.DataFrame) -> Path:
    """Create a grouped bar chart and save to disk.

    Args:
        summary(pd.DataFrame): Output of :func:`build_summary`.

    Returns:
        Path to the saved figure.
    """
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    pivot_mean = summary.pivot(index="metric", columns="model", values="mean")
    pivot_std = summary.pivot(index="metric", columns="model", values="std")

    fig, ax = plt.subplots()
    pivot_mean.plot(kind="bar", yerr=pivot_std, capsize=4, ax=ax)

    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.3f", padding=3)

    ax.set_title("Benchmark Metrics (GNN vs RF, mean +- std)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    out_path = IMG_DIR / "benchmark_metrics_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    """Load benchmark data, print a summary, and generate the bar chart."""
    df = pd.read_csv(RESULTS_DIR / "summary.csv")
    logger.info("Loaded columns: %s", df.columns.tolist())

    summary = build_summary(df)
    summary["var"] = summary["std"] ** 2

    logger.info("=== METRIC SUMMARY (mean +- var) ===")
    for _, row in summary.iterrows():
        logger.info(
            "%s | %s -> mean: %.4f, var: %.4f",
            row["model"],
            row["metric"],
            row["mean"],
            row["var"],
        )

    out_path = plot_benchmark(summary)
    logger.info("Saved -> %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
