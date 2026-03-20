"""Dataset description figures and statistics for the report.

Generates:
  - MOSES: molecular weight distribution, SMILES length distribution
  - HIV: class balance bar chart, MW distribution by class
  - Split size tables (printed + JSON)
  - Combined descriptor comparison (MOSES vs HIV actives vs HIV inactives)

Usage:
    uv run python -m src.evaluation.dataset
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from src.config.data import root_dir
from src.data.loader import (
    get_gnn_datasplit,
    get_pretrain_datasplit,
    get_transfer_dataset,
)

matplotlib.use("Agg")
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

OUT_DIR = root_dir / "evaluation" / "dataset"
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

COLORS = {
    "moses": "#4C72B0",
    "hiv_active": "#D62728",
    "hiv_inactive": "#7F7F7F",
}


def _compute_descriptors(smiles_list: list[str], max_n: int = 5000) -> pd.DataFrame:
    """Compute molecular descriptors for a list of SMILES strings.

    Randomly sub-samples to *max_n* entries when the input is larger.

    Args:
        smiles_list(list[str]): Input SMILES strings.
        max_n(int): Maximum number of molecules to process.

    Returns:
        pd.DataFrame with columns MW, LogP, TPSA, HBA, HBD, NumRings, SMILES_len.
    """
    rng = np.random.default_rng(42)
    if len(smiles_list) > max_n:
        idx = rng.choice(len(smiles_list), max_n, replace=False)
        smiles_list = [smiles_list[i] for i in idx]

    records = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        records.append(
            {
                "MW": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "HBA": Descriptors.NumHAcceptors(mol),
                "HBD": Descriptors.NumHDonors(mol),
                "NumRings": Descriptors.RingCount(mol),
                "SMILES_len": len(smi),
            }
        )
    return pd.DataFrame(records)


def fig_class_balance(hiv_df: pd.DataFrame) -> Path:
    """Bar chart of HIV class balance.

    Args:
        hiv_df(pd.DataFrame): Full HIV dataset with a ``label`` column.

    Returns:
        Path to the saved PNG.
    """
    counts = hiv_df["label"].value_counts().sort_index()
    labels = ["Inactive (0)", "Active (1)"]
    colors = [COLORS["hiv_inactive"], COLORS["hiv_active"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        labels, [counts[0], counts[1]], color=colors, edgecolor="black", width=0.5
    )
    for bar, c in zip(bars, [counts[0], counts[1]]):
        pct = c / len(hiv_df) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{c:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Number of molecules")
    ax.set_title("HIV Dataset - Class Distribution")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = OUT_DIR / "hiv_class_balance.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def fig_descriptor_distributions(
    moses_desc: pd.DataFrame,
    hiv_active_desc: pd.DataFrame,
    hiv_inactive_desc: pd.DataFrame,
) -> Path:
    """Overlapping histograms of MW, LogP, TPSA, and ring count.

    Args:
        moses_desc(pd.DataFrame): Descriptors for MOSES molecules.
        hiv_active_desc(pd.DataFrame): Descriptors for HIV-active molecules.
        hiv_inactive_desc(pd.DataFrame): Descriptors for HIV-inactive molecules.

    Returns:
        Path to the saved PNG.
    """
    props = [
        ("MW", "Molecular Weight (Da)", (0, 700)),
        ("LogP", "LogP", (-4, 10)),
        ("TPSA", "Topological Polar Surface Area", (0, 250)),
        ("NumRings", "Number of Rings", (0, 12)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (prop, xlabel, xlim) in zip(axes, props):
        for desc_df, label, color in [
            (moses_desc, "MOSES", COLORS["moses"]),
            (hiv_inactive_desc, "HIV Inactive", COLORS["hiv_inactive"]),
            (hiv_active_desc, "HIV Active", COLORS["hiv_active"]),
        ]:
            if prop in desc_df.columns:
                vals = desc_df[prop].dropna()
                ax.hist(
                    vals,
                    bins=50,
                    alpha=0.5,
                    label=label,
                    color=color,
                    density=True,
                    edgecolor="none",
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_xlim(xlim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Molecular Descriptor Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = OUT_DIR / "descriptor_distributions.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def fig_smiles_length(
    moses_desc: pd.DataFrame,
    hiv_active_desc: pd.DataFrame,
    hiv_inactive_desc: pd.DataFrame,
) -> Path:
    """SMILES string length distribution histogram.

    Args:
        moses_desc(pd.DataFrame): Descriptors for MOSES molecules.
        hiv_active_desc(pd.DataFrame): Descriptors for HIV-active molecules.
        hiv_inactive_desc(pd.DataFrame): Descriptors for HIV-inactive molecules.

    Returns:
        Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for desc_df, label, color in [
        (moses_desc, "MOSES", COLORS["moses"]),
        (hiv_inactive_desc, "HIV Inactive", COLORS["hiv_inactive"]),
        (hiv_active_desc, "HIV Active", COLORS["hiv_active"]),
    ]:
        vals = desc_df["SMILES_len"].dropna()
        ax.hist(
            vals,
            bins=60,
            alpha=0.5,
            label=label,
            color=color,
            density=True,
            edgecolor="none",
        )
    ax.set_xlabel("SMILES String Length")
    ax.set_ylabel("Density")
    ax.set_title("SMILES Length Distribution")
    ax.set_xlim(0, 200)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = OUT_DIR / "smiles_length_distribution.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def fig_split_sizes() -> Path:
    """Horizontal bar chart showing dataset split sizes.

    Returns:
        Path to the saved PNG.
    """
    pretrain = get_pretrain_datasplit()
    gnn = get_gnn_datasplit()
    transfer = get_transfer_dataset()

    data = [
        (
            "MOSES\n(Pretrain)",
            len(pretrain.train_set),
            len(pretrain.val_set),
            len(pretrain.test_set),
        ),
        ("HIV\n(GNN)", len(gnn.train_set), len(gnn.val_set), len(gnn.test_set)),
        ("HIV Actives\n(Transfer)", len(transfer.train_set), len(transfer.val_set), 0),
    ]

    names = [d[0] for d in data]
    trains = [d[1] for d in data]
    vals = [d[2] for d in data]
    tests = [d[3] for d in data]

    fig, ax = plt.subplots(figsize=(10, 4))
    y = range(len(names))
    ax.barh(y, trains, label="Train", color="#4C72B0", edgecolor="black")
    ax.barh(
        y, vals, left=trains, label="Validation", color="#55A868", edgecolor="black"
    )
    ax.barh(
        y,
        tests,
        left=[t + v for t, v in zip(trains, vals)],
        label="Test",
        color="#DD8452",
        edgecolor="black",
    )

    for i, (name, tr, va, te) in enumerate(data):
        total = tr + va + te
        ax.text(total + 200, i, f"Total: {total:,}", va="center", fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Number of Molecules")
    ax.set_title("Dataset Split Sizes")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = OUT_DIR / "split_sizes.png"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def main() -> None:
    """Load datasets, compute descriptors, generate figures, and save stats."""
    logger.info("=" * 60)
    logger.info("DATASET DESCRIPTION")
    logger.info("=" * 60)

    logger.info("Loading datasets …")
    pretrain = get_pretrain_datasplit()
    gnn = get_gnn_datasplit()
    transfer = get_transfer_dataset()

    hiv_full = pd.concat([gnn.train_set, gnn.val_set, gnn.test_set], ignore_index=True)
    hiv_actives = hiv_full[hiv_full["label"] == 1]
    hiv_inactives = hiv_full[hiv_full["label"] == 0]

    # Log split sizes.
    logger.info("MOSES (Pretraining):")
    logger.info("  Train: %8s", f"{len(pretrain.train_set):,}")
    logger.info("  Val:   %8s", f"{len(pretrain.val_set):,}")
    logger.info("  Test:  %8s", f"{len(pretrain.test_set):,}")
    logger.info(
        "  Total: %8s",
        f"{len(pretrain.train_set) + len(pretrain.val_set) + len(pretrain.test_set):,}",
    )

    logger.info("HIV (GNN Classification):")
    logger.info(
        "  Train: %8s  (actives: %s)",
        f"{len(gnn.train_set):,}",
        f"{(gnn.train_set['label'] == 1).sum():,}",
    )
    logger.info(
        "  Val:   %8s  (actives: %s)",
        f"{len(gnn.val_set):,}",
        f"{(gnn.val_set['label'] == 1).sum():,}",
    )
    logger.info(
        "  Test:  %8s  (actives: %s)",
        f"{len(gnn.test_set):,}",
        f"{(gnn.test_set['label'] == 1).sum():,}",
    )
    logger.info(
        "  Total: %8s  (actives: %s, ratio: 1:%d)",
        f"{len(hiv_full):,}",
        f"{len(hiv_actives):,}",
        len(hiv_inactives) // len(hiv_actives),
    )

    logger.info("HIV Actives (Transfer Learning):")
    logger.info("  Train: %8s", f"{len(transfer.train_set):,}")
    logger.info("  Val:   %8s", f"{len(transfer.val_set):,}")

    # Compute descriptors.
    logger.info("Computing molecular descriptors …")
    moses_desc = _compute_descriptors(pretrain.train_set["SMILES"].tolist())
    hiv_active_desc = _compute_descriptors(hiv_actives["SMILES"].tolist())
    hiv_inactive_desc = _compute_descriptors(hiv_inactives["SMILES"].tolist())

    for name, desc in [
        ("MOSES", moses_desc),
        ("HIV Active", hiv_active_desc),
        ("HIV Inactive", hiv_inactive_desc),
    ]:
        logger.info("%s descriptors (n=%d):", name, len(desc))
        for col in ["MW", "LogP", "TPSA", "NumRings", "SMILES_len"]:
            if col in desc.columns:
                logger.info(
                    "  %-12s: %.1f +- %.1f  (range: %.0f-%.0f)",
                    col,
                    desc[col].mean(),
                    desc[col].std(),
                    desc[col].min(),
                    desc[col].max(),
                )

    # Generate figures.
    logger.info("Generating figures …")
    paths = [
        fig_class_balance(hiv_full),
        fig_descriptor_distributions(moses_desc, hiv_active_desc, hiv_inactive_desc),
        fig_smiles_length(moses_desc, hiv_active_desc, hiv_inactive_desc),
        fig_split_sizes(),
    ]

    for p in paths:
        logger.info("  → %s", p)

    # Save stats JSON.
    stats = {
        "moses": {
            "train": len(pretrain.train_set),
            "val": len(pretrain.val_set),
            "test": len(pretrain.test_set),
            "total": len(pretrain.train_set)
            + len(pretrain.val_set)
            + len(pretrain.test_set),
            "descriptors": {
                col: {
                    "mean": round(moses_desc[col].mean(), 2),
                    "std": round(moses_desc[col].std(), 2),
                }
                for col in ["MW", "LogP", "TPSA", "NumRings", "SMILES_len"]
            },
        },
        "hiv": {
            "train": len(gnn.train_set),
            "val": len(gnn.val_set),
            "test": len(gnn.test_set),
            "total": len(hiv_full),
            "n_actives": int(len(hiv_actives)),
            "n_inactives": int(len(hiv_inactives)),
            "imbalance_ratio": round(len(hiv_inactives) / len(hiv_actives), 1),
        },
        "hiv_actives_transfer": {
            "train": len(transfer.train_set),
            "val": len(transfer.val_set),
        },
    }
    stats_path = OUT_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats JSON → %s", stats_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
