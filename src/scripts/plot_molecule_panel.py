#!/usr/bin/env python
"""Molecule panel: shortest / median / longest representative structures.

For each source (ground-truth HIV actives + 4 generator stages), samples
or selects molecules and picks the shortest, median-length, and longest
SMILES.  Ground-truth actives are taken directly from the dataset (no
prediction); generated molecules are filtered to FP-evaluator-predicted active
(independent Morgan-FP + logistic regression classifier, threshold = 0.5).

Outputs:
    img/molecule_panel.png

Usage::

    uv run python -m src.scripts.plot_molecule_panel
    uv run python -m src.scripts.plot_molecule_panel --n-sample 1000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw

from src.gnn.fingerprint_evaluator import FingerprintEvaluator
from src.pretrain.model import SmilesLSTM
from src.pretrain.vocabulary import SMILESTokenizer, Vocabulary

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "img"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 10,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


GENERATOR_STAGES = {
    "Pretrain": ROOT / "artifacts" / "pretrain" / "seed_42" / "best_prior.pt",
    "Transfer": ROOT
    / "artifacts"
    / "transfer_learning"
    / "seed_42"
    / "best_transfer.pt",
    "Pretrain-RL": ROOT
    / "artifacts"
    / "pretrain_rl"
    / "seed_42"
    / "best_pretrain_rl.pt",
    "Transfer-RL": ROOT
    / "artifacts"
    / "transfer_rl"
    / "seed_42"
    / "best_transfer_rl.pt",
}

FP_EVALUATOR_PKL = ROOT / "artifacts" / "gnn" / "seed_42" / "fp_evaluator.pkl"
HIV_CSV = ROOT / "data" / "training_data" / "hiv.csv"

BINS = ["shortest", "median", "longest"]


def _pick_three(smiles_list: list[str]) -> dict[str, str]:
    """Return the shortest, median, and longest entry from a sorted list.

    Args:
        smiles_list(list[str]): SMILES strings sorted by length.

    Returns:
        dict mapping ``"shortest"``, ``"median"``, ``"longest"`` to a SMILES.
    """
    if len(smiles_list) < 3:
        # pad with what we have
        mapping = {}
        for i, tag in enumerate(BINS):
            mapping[tag] = smiles_list[min(i, len(smiles_list) - 1)]
        return mapping
    return {
        "shortest": smiles_list[0],
        "median": smiles_list[len(smiles_list) // 2],
        "longest": smiles_list[-1],
    }


def _mol_legend(smi: str, prob: float | None = None) -> str:
    """Build a legend string with MW and optionally P(active).

    Args:
        smi(str): SMILES string.
        prob(float | None): Predicted probability of HIV activity.

    Returns:
        Formatted legend string.
    """
    mol = Chem.MolFromSmiles(smi)
    mw = round(Descriptors.MolWt(mol), 0) if mol else 0
    parts = [f"MW={mw:.0f}"]
    if prob is not None:
        parts.append(f"P={prob:.2f}")
    parts.append(f"len={len(smi)}")
    return "  ".join(parts)


def gather_gt_actives() -> dict[str, tuple[str, None]]:
    """Pick shortest / median / 10th-longest from confirmed HIV actives.

    Returns:
        dict mapping bin names to ``(smiles, None)`` tuples.
    """
    df = pd.read_csv(HIV_CSV)
    actives = df[df["label"] == 1]["SMILES"].tolist()
    valid = sorted(
        [s for s in actives if Chem.MolFromSmiles(s) is not None],
        key=len,
    )
    picks = {
        "shortest": valid[0],
        "median": valid[len(valid) // 2],
        "longest": valid[-10],  # 10th longest - avoids the huge outliers
    }
    return {k: (v, None) for k, v in picks.items()}


def gather_generated(
    stage_name: str,
    ckpt_path: Path,
    predictor: FingerprintEvaluator,
    n_sample: int = 500,
) -> dict[str, tuple[str, float]]:
    """Sample molecules, filter to FP-predicted active, pick 3 representatives.

    Args:
        stage_name(str): Display name of the pipeline stage.
        ckpt_path(Path): Path to the model checkpoint.
        predictor(FingerprintEvaluator): Trained FP evaluator instance.
        n_sample(int): Number of SMILES to sample.

    Returns:
        dict mapping bin names to ``(smiles, probability)`` tuples.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab = Vocabulary(token_to_id=ckpt["vocab"])
    tokenizer = SMILESTokenizer()
    model = SmilesLSTM(**ckpt["model_params"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    smiles_all, _ = model.sample(
        vocab, tokenizer, n=n_sample, temperature=1.0, device=torch.device("cpu")
    )

    active: list[tuple[str, float]] = []
    for smi in smiles_all:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        label, prob = predictor.predict(smi)
        if label == 1:
            active.append((smi, prob))

    active.sort(key=lambda x: len(x[0]))
    logger.info("%s: %d/%d predicted-active", stage_name, len(active), n_sample)

    if not active:
        return {k: ("C", 0.0) for k in BINS}  # fallback dummy

    smiles_only = [s for s, _ in active]
    prob_map = {s: p for s, p in active}
    picks = _pick_three(smiles_only)
    return {k: (v, prob_map[v]) for k, v in picks.items()}


def plot(n_sample: int = 500) -> None:
    """Build and save the molecule panel figure.

    Args:
        n_sample(int): Number of SMILES to sample per generator stage.
    """
    predictor = FingerprintEvaluator.load(FP_EVALUATOR_PKL)

    sources: list[tuple[str, dict[str, tuple[str, float | None]]]] = []
    logger.info("Gathering ground-truth actives …")
    sources.append(("GT Active", gather_gt_actives()))

    for stage_name, ckpt_path in GENERATOR_STAGES.items():
        logger.info("Sampling %s …", stage_name)
        sources.append(
            (stage_name, gather_generated(stage_name, ckpt_path, predictor, n_sample))
        )

    # Layout: columns = sources (5), rows = bins (shortest / median / longest)
    n_cols = len(sources)
    n_rows = len(BINS)
    SUB_W, SUB_H = 280, 220

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.8, n_rows * 2.6 + 0.8),
    )
    fig.subplots_adjust(
        hspace=0.05, wspace=0.04, top=0.86, bottom=0.02, left=0.01, right=0.99
    )

    for col_idx, (source_name, picks) in enumerate(sources):
        for row_idx, bin_name in enumerate(BINS):
            ax = axes[row_idx, col_idx]
            smi, prob = picks[bin_name]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mol = Chem.MolFromSmiles("C")

            # Draw molecule to PIL image.
            img = Draw.MolToImage(mol, size=(SUB_W, SUB_H))
            ax.imshow(img)
            ax.set_axis_off()

            # Annotation below molecule.
            mol_obj = Chem.MolFromSmiles(smi)
            mw = round(Descriptors.MolWt(mol_obj), 0) if mol_obj else 0
            label_parts = [f"MW={mw:.0f}", f"len={len(smi)}"]
            if prob is not None:
                label_parts.insert(0, f"P={prob:.2f}")
            ax.set_xlabel("  ".join(label_parts), fontsize=7, labelpad=1)
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            # Re-enable xlabel visibility.
            ax.xaxis.set_visible(True)

        # Column title = source name.
        axes[0, col_idx].set_title(source_name, fontsize=10, fontweight="bold", pad=6)

    # Row labels on the left.
    for row_idx, bin_name in enumerate(BINS):
        axes[row_idx, 0].set_ylabel(
            bin_name.capitalize(),
            fontsize=9,
            fontweight="bold",
            rotation=90,
            labelpad=8,
        )
        axes[row_idx, 0].yaxis.set_visible(True)

    fig.suptitle(
        "Molecule Panel - Shortest / Median / Longest SMILES predicted active by FP Evaluator",
        fontsize=11,
        fontweight="bold",
        y=0.95,
    )

    out_path = OUT / "molecule_panel.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate molecule panel plot")
    parser.add_argument(
        "--n-sample",
        type=int,
        default=500,
        help="Number of SMILES to sample per generator stage (default: 500)",
    )
    args = parser.parse_args()
    plot(n_sample=args.n_sample)
