"""Evaluation of the transfer-learned SMILES LSTM on HIV actives.

Samples molecules, computes validity / uniqueness / novelty / diversity / SA,
scores HIV activity via the GNN predictor, and plots NLL training curves.

Usage::

    uv run python -m src.transfer.evaluation
    uv run python -m src.transfer.evaluation --seed 123
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Draw

from src.config.data import root_dir
from src.config.training import TransferLearningConfig
from src.data.loader import get_pretrain_datasplit, get_transfer_dataset
from src.pretrain.evaluation import (
    evaluate_generated_molecules,
    generate_smiles,
    gnn_checkpoint_for_seed,
    load_hiv_predictor,
    load_model_from_checkpoint,
    predict_hiv_activity,
)
from src.pretrain.model import SmilesLSTM
from src.pretrain.vocabulary import SMILESTokenizer, Vocabulary

matplotlib.use("Agg")
RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


def run_evaluation(
    checkpoint_path: Optional[Path] = None,
    n_samples: int = 5000,
    device: str = "cpu",
    seed: int = 42,
) -> Dict:
    """Full evaluation pipeline for the transfer-learned LSTM.

    Loads a checkpoint, samples SMILES, computes molecular quality metrics,
    runs the GNN HIV-activity predictor, renders a molecule grid, and
    plots training curves from saved metadata.

    Args:
        checkpoint_path(Path | None): Explicit path to a ``best_transfer.pt``
            file.  When *None* the default path from *config* is used.
        n_samples(int): Number of SMILES strings to sample.
        device(str): Torch device string (``"cpu"`` or ``"cuda"``).
        seed(int): Random seed that selects the checkpoint directory.

    Returns:
        dict: Evaluation metrics including validity, uniqueness, novelty,
        diversity, SA score statistics, and optional HIV activity scores.
    """

    config = TransferLearningConfig(seed=seed)
    if checkpoint_path is None:
        checkpoint_path = Path(config.checkpoint_outdir) / "best_transfer.pt"

    logger.info("Checkpoint: %s", checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Run transfer learning first: "
            "uv run python -m src.transfer.train"
        )

    # Load model and vocab.
    logger.info("Loading model …")
    model, vocab = load_model_from_checkpoint(checkpoint_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Vocab: %d tokens | Params: %s", len(vocab), f"{n_params:,}")

    # Sample molecules.
    logger.info("Sampling %d molecules …", n_samples)
    generated = generate_smiles(model, vocab, n=n_samples, device=device)

    # Training sets for novelty check.
    logger.info("Loading training sets for novelty check …")
    pretrain_data = get_pretrain_datasplit()
    pretrain_smiles = set(pretrain_data.train_set["SMILES"].tolist())

    transfer_data = get_transfer_dataset()
    transfer_smiles = set(transfer_data.train_set["SMILES"].tolist())

    # Novelty = not in either training set.
    training_set = pretrain_smiles | transfer_smiles

    # Evaluate validity, uniqueness, novelty, diversity, and SA score.
    logger.info("Evaluating …")
    results = evaluate_generated_molecules(generated, training_set)

    logger.info("=" * 60)
    logger.info("TRANSFER-LEARNED LSTM EVALUATION")
    logger.info("=" * 60)
    logger.info("Generated:     %s", f"{results['n_generated']:,}")
    logger.info(
        "Valid:          %s (%s)",
        f"{results['n_valid']:,}",
        f"{results['validity']:.1%}",
    )
    logger.info(
        "Unique:         %s (%s)",
        f"{results['n_unique']:,}",
        f"{results['uniqueness']:.1%}",
    )
    logger.info(
        "Novel:          %s (%s)",
        f"{results['n_novel']:,}",
        f"{results['novelty']:.1%}",
    )
    logger.info("Diversity:      %.4f", results["diversity"])
    if results["sa_score_mean"] is not None:
        logger.info(
            "SA Score:       %.2f +- %.2f",
            results["sa_score_mean"],
            results["sa_score_std"],
        )
    logger.info("=" * 60)

    # HIV activity prediction (GNN classifier).
    logger.info("HIV activity prediction (GNN classifier) …")
    gnn_ckpt = gnn_checkpoint_for_seed(seed)
    predictor = load_hiv_predictor(gnn_ckpt, device=device)
    if predictor is not None:
        valid_smiles = results["valid_smiles"][:1000]
        hiv_results = predict_hiv_activity(predictor, valid_smiles)
        if hiv_results:
            probs = [r["prob"] for r in hiv_results]
            labels = [r["label"] for r in hiv_results]
            active_frac = sum(labels) / len(labels)
            results["hiv_mean_prob"] = float(np.mean(probs))
            results["hiv_active_frac"] = active_frac

            logger.info("Scored:         %d molecules", len(hiv_results))
            logger.info("Mean P(active): %.4f", results["hiv_mean_prob"])
            logger.info(
                "Active (>%.2f): %d (%.1f%%)",
                predictor.threshold,
                sum(labels),
                active_frac * 100,
            )
        else:
            logger.warning("No valid molecules could be scored.")
    else:
        logger.warning("GNN predictor not available - skipping HIV activity scoring.")

    # Visualize top-20 highest predicted activity molecules.
    img_dir = config.img_outdir
    img_dir.mkdir(parents=True, exist_ok=True)

    if predictor is not None and hiv_results:
        sorted_hits = sorted(hiv_results, key=lambda r: r["prob"], reverse=True)[:20]
        top_mols = [Chem.MolFromSmiles(r["smiles"]) for r in sorted_hits]
        top_mols = [m for m, r in zip(top_mols, sorted_hits) if m is not None]
        sorted_hits = [
            r
            for m, r in zip(
                [Chem.MolFromSmiles(r["smiles"]) for r in sorted_hits], sorted_hits
            )
            if m is not None
        ]
        legends = [f"P(active)={r['prob']:.3f}" for r in sorted_hits]

        fig, ax = plt.subplots(figsize=(14, 12))
        grid_img = Draw.MolsToGridImage(
            top_mols,
            molsPerRow=5,
            subImgSize=(300, 250),
            legends=legends,
        )
        ax.imshow(grid_img)
        ax.axis("off")
        ax.set_title(
            f"Top {len(top_mols)} Generated Molecules by Predicted HIV Activity\n"
            f"Activity threshold = {predictor.threshold:.2f}",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        img_path = img_dir / "transfer_sample_mols.png"
        fig.savefig(str(img_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Molecule grid saved to %s", img_path)
    elif generated and results.get("valid_mols"):
        sample_mols = results["valid_mols"][:20]
        img = Draw.MolsToGridImage(
            sample_mols,
            molsPerRow=5,
            subImgSize=(250, 200),
            legends=[f"Mol {i + 1}" for i in range(len(sample_mols))],
        )
        img_path = img_dir / "transfer_sample_mols.png"
        img.save(str(img_path))
        logger.info("Molecule grid saved to %s", img_path)
    else:
        logger.warning("No valid molecules to display.")

    # Save results to JSON (excluding non-serializable items like molecule objects).
    serializable = {
        k: v
        for k, v in results.items()
        if k
        not in (
            "valid_mols",
            "valid_smiles",
            "unique_smiles",
            "novel_smiles",
            "sa_scores",
        )
    }
    eval_dir = config.eval_outdir
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / "transfer_eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Evaluation results saved to %s", eval_path)

    # Plot NLL training curves from metadata, if available.
    meta_path = Path(config.checkpoint_outdir) / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        history = meta.get("history", [])
        if history:
            epochs = [h["epoch"] for h in history]
            train_nlls = [h["train_nll"] for h in history]
            val_nlls = [h["val_nll"] for h in history]
            validity = [h["validity"] for h in history]
            uniqueness = [h["uniqueness"] for h in history]

            img_dir = config.img_outdir
            img_dir.mkdir(parents=True, exist_ok=True)

            # NLL curves.
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, train_nlls, label="Train NLL", linewidth=2)
            ax.plot(epochs, val_nlls, label="Val NLL", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("NLL")
            ax.set_title("Transfer Learning NLL")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            nll_path = img_dir / "transfer_nll_curves.png"
            fig.savefig(str(nll_path), dpi=150)
            plt.close(fig)
            logger.info("NLL curves saved to %s", nll_path)

            # Validity and uniqueness curves.
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, validity, label="Validity", linewidth=2)
            ax.plot(epochs, uniqueness, label="Uniqueness", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Fraction")
            ax.set_ylim(0, 1.05)
            ax.set_title("Transfer Learning Sample Quality")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            quality_path = img_dir / "transfer_quality_curves.png"
            fig.savefig(str(quality_path), dpi=150)
            plt.close(fig)
            logger.info("Quality curves saved to %s", quality_path)
    else:
        logger.warning(
            "No training metadata found at %s - skipping NLL plots.", meta_path
        )

    return results


# Run the evaluation function if this script is executed directly.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate transfer-learned SMILES LSTM"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed (selects checkpoint dir)"
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    run_evaluation(
        checkpoint_path=ckpt,
        n_samples=args.n_samples,
        device=args.device,
        seed=args.seed,
    )
