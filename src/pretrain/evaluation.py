"""Evaluation of the pre-trained SMILES LSTM generator.

Samples molecules from a checkpoint, then computes:

* Validity, uniqueness, novelty (vs. MOSES training set)
* Internal diversity (Tanimoto distance on Morgan fingerprints)
* SA Score distribution
* HIV activity prediction (via GNN classifier, if available)

Usage::

    uv run python -m src.pretrain.evaluation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, Draw, rdMolDescriptors

from src.config.data import root_dir
from src.config.training import PretrainingConfig
from src.data.loader import get_pretrain_datasplit
from src.pretrain.model import SmilesLSTM
from src.pretrain.vocabulary import SMILESTokenizer, Vocabulary

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


_LEGACY_GNN_CHECKPOINT = root_dir / "src" / "gnn" / "checkpoints" / "gnn_model.pt"


def gnn_checkpoint_for_seed(seed: int) -> Path:
    """Return the GNN checkpoint path matching the given seed."""
    return root_dir / "artifacts" / "gnn" / f"seed_{seed}" / "best_gnn.pt"


# Keep a module-level default for backwards compatibility
GNN_CHECKPOINT = gnn_checkpoint_for_seed(42)


def calc_sa_score(mol) -> float:
    """Approximate synthetic-accessibility score (1 = easy, 10 = hard).

    Attempts to use the `sascorer` package; falls back to a heuristic based
    on ring count, stereocentres and molecular weight.

    Args:
        mol: RDKit Mol object.

    Returns:
        SA score clamped to [1, 10].
    """
    try:
        from sascorer import calculateScore

        return calculateScore(mol)
    except ImportError:
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        mw = Descriptors.MolWt(mol)
        score = 1 + 0.5 * num_rings + num_stereo + 0.005 * mw
        return min(10.0, max(1.0, score))


# GNN based HIV activity scoring.
def load_hiv_predictor(checkpoint_path: Path, device: str = "cpu"):
    """Load the GNN-based HIV activity predictor.

    Falls back to a legacy checkpoint location when the primary path is
    missing.  Returns *None* when no usable checkpoint can be found.

    Args:
        checkpoint_path(Path): primary checkpoint path.
        device(str): torch device string.

    Returns:
        HIVActivityPredictor or *None*.
    """
    # Fall back to legacy checkpoint location if the new one doesn't exist
    if not checkpoint_path.exists() and _LEGACY_GNN_CHECKPOINT.exists():
        checkpoint_path = _LEGACY_GNN_CHECKPOINT
    if not checkpoint_path.exists():
        logger.warning("GNN checkpoint not found at %s", checkpoint_path)
        return None
    try:
        from src.gnn.predictor import HIVActivityPredictor

        predictor = HIVActivityPredictor(str(checkpoint_path), device=device)
        return predictor
    except Exception as e:
        logger.warning("Could not load GNN predictor: %s", e)
        return None


def predict_hiv_activity(predictor, smiles_list: List[str]) -> List[dict]:
    """Score SMILES with the GNN predictor.

    Args:
        predictor: HIVActivityPredictor instance.
        smiles_list(List[str]): SMILES to score.

    Returns:
        List of dicts with keys *smiles*, *label*, *prob*.
    """
    results = []
    for smi in smiles_list:
        label, prob = predictor.predict(smi)
        if label is not None:
            results.append({"smiles": smi, "label": label, "prob": prob})
    return results


# Main evauluation function computes all metrics on the generated molecules.
def evaluate_generated_molecules(
    generated: List[str],
    training_set: Set[str],
    n_diversity_sample: int = 1000,
) -> Dict:
    """Compute validity, uniqueness, novelty, diversity and SA scores.

    Args:
        generated(List[str]): raw generated SMILES.
        training_set(Set[str]): canonical SMILES from the training corpus.
        n_diversity_sample(int): cap for the diversity sample.

    Returns:
        Dict of evaluation metrics and intermediate lists.
    """

    results: Dict = {
        "n_generated": len(generated),
        "valid_smiles": [],
        "valid_mols": [],
    }

    # Validity
    for smi in generated:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol)
            results["valid_smiles"].append(canonical)
            results["valid_mols"].append(mol)

    n_valid = len(results["valid_smiles"])
    results["n_valid"] = n_valid
    results["validity"] = n_valid / len(generated) if generated else 0

    # Uniqueness
    unique_smiles = list(set(results["valid_smiles"]))
    n_unique = len(unique_smiles)
    results["n_unique"] = n_unique
    results["uniqueness"] = n_unique / n_valid if n_valid > 0 else 0
    results["unique_smiles"] = unique_smiles

    # Novelty (vs. training set)
    novel_smiles = [s for s in unique_smiles if s not in training_set]
    results["n_novel"] = len(novel_smiles)
    results["novelty"] = len(novel_smiles) / n_unique if n_unique > 0 else 0
    results["novel_smiles"] = novel_smiles

    # Internal diversity (Tanimoto distance, Morgan2 2048-bit)
    if len(unique_smiles) >= 2:
        sample = unique_smiles[:n_diversity_sample]
        mols = [Chem.MolFromSmiles(s) for s in sample]
        mols = [m for m in mols if m is not None]
        if len(mols) >= 2:
            fps = [
                AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols
            ]
            n_fp = min(500, len(fps))
            distances = []
            for i in range(n_fp):
                for j in range(i + 1, n_fp):
                    distances.append(1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            results["diversity"] = float(np.mean(distances)) if distances else 0.0
        else:
            results["diversity"] = 0.0
    else:
        results["diversity"] = 0.0

    # SA Score
    sa_scores = []
    for mol in results["valid_mols"][:500]:
        try:
            sa_scores.append(calc_sa_score(mol))
        except Exception:
            pass
    results["sa_score_mean"] = float(np.mean(sa_scores)) if sa_scores else None
    results["sa_score_std"] = float(np.std(sa_scores)) if sa_scores else None
    results["sa_scores"] = sa_scores

    return results


def load_model_from_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    """Load a SmilesLSTM and Vocabulary from a training checkpoint.

    Args:
        checkpoint_path(Path): path to the ``.pt`` file.
        device(str): torch device string.

    Returns:
        Tuple of (SmilesLSTM, Vocabulary).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = Vocabulary(ckpt["vocab"])
    model = SmilesLSTM(**ckpt["model_params"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, vocab


def generate_smiles(
    model: SmilesLSTM,
    vocab: Vocabulary,
    n: int = 5000,
    batch_size: int = 512,
    device: str = "cpu",
) -> List[str]:
    """Sample SMILES from the model in batches.

    Args:
        model(SmilesLSTM): trained language model.
        vocab(Vocabulary): token-to-id mapping.
        n(int): total number of molecules to generate.
        batch_size(int): molecules per sampling call.
        device(str): torch device string.

    Returns:
        List of generated SMILES strings.
    """
    tokenizer = SMILESTokenizer()
    all_smiles: List[str] = []
    device_t = torch.device(device)
    remaining = n
    while remaining > 0:
        batch_n = min(batch_size, remaining)
        smiles, _ = model.sample(vocab, tokenizer, n=batch_n, device=device_t)
        all_smiles.extend(smiles)
        remaining -= batch_n
    return all_smiles


# Main evaluation function computes all metrics on the generated molecules.
def run_evaluation(
    checkpoint_path: Optional[Path] = None,
    n_samples: int = 5000,
    device: str = "cpu",
    seed: int = 42,
) -> Dict:
    """Full evaluation pipeline for the pre-trained LSTM generator.

    Args:
        checkpoint_path(Path | None): model checkpoint; inferred from seed when *None*.
        n_samples(int): number of molecules to sample.
        device(str): torch device string.
        seed(int): random seed (selects checkpoint directory).

    Returns:
        Dict of evaluation metrics.
    """

    config = PretrainingConfig(seed=seed)
    if checkpoint_path is None:
        checkpoint_path = Path(config.checkpoint_outdir) / "best_prior.pt"

    logger.info("Checkpoint: %s", checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train first with: "
            "uv run python -m src.pretrain.train"
        )
    # Load model and vocab from checkpoint.
    logger.info("Loading model …")
    model, vocab = load_model_from_checkpoint(checkpoint_path, device)
    logger.info(
        "  Vocab: %d tokens  |  Params: %s",
        len(vocab),
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # Generate molecules.
    logger.info("Sampling %d molecules …", n_samples)
    generated = generate_smiles(model, vocab, n=n_samples, device=device)

    # Training set for novelty check.
    logger.info("Loading training set for novelty check …")
    data = get_pretrain_datasplit()
    training_set = set(data.train_set["SMILES"].tolist())

    # Evaluate generated molecules.
    logger.info("Evaluating …")
    results = evaluate_generated_molecules(generated, training_set)

    logger.info("=" * 60)
    logger.info("PRE-TRAINED LSTM EVALUATION")
    logger.info("=" * 60)
    logger.info("  Generated:     %s", f"{results['n_generated']:,}")
    logger.info(
        "  Valid:          %s (%.1f%%)",
        f"{results['n_valid']:,}",
        results["validity"] * 100,
    )
    logger.info(
        "  Unique:         %s (%.1f%%)",
        f"{results['n_unique']:,}",
        results["uniqueness"] * 100,
    )
    logger.info(
        "  Novel:          %s (%.1f%%)",
        f"{results['n_novel']:,}",
        results["novelty"] * 100,
    )
    logger.info("  Diversity:      %.4f", results["diversity"])
    if results["sa_score_mean"] is not None:
        logger.info(
            "  SA Score:       %.2f +- %.2f",
            results["sa_score_mean"],
            results["sa_score_std"],
        )
    logger.info("=" * 60)

    # HIV activity prediction (GNN classifier)
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

            logger.info("  Scored:         %d molecules", len(hiv_results))
            logger.info("  Mean P(active): %.4f", results["hiv_mean_prob"])
            logger.info(
                "  Active (>%.2f): %d (%.1f%%)",
                predictor.threshold,
                sum(labels),
                active_frac * 100,
            )
        else:
            logger.warning("  No valid molecules could be scored.")
    else:
        logger.warning("  GNN predictor not available - skipping HIV activity scoring.")
        logger.warning("  Train a GNN classifier first, or place a checkpoint at:")
        logger.warning("    %s", gnn_ckpt)

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
        img_path = img_dir / "pretrain_sample_mols.png"
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
        img_path = img_dir / "pretrain_sample_mols.png"
        img.save(str(img_path))
        logger.info("Molecule grid saved to %s", img_path)
    else:
        logger.warning("No valid molecules to display.")

    # Save results to JSON.
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
    eval_path = eval_dir / "pretrain_eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Evaluation results saved to %s", eval_path)

    # Visualize training curves if metadata is available.
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

            # -- NLL curves --
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, train_nlls, label="Train NLL", linewidth=2)
            ax.plot(epochs, val_nlls, label="Val NLL", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("NLL")
            ax.set_title("Pre-training NLL")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            nll_path = img_dir / "pretrain_nll_curves.png"
            fig.savefig(str(nll_path), dpi=150)
            plt.close(fig)
            logger.info("NLL curves saved to %s", nll_path)

            # -- Validity & uniqueness curves --
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, validity, label="Validity", linewidth=2)
            ax.plot(epochs, uniqueness, label="Uniqueness", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Fraction")
            ax.set_ylim(0, 1.05)
            ax.set_title("Pre-training Sample Quality")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            quality_path = img_dir / "pretrain_quality_curves.png"
            fig.savefig(str(quality_path), dpi=150)
            plt.close(fig)
            logger.info("Quality curves saved to %s", quality_path)
    else:
        logger.info("No training metadata found at %s - skipping NLL plots.", meta_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate pre-trained SMILES LSTM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: best_prior.pt)",
    )
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
