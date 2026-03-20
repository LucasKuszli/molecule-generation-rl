"""Evaluation of REINFORCE-tuned SMILES LSTM.

Samples molecules, computes validity / uniqueness / novelty / diversity / SA,
scores HIV activity via GNN and an independent FP evaluator, assesses model
collapse indicators, and generates comprehensive training-curve and
distribution plots.

Usage::

    uv run python -m src.reinforce.evaluation
    uv run python -m src.reinforce.evaluation --stage transfer_rl --seed 67
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
from rdkit.Chem import Draw

from src.config.training import RLConfig
from src.data.loader import get_pretrain_datasplit, get_transfer_dataset
from src.data.preprocess import scaffold_smiles
from src.gnn.fingerprint_evaluator import load_fp_evaluator
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

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

# Collapse indicator thresholds (can be tuned based on validation experiments).
_THRESHOLDS = {
    "validity_low": 0.70,
    "uniqueness_low": 0.80,
    "diversity_low": 0.60,
    "nll_low": 20.0,
}


def assess_collapse(results: Dict, history: List[Dict]) -> Dict:
    """Evaluate collapse indicators from generation results and training history.

    Args:
        results(dict): Molecular quality metrics from
            :func:`evaluate_generated_molecules`.
        history(list[dict]): Per-epoch training records.

    Returns:
        dict: Collapse flags, warnings, and an overall ``health`` rating
        (``"HEALTHY"`` / ``"CAUTION"`` / ``"AT RISK"`` / ``"COLLAPSED"``).
    """
    flags: List[str] = []
    warnings: List[str] = []

    # Generation-time checks.
    validity = results.get("validity", 1.0)
    uniqueness = results.get("uniqueness", 1.0)
    diversity = results.get("diversity", 1.0)

    if validity < _THRESHOLDS["validity_low"]:
        flags.append(
            f"LOW VALIDITY: {validity:.1%} (threshold {_THRESHOLDS['validity_low']:.0%})"
        )
    elif validity < 0.85:
        warnings.append(f"Validity declining: {validity:.1%}")

    if uniqueness < _THRESHOLDS["uniqueness_low"]:
        flags.append(f"LOW UNIQUENESS: {uniqueness:.1%} - possible mode collapse")
    elif uniqueness < 0.90:
        warnings.append(f"Uniqueness below 90%: {uniqueness:.1%}")

    if diversity < _THRESHOLDS["diversity_low"]:
        flags.append(
            f"LOW DIVERSITY: {diversity:.4f} - model may be stuck in narrow mode"
        )
    elif diversity < 0.70:
        warnings.append(f"Diversity declining: {diversity:.4f}")

    # Training history checks.
    if history:
        final = history[-1]
        nll = final.get("mean_nll", 50.0)
        if nll < _THRESHOLDS["nll_low"]:
            flags.append(f"LOW NLL: {nll:.1f} - over-concentration of probability mass")

        # Check reward vs std, high reward but std->0 means single-mode collapse.
        std_r = final.get("std_reward", 1.0)
        mean_r = final.get("mean_reward", 0.0)
        if mean_r > 0.5 and std_r < 0.05:
            flags.append(
                f"REWARD STD COLLAPSED: std={std_r:.4f} with mean={mean_r:.4f}"
            )

        # Check for validity crash over training.
        validities = [h.get("validity", 1.0) for h in history]
        if len(validities) >= 5:
            start_val = np.mean(validities[:3])
            end_val = np.mean(validities[-3:])
            drop = start_val - end_val
            if drop > 0.20:
                flags.append(
                    f"VALIDITY CRASHED during training: {start_val:.1%} -> {end_val:.1%}"
                )
            elif drop > 0.10:
                warnings.append(f"Validity dropped {drop:.0%} over training")

        # Check for uniqueness/diversity crash if tracked.
        uniq_vals = [h.get("uniqueness") for h in history if "uniqueness" in h]
        if len(uniq_vals) >= 5:
            start_u = np.mean(uniq_vals[:3])
            end_u = np.mean(uniq_vals[-3:])
            if start_u - end_u > 0.15:
                flags.append(f"UNIQUENESS CRASHED: {start_u:.1%} -> {end_u:.1%}")

        div_vals = [h.get("diversity") for h in history if "diversity" in h]
        if len(div_vals) >= 5:
            start_d = np.mean(div_vals[:3])
            end_d = np.mean(div_vals[-3:])
            if start_d - end_d > 0.15:
                flags.append(f"DIVERSITY CRASHED: {start_d:.4f} -> {end_d:.4f}")

    # Overall health.
    if flags:
        health = "COLLAPSED" if len(flags) >= 2 else "AT RISK"
    elif warnings:
        health = "CAUTION"
    else:
        health = "HEALTHY"

    return {
        "health": health,
        "flags": flags,
        "warnings": warnings,
        "thresholds": _THRESHOLDS,
    }


# Scaffold novelty analysis.
def scaffold_novelty_analysis(
    generated_smiles: List[str],
    training_smiles: Set[str],
    max_train_for_scaffold: int = 50_000,
) -> Dict:
    """Compare Bemis-Murcko scaffolds of generated molecules to training set.

    Args:
        generated_smiles(list[str]): SMILES from the generative model.
        training_smiles(set[str]): Training set SMILES for novelty reference.
        max_train_for_scaffold(int): If the training set is larger than this,
            a random subsample is used for scaffold extraction.

    Returns:
        dict: Keys include ``n_generated_scaffolds``,
        ``n_training_scaffolds``, ``n_novel_scaffolds``,
        ``scaffold_novelty``, and ``top_scaffolds``.
    """
    from collections import Counter

    # Compute scaffolds for training set (subsample if very large).
    train_list = list(training_smiles)
    if len(train_list) > max_train_for_scaffold:
        import random as _rng

        _rng.seed(0)
        train_list = _rng.sample(train_list, max_train_for_scaffold)
        logger.info(
            "Subsampled %d / %d training SMILES for scaffold analysis",
            max_train_for_scaffold,
            len(training_smiles),
        )

    train_scaffolds: Set[str] = set()
    n_train = len(train_list)
    for i, smi in enumerate(train_list):
        sc = scaffold_smiles(smi)
        if sc:
            train_scaffolds.add(sc)
        if (i + 1) % 10_000 == 0:
            logger.debug("Training scaffolds: %d/%d …", i + 1, n_train)

    # Compute scaffolds for generated molecules (valid only).
    gen_scaffold_counts: Counter = Counter()
    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        sc = scaffold_smiles(Chem.MolToSmiles(mol))
        if sc:
            gen_scaffold_counts[sc] += 1

    gen_scaffolds = set(gen_scaffold_counts.keys())
    novel_scaffolds = gen_scaffolds - train_scaffolds

    top_scaffolds = gen_scaffold_counts.most_common(20)

    return {
        "n_generated_scaffolds": len(gen_scaffolds),
        "n_training_scaffolds": len(train_scaffolds),
        "n_novel_scaffolds": len(novel_scaffolds),
        "scaffold_novelty": len(novel_scaffolds) / len(gen_scaffolds)
        if gen_scaffolds
        else 0.0,
        "top_scaffolds": [{"scaffold": sc, "count": c} for sc, c in top_scaffolds],
    }


def _plot_training_dashboard(
    history: List[Dict],
    stage_label: str,
    seed: int,
    img_dir: Path,
    stage: str,
) -> Optional[Path]:
    """Render a 6-panel training dashboard and save it as PNG.

    Panels: reward, validity, uniqueness, diversity, NLL, and loss.

    Args:
        history(list[dict]): Per-epoch training records.
        stage_label(str): Human-readable stage name for the title.
        seed(int): Random seed (shown in the title).
        img_dir(Path): Output directory for the saved image.
        stage(str): Stage key used in the filename.

    Returns:
        Path | None: Path to saved PNG, or *None* when *history* is empty.
    """
    if not history:
        return None

    epochs = [h["epoch"] for h in history]
    has_uniq = "uniqueness" in history[0]
    has_div = "diversity" in history[0]

    n_panels = 3 + int(has_uniq) + int(has_div) + 1  # +1 for NLL
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()
    panel = 0

    # Reward curve with mean +- std and max.
    ax = axes[panel]
    panel += 1
    mean_r = [h["mean_reward"] for h in history]
    std_r = [h["std_reward"] for h in history]
    max_r = [h["max_reward"] for h in history]
    ax.plot(epochs, mean_r, label="Mean Reward", linewidth=2, color="#1f77b4")
    ax.fill_between(
        epochs,
        [m - s for m, s in zip(mean_r, std_r)],
        [m + s for m, s in zip(mean_r, std_r)],
        alpha=0.2,
        color="#1f77b4",
    )
    ax.plot(epochs, max_r, label="Max Reward", linewidth=1, alpha=0.6, color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("P(active)")
    ax.set_title("Reward")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Validity.
    ax = axes[panel]
    panel += 1
    vals = [h["validity"] for h in history]
    ax.plot(epochs, vals, linewidth=2, color="#2ca02c")
    ax.axhline(
        _THRESHOLDS["validity_low"],
        ls="--",
        color="red",
        alpha=0.5,
        label=f"Threshold ({_THRESHOLDS['validity_low']:.0%})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction")
    ax.set_title("Validity")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Uniqueness.
    if has_uniq:
        ax = axes[panel]
        panel += 1
        uniqs = [h["uniqueness"] for h in history]
        ax.plot(epochs, uniqs, linewidth=2, color="#9467bd")
        ax.axhline(
            _THRESHOLDS["uniqueness_low"],
            ls="--",
            color="red",
            alpha=0.5,
            label=f"Threshold ({_THRESHOLDS['uniqueness_low']:.0%})",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fraction")
        ax.set_title("Uniqueness")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Diversity.
    if has_div:
        ax = axes[panel]
        panel += 1
        divs = [h["diversity"] for h in history]
        ax.plot(epochs, divs, linewidth=2, color="#d62728")
        ax.axhline(
            _THRESHOLDS["diversity_low"],
            ls="--",
            color="red",
            alpha=0.5,
            label=f"Threshold ({_THRESHOLDS['diversity_low']:.0%})",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Tanimoto Distance")
        ax.set_title("Diversity")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # NLL.
    ax = axes[panel]
    panel += 1
    nlls = [h["mean_nll"] for h in history]
    ax.plot(epochs, nlls, linewidth=2, color="#8c564b")
    ax.axhline(
        _THRESHOLDS["nll_low"],
        ls="--",
        color="red",
        alpha=0.5,
        label=f"Collapse ({_THRESHOLDS['nll_low']:.0f})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL")
    ax.set_title("Mean NLL")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Loss.
    ax = axes[panel]
    panel += 1
    losses = [h["loss"] for h in history]
    ax.plot(epochs, losses, linewidth=2, color="#e377c2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("REINFORCE Loss")
    ax.grid(True, alpha=0.3)

    # Hide unused axes.
    for i in range(panel, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"{stage_label} Training Dashboard (seed {seed})",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    path = img_dir / f"{stage}_training_dashboard.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_reward_distribution(
    probs: List[float],
    stage_label: str,
    seed: int,
    img_dir: Path,
    stage: str,
) -> Optional[Path]:
    """Histogram of P(active) for generated molecules.

    Args:
        probs(list[float]): Predicted activity probabilities.
        stage_label(str): Human-readable stage name for the title.
        seed(int): Random seed.
        img_dir(Path): Output directory.
        stage(str): Stage key for the filename.

    Returns:
        Path | None: Saved image path, or *None* when *probs* is empty.
    """
    if not probs:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs, bins=50, edgecolor="black", alpha=0.7, color="#1f77b4")
    ax.axvline(
        np.mean(probs),
        ls="--",
        color="red",
        linewidth=2,
        label=f"Mean = {np.mean(probs):.4f}",
    )
    ax.axvline(
        np.median(probs),
        ls="--",
        color="orange",
        linewidth=2,
        label=f"Median = {np.median(probs):.4f}",
    )
    ax.set_xlabel("P(active)")
    ax.set_ylabel("Count")
    ax.set_title(f"{stage_label} - P(active) Distribution (seed {seed})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = img_dir / f"{stage}_reward_distribution.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_sa_distribution(
    sa_scores: List[float],
    stage_label: str,
    seed: int,
    img_dir: Path,
    stage: str,
) -> Optional[Path]:
    """Histogram of SA scores for generated molecules.

    Args:
        sa_scores(list[float]): Synthetic-accessibility scores.
        stage_label(str): Human-readable stage name for the title.
        seed(int): Random seed.
        img_dir(Path): Output directory.
        stage(str): Stage key for the filename.

    Returns:
        Path | None: Saved image path, or *None* when *sa_scores* is empty.
    """
    if not sa_scores:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sa_scores, bins=40, edgecolor="black", alpha=0.7, color="#2ca02c")
    ax.axvline(
        np.mean(sa_scores),
        ls="--",
        color="red",
        linewidth=2,
        label=f"Mean = {np.mean(sa_scores):.2f}",
    )
    ax.set_xlabel("SA Score (1=easy, 10=hard)")
    ax.set_ylabel("Count")
    ax.set_title(f"{stage_label} - SA Score Distribution (seed {seed})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = img_dir / f"{stage}_sa_distribution.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# Main evaluation function.
def run_evaluation(
    checkpoint_path: Optional[Path] = None,
    n_samples: int = 5000,
    device: str = "cpu",
    seed: int = 42,
    stage: str = "pretrain_rl",
) -> Dict:
    """Full evaluation pipeline for a REINFORCE-tuned LSTM.

    Loads the checkpoint, samples SMILES, computes quality metrics, runs
    both GNN and FP HIV-activity predictors, performs scaffold novelty
    analysis, assesses mode collapse indicators, and saves plots and a
    JSON report.

    Args:
        checkpoint_path(Path | None): Explicit checkpoint path.  When
            *None* the default from *config* is used.
        n_samples(int): Number of SMILES strings to sample.
        device(str): Torch device string.
        seed(int): Random seed that selects the checkpoint directory.
        stage(str): ``"pretrain_rl"`` or ``"transfer_rl"``.

    Returns:
        dict: Comprehensive evaluation metrics.
    """

    config = RLConfig(seed=seed, stage=stage)
    if checkpoint_path is None:
        checkpoint_path = config.checkpoint_outdir / f"best_{stage}.pt"

    logger.info("Checkpoint: %s", checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Run RL training first: "
            f"uv run python -m src.reinforce.train --stage {stage}"
        )

    # Load model.
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

    training_set = pretrain_smiles | transfer_smiles

    # Evaluate generated molecules.
    logger.info("Evaluating …")
    results = evaluate_generated_molecules(generated, training_set)

    stage_label = stage.replace("_", " ").upper()
    logger.info("=" * 60)
    logger.info("%s EVALUATION - seed %d", stage_label, seed)
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

    # HIV activity via GNN predictor.
    logger.info("HIV activity prediction (GNN classifier) …")
    gnn_ckpt = gnn_checkpoint_for_seed(seed)
    predictor = load_hiv_predictor(gnn_ckpt, device=device)
    hiv_results = None
    all_probs: List[float] = []
    if predictor is not None:
        valid_smiles = list(dict.fromkeys(results["valid_smiles"]))[:1000]
        hiv_results = predict_hiv_activity(predictor, valid_smiles)
        if hiv_results:
            all_probs = [r["prob"] for r in hiv_results]
            labels = [r["label"] for r in hiv_results]
            active_frac = sum(labels) / len(labels)
            results["hiv_mean_prob"] = float(np.mean(all_probs))
            results["hiv_median_prob"] = float(np.median(all_probs))
            results["hiv_std_prob"] = float(np.std(all_probs))
            results["hiv_max_prob"] = float(np.max(all_probs))
            results["hiv_active_frac"] = active_frac
            results["hiv_n_scored"] = len(hiv_results)

            logger.info("Scored:         %d molecules", len(hiv_results))
            logger.info("Mean P(active): %.4f", results["hiv_mean_prob"])
            logger.info("Median P(act):  %.4f", results["hiv_median_prob"])
            logger.info("Std P(active):  %.4f", results["hiv_std_prob"])
            logger.info("Max P(active):  %.4f", results["hiv_max_prob"])
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

    # Independent FP evaluator.
    logger.info("Independent FP evaluator …")
    fp_eval = load_fp_evaluator(seed)
    fp_results_list = None
    if fp_eval is not None:
        valid_smiles_for_fp = list(dict.fromkeys(results["valid_smiles"]))[:1000]
        fp_results_list = fp_eval.predict_batch(valid_smiles_for_fp)
        if fp_results_list:
            fp_probs = [r["prob"] for r in fp_results_list]
            fp_labels = [r["label"] for r in fp_results_list]
            fp_active_frac = sum(fp_labels) / len(fp_labels)
            results["fp_mean_prob"] = float(np.mean(fp_probs))
            results["fp_median_prob"] = float(np.median(fp_probs))
            results["fp_active_frac"] = fp_active_frac
            results["fp_n_scored"] = len(fp_results_list)

            logger.info("Scored:         %d molecules", len(fp_results_list))
            logger.info("Mean P(active): %.4f", results["fp_mean_prob"])
            logger.info(
                "Active:         %d (%.1f%%)", sum(fp_labels), fp_active_frac * 100
            )

            # Agreement with GNN (if both scored the same molecules).
            if hiv_results:
                gnn_map = {r["smiles"]: r["label"] for r in hiv_results}
                agree, total_cmp = 0, 0
                for r in fp_results_list:
                    if r["smiles"] in gnn_map:
                        total_cmp += 1
                        if r["label"] == gnn_map[r["smiles"]]:
                            agree += 1
                if total_cmp > 0:
                    agreement = agree / total_cmp
                    results["gnn_fp_agreement"] = agreement
                    results["gnn_fp_n_compared"] = total_cmp
                    logger.info(
                        "GNN-FP agreement: %.1f%% (%d/%d)",
                        agreement * 100,
                        agree,
                        total_cmp,
                    )
        else:
            logger.warning("No valid molecules could be scored by FP evaluator.")
    else:
        logger.warning(
            "FP evaluator not available - skipping. "
            "Train with: uv run python -m src.gnn.fingerprint_evaluator"
        )

    # Scaffold novelty analysis.
    logger.info("Scaffold novelty analysis …")
    scaffold_results = scaffold_novelty_analysis(generated, training_set)
    results["scaffold_analysis"] = scaffold_results
    logger.info("Generated scaffolds: %d", scaffold_results["n_generated_scaffolds"])
    logger.info("Training scaffolds:  %d", scaffold_results["n_training_scaffolds"])
    logger.info(
        "Novel scaffolds:     %d (%.1f%%)",
        scaffold_results["n_novel_scaffolds"],
        scaffold_results["scaffold_novelty"] * 100,
    )

    # Collapse assessment.
    meta_path = config.checkpoint_outdir / "meta.json"
    history: List[Dict] = []
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        history = meta.get("history", [])

    collapse = assess_collapse(results, history)
    results["collapse_assessment"] = collapse

    logger.info("=" * 60)
    logger.info("COLLAPSE ASSESSMENT: %s", collapse["health"])
    logger.info("=" * 60)
    if collapse["flags"]:
        for flag in collapse["flags"]:
            logger.warning("[FLAG]    %s", flag)
    if collapse["warnings"]:
        for warn in collapse["warnings"]:
            logger.warning("[WARN]    %s", warn)
    if not collapse["flags"] and not collapse["warnings"]:
        logger.info("All metrics within healthy ranges.")
    if history:
        final = history[-1]
        logger.info("Final training epoch (%d):", final["epoch"])
        logger.info("  Mean reward:  %.4f", final["mean_reward"])
        logger.info("  Std reward:   %.4f", final["std_reward"])
        logger.info("  Validity:     %.2f%%", final["validity"] * 100)
        if "uniqueness" in final:
            logger.info("  Uniqueness:   %.2f%%", final["uniqueness"] * 100)
        if "diversity" in final:
            logger.info("  Diversity:    %.4f", final["diversity"])
        logger.info("  Mean NLL:     %.1f", final["mean_nll"])
    logger.info("=" * 60)

    # Plots.
    img_dir = config.img_outdir
    img_dir.mkdir(parents=True, exist_ok=True)

    saved_images: List[str] = []

    # Training dashboard.
    path = _plot_training_dashboard(history, stage_label, seed, img_dir, stage)
    if path:
        saved_images.append(str(path))
        logger.info("Training dashboard  -> %s", path)

    # Reward (P(active)) distribution.
    path = _plot_reward_distribution(all_probs, stage_label, seed, img_dir, stage)
    if path:
        saved_images.append(str(path))
        logger.info("Reward distribution -> %s", path)

    # SA score distribution.
    if results.get("sa_scores"):
        path = _plot_sa_distribution(
            results["sa_scores"], stage_label, seed, img_dir, stage
        )
        if path:
            saved_images.append(str(path))
            logger.info("SA distribution     -> %s", path)

    # Top-20 molecules grid.
    if predictor is not None and hiv_results:
        sorted_hits = sorted(hiv_results, key=lambda r: r["prob"], reverse=True)[:20]
        top_mols = [Chem.MolFromSmiles(r["smiles"]) for r in sorted_hits]
        valid_pairs = [(m, r) for m, r in zip(top_mols, sorted_hits) if m is not None]
        if valid_pairs:
            top_mols, sorted_hits = zip(*valid_pairs)
            legends = [f"P(active)={r['prob']:.3f}" for r in sorted_hits]

            fig, ax = plt.subplots(figsize=(14, 12))
            grid_img = Draw.MolsToGridImage(
                list(top_mols),
                molsPerRow=5,
                subImgSize=(300, 250),
                legends=legends,
            )
            ax.imshow(grid_img)
            ax.axis("off")
            ax.set_title(
                f"Top {len(top_mols)} Generated Molecules by Predicted HIV Activity\n"
                f"({stage_label}, seed {seed})",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout()
            img_path = img_dir / f"{stage}_top_molecules.png"
            fig.savefig(str(img_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_images.append(str(img_path))
            logger.info("Top molecules       -> %s", img_path)

    results["saved_images"] = saved_images

    # Make results JSON-serializable by excluding non-serializable entries.
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
            "saved_images",
        )
    }
    eval_dir = config.eval_outdir
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / f"{stage}_eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Evaluation JSON     -> %s", eval_path)

    return results


# Entry point for command-line execution.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate REINFORCE-tuned SMILES LSTM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument(
        "--stage",
        type=str,
        default="pretrain_rl",
        choices=["pretrain_rl", "transfer_rl"],
    )
    args = parser.parse_args()

    run_evaluation(
        n_samples=args.n_samples,
        device=args.device,
        seed=args.seed,
        stage=args.stage,
    )
