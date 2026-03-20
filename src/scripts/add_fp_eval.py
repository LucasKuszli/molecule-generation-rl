"""Add fingerprint-evaluator metrics to pretrain & transfer eval results.

Re-generates molecules from each checkpoint, scores with both the GNN and
the FP evaluator, and overwrites the eval JSON with the enriched results.
Finally re-aggregates the summary JSONs.

Usage:
    uv run python -m src.scripts.add_fp_eval --device cpu
    uv run python -m src.scripts.add_fp_eval --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Set

import numpy as np

from src.config.data import root_dir
from src.config.seed import DEFAULT_SEEDS
from src.data.loader import get_pretrain_datasplit, get_transfer_dataset
from src.gnn.fingerprint_evaluator import load_fp_evaluator
from src.pretrain.evaluation import (
    evaluate_generated_molecules,
    generate_smiles,
    gnn_checkpoint_for_seed,
    load_hiv_predictor,
    load_model_from_checkpoint,
    predict_hiv_activity,
)
from src.reinforce.evaluation import scaffold_novelty_analysis

logger = logging.getLogger(__name__)


def evaluate_stage(
    stage: str,
    seed: int,
    n_samples: int,
    device: str,
    training_set: Set[str],
) -> Dict:
    """Generate molecules, score with GNN + FP evaluator, and save results.

    Args:
        stage(str): Pipeline stage (``"pretrain"`` or ``"transfer"``).
        seed(int): Random seed identifying the checkpoint.
        n_samples(int): Number of SMILES to sample from the model.
        device(str): Torch device string.
        training_set(Set[str]): Known training SMILES for novelty computation.

    Returns:
        Dict of serialisable evaluation results.
    """
    if stage == "pretrain":
        ckpt_path = (
            root_dir / "artifacts" / "pretrain" / f"seed_{seed}" / "best_prior.pt"
        )
        eval_dir = root_dir / "evaluation" / "pretrain" / f"seed_{seed}"
        eval_filename = "pretrain_eval_results.json"
    elif stage == "transfer":
        ckpt_path = (
            root_dir
            / "artifacts"
            / "transfer_learning"
            / f"seed_{seed}"
            / "best_transfer.pt"
        )
        eval_dir = root_dir / "evaluation" / "transfer_learning" / f"seed_{seed}"
        eval_filename = "transfer_eval_results.json"
    else:
        raise ValueError(f"Unknown stage: {stage}")

    logger.info("=" * 60)
    logger.info("%s - seed %d", stage.upper(), seed)
    logger.info("=" * 60)
    logger.info("Checkpoint: %s", ckpt_path)

    if not ckpt_path.exists():
        logger.warning("Checkpoint not found - skipping")
        return {}

    model, vocab = load_model_from_checkpoint(ckpt_path, device)
    logger.info(
        "Loaded model (%s params)", f"{sum(p.numel() for p in model.parameters()):,}"
    )

    logger.info("Sampling %d molecules …", n_samples)
    generated = generate_smiles(model, vocab, n=n_samples, device=device)

    logger.info("Evaluating validity / uniqueness / novelty / diversity / SA …")
    results = evaluate_generated_molecules(generated, training_set)
    logger.info(
        "  Valid: %s (%.1f%%)", f"{results['n_valid']:,}", results["validity"] * 100
    )
    logger.info(
        "  Unique: %s (%.1f%%)", f"{results['n_unique']:,}", results["uniqueness"] * 100
    )
    logger.info(
        "  Novel: %s (%.1f%%)", f"{results['n_novel']:,}", results["novelty"] * 100
    )
    logger.info("  Diversity: %.4f", results["diversity"])
    if results["sa_score_mean"] is not None:
        logger.info(
            "  SA Score: %.2f +- %.2f",
            results["sa_score_mean"],
            results["sa_score_std"],
        )

    # GNN scoring
    logger.info("GNN scoring …")
    gnn_ckpt = gnn_checkpoint_for_seed(seed)
    predictor = load_hiv_predictor(gnn_ckpt, device=device)
    if predictor is not None:
        valid_smiles = list(dict.fromkeys(results["valid_smiles"]))[:1000]
        hiv_results = predict_hiv_activity(predictor, valid_smiles)
        if hiv_results:
            probs = [r["prob"] for r in hiv_results]
            labels = [r["label"] for r in hiv_results]
            results["hiv_mean_prob"] = float(np.mean(probs))
            results["hiv_median_prob"] = float(np.median(probs))
            results["hiv_std_prob"] = float(np.std(probs))
            results["hiv_max_prob"] = float(np.max(probs))
            results["hiv_active_frac"] = sum(labels) / len(labels)
            results["hiv_n_scored"] = len(hiv_results)
            logger.info("  Mean P(active): %.4f", results["hiv_mean_prob"])
            logger.info("  Active frac:    %.1f%%", results["hiv_active_frac"] * 100)
    else:
        hiv_results = None
        logger.info("  GNN not available - skipped")

    # FP evaluator scoring
    logger.info("FP evaluator scoring …")
    fp_eval = load_fp_evaluator(seed)
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
            logger.info("  Mean P(active): %.4f", results["fp_mean_prob"])
            logger.info("  Active frac:    %.1f%%", fp_active_frac * 100)

            # GNN-FP agreement
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
                        "  GNN-FP agreement: %.1f%% (%d/%d)",
                        agreement * 100,
                        agree,
                        total_cmp,
                    )
    else:
        logger.info("  FP evaluator not available - skipped")

    # Scaffold novelty
    logger.info("Scaffold novelty analysis …")
    scaffold_results = scaffold_novelty_analysis(generated, training_set)
    results["scaffold_analysis"] = scaffold_results
    logger.info(
        "  Novel scaffolds: %d (%.1f%%)",
        scaffold_results["n_novel_scaffolds"],
        scaffold_results["scaffold_novelty"] * 100,
    )

    # Save results
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
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / eval_filename
    with open(eval_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved → %s", eval_path)

    return serializable


# Aggregation: re-compute summary stats across seeds, now including the FP metrics.
PRETRAIN_METRICS = [
    "validity",
    "uniqueness",
    "novelty",
    "diversity",
    "sa_score_mean",
    "hiv_mean_prob",
    "hiv_median_prob",
    "hiv_std_prob",
    "hiv_max_prob",
    "hiv_active_frac",
    "fp_mean_prob",
    "fp_median_prob",
    "fp_active_frac",
    "gnn_fp_agreement",
]


def aggregate_stage(stage: str, seeds: list[int]) -> None:
    """Aggregate per-seed results into a single summary JSON.

    Args:
        stage(str): Pipeline stage (``"pretrain"`` or ``"transfer"``).
        seeds(list[int]): Seed values to aggregate over.
    """
    if stage == "pretrain":
        eval_dirs = [root_dir / "evaluation" / "pretrain" / f"seed_{s}" for s in seeds]
        filenames = ["pretrain_eval_results.json"] * len(seeds)
        summary_name = "pretrain_summary.json"
    elif stage == "transfer":
        eval_dirs = [
            root_dir / "evaluation" / "transfer_learning" / f"seed_{s}" for s in seeds
        ]
        filenames = ["transfer_eval_results.json"] * len(seeds)
        summary_name = "transfer_summary.json"
    else:
        raise ValueError(stage)

    results = []
    for d, fn, seed in zip(eval_dirs, filenames, seeds):
        path = d / fn
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            data["seed"] = seed
            results.append(data)
        else:
            logger.warning("Missing %s", path)

    if not results:
        logger.warning("No results for %s - skipping", stage)
        return

    summary: Dict = {}
    for key in PRETRAIN_METRICS:
        values = [r[key] for r in results if r.get(key) is not None]
        if values:
            summary[key] = {
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values)), 6),
                "values": [round(v, 6) for v in values],
                "n": len(values),
            }

    out = {
        "stage": stage,
        "seeds": [r["seed"] for r in results],
        "metrics": summary,
        "per_seed": results,
    }

    agg_dir = root_dir / "evaluation" / "summary"
    agg_dir.mkdir(parents=True, exist_ok=True)
    agg_path = agg_dir / summary_name
    with open(agg_path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info("=" * 60)
    logger.info("%s SUMMARY (%d seeds)", stage.upper(), len(results))
    logger.info("=" * 60)
    for key, vals in summary.items():
        logger.info("  %-20s: %.4f +- %.4f", key, vals["mean"], vals["std"])
    logger.info("Saved → %s", agg_path)


# CLI entry point: loop over stages and seeds, evaluate, and aggregate.
def main() -> None:
    """Parse CLI arguments and run FP evaluation + aggregation."""
    parser = argparse.ArgumentParser(
        description="Add FP-evaluator metrics to pretrain & transfer evaluation results"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=["pretrain", "transfer"],
        choices=["pretrain", "transfer"],
    )
    args = parser.parse_args()

    logger.info("Loading training sets …")
    pretrain_data = get_pretrain_datasplit()
    pretrain_smiles = set(pretrain_data.train_set["SMILES"].tolist())

    transfer_data = get_transfer_dataset()
    transfer_smiles = set(transfer_data.train_set["SMILES"].tolist())

    training_set = pretrain_smiles | transfer_smiles
    logger.info("Combined training set: %s SMILES", f"{len(training_set):,}")

    for stage in args.stages:
        for seed in args.seeds:
            evaluate_stage(
                stage=stage,
                seed=seed,
                n_samples=args.n_samples,
                device=args.device,
                training_set=training_set,
            )

        aggregate_stage(stage, args.seeds)

    logger.info("#" * 60)
    logger.info("ALL DONE")
    logger.info("#" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
