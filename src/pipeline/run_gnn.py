"""Pipeline runner: tune GNN hyperparameters -> train 3 seeds -> evaluate -> aggregate.

Usage::

    uv run python -m src.pipeline.run_gnn
    uv run python -m src.pipeline.run_gnn --skip-tuning
    uv run python -m src.pipeline.run_gnn --skip-train
    uv run python -m src.pipeline.run_gnn --only-aggregate
    uv run python -m src.pipeline.run_gnn --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.config.data import root_dir
from src.config.seed import DEFAULT_SEEDS
from src.config.training import GNNConfig

logger = logging.getLogger(__name__)


# Metrics to include in the GNN summary JSON (mean +- std across seeds).
GNN_SUMMARY_METRICS = [
    "pr_auc",
    "roc_auc",
    "mcc",
    "balanced_acc",
    "f1",
]


def _load_seed_results(seeds: list[int]) -> list[dict]:
    """Load per-seed GNN evaluation JSONs.

    Args:
        seeds(list[int]): Seeds to load.

    Returns:
        list[dict]: One result dict per seed with a saved JSON.
    """
    results = []
    for seed in seeds:
        cfg = GNNConfig(seed=seed)
        path = cfg.eval_outdir / "gnn_eval_results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            data["seed"] = seed
            results.append(data)
        else:
            logger.warning("Missing %s", path)
    return results


def _aggregate(results: list[dict]) -> dict:
    """Compute mean +- std for GNN summary metrics.

    Args:
        results(list[dict]): Per-seed evaluation results.

    Returns:
        dict: Mapping metric name -> ``{mean, std, values, n}``.
    """
    summary = {}
    for key in GNN_SUMMARY_METRICS:
        values = [r[key] for r in results if r.get(key) is not None]
        if values:
            summary[key] = {
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values)), 6),
                "values": [round(v, 6) for v in values],
                "n": len(values),
            }
    return summary


def aggregate_and_save(seeds: list[int]) -> None:
    """Aggregate per-seed GNN results and save a summary JSON.

    Args:
        seeds(list[int]): Seeds whose results should be aggregated.
    """
    agg_dir = root_dir / "evaluation" / "summary"
    agg_dir.mkdir(parents=True, exist_ok=True)

    results = _load_seed_results(seeds)
    if not results:
        logger.warning("No GNN results found - skipping summary.")
        return

    summary = _aggregate(results)
    out = {
        "stage": "gnn",
        "seeds": [r["seed"] for r in results],
        "metrics": summary,
        "per_seed": results,
    }
    path = agg_dir / "gnn_summary.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info("=" * 60)
    logger.info("GNN SUMMARY  (%d seeds)", len(results))
    logger.info("=" * 60)
    for key, vals in summary.items():
        logger.info("%-20s: %.4f +- %.4f", key, vals["mean"], vals["std"])
    logger.info("=" * 60)
    logger.info("Saved to %s", path)


# CLI
def main() -> None:
    """Parse CLI arguments and orchestrate the GNN pipeline."""
    parser = argparse.ArgumentParser(
        description="GNN pipeline: tune -> train (3 seeds) -> evaluate -> aggregate"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip tuning, use existing best hyperparameters",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training, only evaluate existing checkpoints",
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip evaluation, only train"
    )
    parser.add_argument(
        "--only-aggregate",
        action="store_true",
        help="Skip all training/eval, just aggregate existing results",
    )
    parser.add_argument(
        "--max-tuning-epochs",
        type=int,
        default=30,
        help="Max epochs per configuration during tuning",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter tuning",
    )
    parser.add_argument(
        "--use-config",
        action="store_true",
        help="Use GNNConfig defaults directly, skip tuning param override",
    )
    args = parser.parse_args()

    if args.only_aggregate:
        aggregate_and_save(args.seeds)
        return

    # Hyperparameter tuning.
    from src.gnn.tuning import apply_tuned_params, load_best_params, tune

    if not args.skip_tuning:
        logger.info("#" * 70)
        logger.info("STEP 1: HYPERPARAMETER TUNING")
        logger.info("#" * 70)
        tune(
            device=args.device,
            seed=args.seeds[0],
            max_epochs=args.max_tuning_epochs,
            n_trials=args.n_trials,
        )

    best_params = {}
    if not args.use_config:
        try:
            best_params = load_best_params()
            logger.info("Using tuned parameters: %s", best_params)
        except FileNotFoundError:
            logger.warning("No tuning results found - using default hyperparameters.")
    else:
        logger.info("--use-config: using GNNConfig defaults (skipping tuned params).")

    # Train across seeds.
    from src.gnn.train import train

    if not args.skip_train:
        for i, seed in enumerate(args.seeds, 1):
            logger.info("#" * 70)
            logger.info(
                "STEP 2: GNN TRAINING - SEED %d  (%d/%d)", seed, i, len(args.seeds)
            )
            logger.info("#" * 70)

            config = GNNConfig(seed=seed)
            if best_params:
                apply_tuned_params(config, best_params)
            train(config, device=args.device)

    # Evaluate.
    from src.gnn.evaluation import run_evaluation

    if not args.skip_eval:
        for i, seed in enumerate(args.seeds, 1):
            logger.info("#" * 70)
            logger.info(
                "STEP 3: GNN EVALUATION - SEED %d  (%d/%d)", seed, i, len(args.seeds)
            )
            logger.info("#" * 70)
            run_evaluation(device=args.device, seed=seed)

    # Aggregate.
    logger.info("#" * 70)
    logger.info("STEP 4: AGGREGATION")
    logger.info("#" * 70)
    aggregate_and_save(args.seeds)

    logger.info("#" * 70)
    logger.info("GNN PIPELINE COMPLETE - %d seeds", len(args.seeds))
    logger.info("#" * 70)


if __name__ == "__main__":
    main()
