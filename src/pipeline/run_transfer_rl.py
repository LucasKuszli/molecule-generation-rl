"""Pipeline: Transfer + RL across multiple seeds -> evaluate -> aggregate.

Usage::

    uv run python -m src.pipeline.run_transfer_rl
    uv run python -m src.pipeline.run_transfer_rl --device cuda
    uv run python -m src.pipeline.run_transfer_rl --skip-train
    uv run python -m src.pipeline.run_transfer_rl --only-aggregate
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from src.config.data import root_dir
from src.config.seed import DEFAULT_SEEDS
from src.config.training import RLConfig

logger = logging.getLogger(__name__)

STAGE = "transfer_rl"

SUMMARY_METRICS = [
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


def _load_seed_results(seeds: list[int]) -> list[dict]:
    """Load per-seed transfer+RL evaluation JSONs.

    Args:
        seeds(list[int]): Seeds to load.

    Returns:
        list[dict]: One result dict per seed with a saved JSON.
    """
    results = []
    for seed in seeds:
        cfg = RLConfig(seed=seed, stage=STAGE)
        path = cfg.eval_outdir / f"{STAGE}_eval_results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            data["seed"] = seed
            results.append(data)
        else:
            logger.warning("Missing %s", path)
    return results


def _aggregate(results: list[dict]) -> dict:
    """Compute mean +- std for transfer+RL summary metrics.

    Args:
        results(list[dict]): Per-seed evaluation results.

    Returns:
        dict: Mapping metric name -> ``{mean, std, values, n}``.
    """
    summary = {}
    for key in SUMMARY_METRICS:
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
    """Aggregate per-seed transfer+RL results and save a summary JSON.

    Args:
        seeds(list[int]): Seeds whose results should be aggregated.
    """
    agg_dir = root_dir / "evaluation" / "summary"
    agg_dir.mkdir(parents=True, exist_ok=True)

    results = _load_seed_results(seeds)
    if not results:
        logger.warning("No %s results found - skipping summary.", STAGE)
        return

    summary = _aggregate(results)
    out = {
        "stage": STAGE,
        "seeds": [r["seed"] for r in results],
        "metrics": summary,
        "per_seed": results,
    }

    path = agg_dir / f"{STAGE}_summary.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info("=" * 60)
    logger.info("%s SUMMARY  (%d seeds)", STAGE.upper(), len(results))
    logger.info("=" * 60)
    for key, vals in summary.items():
        logger.info("%-20s: %.4f +- %.4f", key, vals["mean"], vals["std"])
    logger.info("=" * 60)
    logger.info("Saved to %s", path)


def main() -> None:
    """Parse CLI arguments and orchestrate the transfer+RL pipeline."""
    parser = argparse.ArgumentParser(
        description="Transfer + RL pipeline: train -> evaluate -> aggregate"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip RL training, only evaluate + aggregate",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation, only train + aggregate",
    )
    parser.add_argument(
        "--only-aggregate",
        action="store_true",
        help="Skip all training/eval, just aggregate existing results",
    )
    args = parser.parse_args()

    if args.only_aggregate:
        aggregate_and_save(args.seeds)
        return

    from src.reinforce.evaluation import run_evaluation
    from src.reinforce.train import train

    for i, seed in enumerate(args.seeds, 1):
        config = RLConfig(seed=seed, stage=STAGE)

        if not args.skip_train:
            logger.info("#" * 70)
            logger.info(
                "TRANSFER+RL TRAINING - SEED %d  (%d/%d)", seed, i, len(args.seeds)
            )
            logger.info("#" * 70)
            train(config, device=args.device)

        if not args.skip_eval:
            logger.info("#" * 70)
            logger.info(
                "TRANSFER+RL EVALUATION - SEED %d  (%d/%d)", seed, i, len(args.seeds)
            )
            logger.info("#" * 70)
            run_evaluation(
                n_samples=args.n_samples,
                device=args.device,
                seed=seed,
                stage=STAGE,
            )

    logger.info("#" * 70)
    logger.info("AGGREGATION")
    logger.info("#" * 70)
    aggregate_and_save(args.seeds)

    logger.info("#" * 70)
    logger.info("TRANSFER+RL PIPELINE COMPLETE - %d seeds", len(args.seeds))
    logger.info("#" * 70)


if __name__ == "__main__":
    main()
