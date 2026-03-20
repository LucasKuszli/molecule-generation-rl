"""Full pipeline: pretrain ×3 -> transfer ×3 (matched seeds), then aggregate.

Usage::

    uv run python -m src.pipeline.run_full
    uv run python -m src.pipeline.run_full --seeds 42 67 1337
    uv run python -m src.pipeline.run_full --skip-pretrain
    uv run python -m src.pipeline.run_full --skip-transfer
    uv run python -m src.pipeline.run_full --only-aggregate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.config.data import root_dir
from src.config.seed import DEFAULT_SEEDS
from src.config.training import PretrainingConfig, TransferLearningConfig
from src.pretrain.evaluation import run_evaluation as pretrain_eval
from src.pretrain.train import train as pretrain
from src.transfer.evaluation import run_evaluation as transfer_eval
from src.transfer.train import train as transfer_train

logger = logging.getLogger(__name__)


SUMMARY_METRICS = [
    "validity",
    "uniqueness",
    "novelty",
    "diversity",
    "sa_score_mean",
    "hiv_mean_prob",
    "hiv_active_frac",
]


def _load_seed_results(stage: str, seeds: list[int]) -> list[dict]:
    """Load per-seed evaluation JSONs for a given stage.

    Args:
        stage(str): ``"pretrain"`` or ``"transfer"``.
        seeds(list[int]): Seeds to load.

    Returns:
        list[dict]: One result dict per seed that has a saved JSON.
    """
    results = []
    for seed in seeds:
        if stage == "pretrain":
            cfg = PretrainingConfig(seed=seed)
            path = cfg.eval_outdir / "pretrain_eval_results.json"
        else:
            cfg = TransferLearningConfig(seed=seed)
            path = cfg.eval_outdir / "transfer_eval_results.json"

        if path.exists():
            with open(path) as f:
                data = json.load(f)
            data["seed"] = seed
            results.append(data)
        else:
            logger.warning("Missing %s", path)
    return results


def _aggregate(results: list[dict]) -> dict:
    """Compute mean +- std for summary metrics across seeds.

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
    """Aggregate per-seed results for both stages and save summaries.

    Args:
        seeds(list[int]): Seeds whose results should be aggregated.
    """
    agg_dir = root_dir / "evaluation" / "summary"
    agg_dir.mkdir(parents=True, exist_ok=True)

    for stage in ["pretrain", "transfer"]:
        results = _load_seed_results(stage, seeds)
        if not results:
            logger.warning("No %s results found - skipping summary.", stage)
            continue

        summary = _aggregate(results)
        out = {
            "stage": stage,
            "seeds": [r["seed"] for r in results],
            "metrics": summary,
            "per_seed": results,
        }

        path = agg_dir / f"{stage}_summary.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

        logger.info("=" * 60)
        logger.info("%s SUMMARY  (%d seeds)", stage.upper(), len(results))
        logger.info("=" * 60)
        for key, vals in summary.items():
            logger.info("%-20s: %.4f +- %.4f", key, vals["mean"], vals["std"])
        logger.info("=" * 60)
        logger.info("Saved to %s", path)


# main CLI entry point to run the full pretrain -> transfer pipeline, with optional skipping of stages.
def main() -> None:
    """Parse CLI arguments and orchestrate pretrain -> transfer pipeline."""
    parser = argparse.ArgumentParser(
        description="Full pipeline: pretrain -> transfer -> aggregate"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument(
        "--only-aggregate",
        action="store_true",
        help="Skip all training/eval, just aggregate existing results",
    )
    args = parser.parse_args()

    if not args.only_aggregate:
        if not args.skip_pretrain:
            for i, seed in enumerate(args.seeds, 1):
                logger.info("#" * 70)
                logger.info("PRETRAIN - SEED %d  (%d/%d)", seed, i, len(args.seeds))
                logger.info("#" * 70)
                pretrain(PretrainingConfig(seed=seed))
                pretrain_eval(n_samples=args.n_samples, device=args.device, seed=seed)

        if not args.skip_transfer:
            for i, seed in enumerate(args.seeds, 1):
                logger.info("#" * 70)
                logger.info("TRANSFER - SEED %d  (%d/%d)", seed, i, len(args.seeds))
                logger.info("#" * 70)
                transfer_train(TransferLearningConfig(seed=seed))
                transfer_eval(n_samples=args.n_samples, device=args.device, seed=seed)

    aggregate_and_save(args.seeds)

    logger.info("#" * 70)
    logger.info("FULL PIPELINE COMPLETE - %d seeds", len(args.seeds))
    logger.info("#" * 70)


if __name__ == "__main__":
    main()
