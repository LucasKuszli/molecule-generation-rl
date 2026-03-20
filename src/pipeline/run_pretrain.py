"""Run pre-training and evaluation across multiple seeds.

Usage::

    uv run python -m src.pipeline.run_pretrain
    uv run python -m src.pipeline.run_pretrain --seeds 1 2 3
    uv run python -m src.pipeline.run_pretrain --seeds 42
"""

from __future__ import annotations

import argparse
import logging

from src.config.seed import DEFAULT_SEEDS
from src.config.training import PretrainingConfig
from src.pretrain.evaluation import run_evaluation
from src.pretrain.train import train

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI arguments and run pre-training for each seed."""
    parser = argparse.ArgumentParser(
        description="Pre-train & evaluate over multiple seeds",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Random seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Molecules to sample during evaluation",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--skip-train", action="store_true", help="Skip training, only run evaluation"
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip evaluation, only run training"
    )
    args = parser.parse_args()

    for i, seed in enumerate(args.seeds, 1):
        logger.info("=" * 70)
        logger.info("SEED %d  (%d/%d)", seed, i, len(args.seeds))
        logger.info("=" * 70)

        config = PretrainingConfig(seed=seed)

        if not args.skip_train:
            train(config)

        if not args.skip_eval:
            run_evaluation(n_samples=args.n_samples, device=args.device, seed=seed)

    logger.info("=" * 70)
    logger.info("ALL DONE - %d seeds completed", len(args.seeds))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
