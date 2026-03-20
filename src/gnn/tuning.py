"""Hyperparameter tuning for the GNN via Optuna (TPE) on the HIV validation set.

Uses Optuna's Tree-structured Parzen Estimator to efficiently search
the hyperparameter space, optimising PR-AUC on the validation set.
Results are saved to ``artifacts/gnn/tuning/``.

Usage::

    uv run python -m src.gnn.tuning
    uv run python -m src.gnn.tuning --device cuda --n-trials 50
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import optuna
import torch

from src.config.training import GNNConfig
from src.gnn.model import HIVPredictor
from src.gnn.train import _build_loader, prepare_dataloaders, train_loop

logger = logging.getLogger(__name__)


# Default search space for tuning.
SEARCH_SPACE = {
    "batch_size": [32, 64, 128],
    "hidden_dim": [64, 128, 256],
    "num_layers": [2, 3, 4],
}


def tune(
    device: str = "cpu",
    seed: int = 42,
    search_space: Dict[str, list | tuple] | None = None,
    max_epochs: int = 30,
    n_trials: int = 50,
) -> Dict:
    """Run Optuna TPE search over hyperparameters.

    Args:
        device(str): torch device string.
        seed(int): random seed.
        search_space(dict | None): mapping of param name to list/tuple of values.
        max_epochs(int): training budget per trial.
        n_trials(int): number of Optuna trials.

    Returns:
        Dict with best config and all trial results.
    """

    if search_space is None:
        search_space = SEARCH_SPACE

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = GNNConfig(seed=seed)
    out_dir = config.tuning_outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data to discover feature dim (loaders rebuilt per trial for batch_size).
    tmp_loader, _, _ = prepare_dataloaders(config)
    in_dim = next(iter(tmp_loader)).x.size(-1)
    del tmp_loader

    all_results: List[Dict] = []

    def objective(trial: optuna.Trial) -> float:
        params: Dict = {}
        for key, space in search_space.items():
            if isinstance(space, list):
                params[key] = trial.suggest_categorical(key, space)
            else:
                params[key] = trial.suggest_float(key, space[0], space[1])

        trial_config = GNNConfig(seed=seed)
        trial_config.training.num_epochs = max_epochs
        trial_config.training.patience = 10

        for k, v in params.items():
            if hasattr(trial_config.training, k):
                setattr(trial_config.training, k, v)
            if hasattr(trial_config.model, k):
                setattr(trial_config.model, k, v)

        # Rebuild loaders with trial batch_size.
        train_loader, val_loader, _ = prepare_dataloaders(trial_config)

        # Reset seeds for fair comparison.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = HIVPredictor(
            in_channels=in_dim,
            edge_dim=trial_config.model.edge_dim,
            hidden_dim=trial_config.model.hidden_dim,
            num_layers=trial_config.model.num_layers,
            dropout=trial_config.model.dropout,
        ).to(device)

        model, threshold, history = train_loop(
            train_loader,
            val_loader,
            model,
            trial_config,
            device,
        )

        trial_best = max(history, key=lambda h: h["pr_auc"]) if history else {}
        pr_auc = trial_best.get("pr_auc", 0.0)

        result = {
            **params,
            "pr_auc": pr_auc,
            "roc_auc": trial_best.get("roc_auc", 0.0),
            "mcc": trial_best.get("mcc", 0.0),
            "balanced_acc": trial_best.get("balanced_acc", 0.0),
            "best_epoch": trial_best.get("epoch", 0),
            "threshold": threshold,
        }
        all_results.append(result)
        return pr_auc

    logger.info("=" * 60)
    logger.info("  GNN HYPERPARAMETER TUNING (Optuna TPE) - %d trials", n_trials)
    logger.info(
        "  Seed: %d  |  Device: %s  |  Max epochs: %d", seed, device, max_epochs
    )
    logger.info("=" * 60)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Build best_combo dict matching the previous format.
    best_trial = study.best_trial
    best_combo = {**best_trial.params}
    best_result = next(
        (r for r in all_results if abs(r["pr_auc"] - best_trial.value) < 1e-8),
        all_results[-1],
    )
    best_combo.update(
        {
            "pr_auc": best_result["pr_auc"],
            "roc_auc": best_result["roc_auc"],
            "mcc": best_result["mcc"],
            "balanced_acc": best_result["balanced_acc"],
            "best_epoch": best_result["best_epoch"],
            "threshold": best_result["threshold"],
        }
    )

    # Save results as JSON.
    search_space_ser = {}
    for k, v in search_space.items():
        if isinstance(v, (list, tuple)):
            search_space_ser[k] = [str(x) for x in v]

    output = {
        "seed": seed,
        "n_trials": n_trials,
        "search_space": search_space_ser,
        "best": best_combo,
        "all_results": sorted(all_results, key=lambda r: r["pr_auc"], reverse=True),
    }

    results_path = out_dir / "tuning_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=" * 60)
    logger.info("  TUNING COMPLETE")
    logger.info("=" * 60)
    logger.info("  Best PR-AUC: %.4f", best_trial.value)
    logger.info("  Best config: %s", best_combo)
    logger.info("  Results saved to %s", results_path)
    logger.info("=" * 60)

    return output


def load_best_params(tuning_dir: Path | None = None) -> Dict:
    """Load the best hyperparameters from a previous tuning run.

    Args:
        tuning_dir(Path | None): directory containing ``tuning_results.json``.

    Returns:
        Dict of best hyperparameters.
    """
    if tuning_dir is None:
        tuning_dir = GNNConfig().tuning_outdir
    path = tuning_dir / "tuning_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No tuning results at {path}. Run tuning first: "
            "uv run python -m src.gnn.tuning"
        )
    with open(path) as f:
        data = json.load(f)
    return data["best"]


def apply_tuned_params(config: GNNConfig, params: Dict) -> GNNConfig:
    """Apply tuned hyperparameters to a ``GNNConfig`` in-place.

    Args:
        config(GNNConfig): configuration to modify.
        params(Dict): parameter dict from :func:`load_best_params`.

    Returns:
        The modified configuration (same object).
    """
    for k, v in params.items():
        if hasattr(config.training, k):
            setattr(config.training, k, v)
        if hasattr(config.model, k):
            setattr(config.model, k, v)
    return config


# CLI entry point.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GNN hyperparameter tuning (Optuna TPE)"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()

    tune(
        device=args.device,
        seed=args.seed,
        max_epochs=args.max_epochs,
        n_trials=args.n_trials,
    )
