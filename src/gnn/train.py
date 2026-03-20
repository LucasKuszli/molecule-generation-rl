"""GNN training loop for HIV activity prediction.

Follows the same conventions as ``src.pretrain.train`` and
``src.transfer.train``:

* Accepts a :class:`~src.config.training.GNNConfig`.
* Seeds all RNGs for reproducibility.
* Saves best checkpoint + ``meta.json`` with training history.
* Early stopping on PR-AUC.

Usage::

    uv run python -m src.gnn.train
    uv run python -m src.gnn.train --seed 67
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import RDLogger
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader as GeoLoader

from src.config.training import GNNConfig
from src.data.loader import get_gnn_datasplit
from src.gnn.dataset import HIVGraphDataset
from src.gnn.model import HIVPredictor

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


def _build_loader(
    smiles: List[str],
    labels: List[float],
    batch_size: int,
    shuffle: bool = False,
) -> GeoLoader:
    """Wrap SMILES and labels in a ``HIVGraphDataset`` and ``DataLoader``.

    Args:
        smiles(List[str]): SMILES strings.
        labels(List[float]): binary activity labels.
        batch_size(int): mini-batch size.
        shuffle(bool): whether to shuffle the loader.

    Returns:
        PyG DataLoader.
    """
    ds = HIVGraphDataset(smiles, labels)
    return GeoLoader(ds, batch_size=batch_size, shuffle=shuffle)


def prepare_dataloaders(
    config: GNNConfig,
) -> Tuple[GeoLoader, GeoLoader, GeoLoader]:
    """Load GNN train/val/test splits and wrap in geometric DataLoaders.

    When ``config.training.inactive_ratio > 0`` the training split is
    undersampled to the specified active-to-inactive ratio.

    Args:
        config(GNNConfig): GNN configuration object.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data = get_gnn_datasplit()
    bs = config.training.batch_size

    train_df = data.train_set

    # Undersample inactives in training set
    ratio = config.training.inactive_ratio
    if ratio > 0:
        actives = train_df[train_df["label"] == 1]
        inactives = train_df[train_df["label"] == 0]
        max_inactives = int(len(actives) * ratio)
        if len(inactives) > max_inactives:
            inactives = inactives.sample(n=max_inactives, random_state=config.seed)
        train_df = pd.concat([actives, inactives]).sample(
            frac=1, random_state=config.seed
        )
        logger.info(
            "  Undersampled training set: %d actives + %d inactives (ratio 1:%.0f)",
            len(actives),
            len(inactives),
            ratio,
        )

    train_loader = _build_loader(
        train_df["SMILES"].tolist(),
        train_df["label"].astype(float).tolist(),
        batch_size=bs,
        shuffle=True,
    )
    val_loader = _build_loader(
        data.val_set["SMILES"].tolist(),
        data.val_set["label"].astype(float).tolist(),
        batch_size=bs,
    )
    test_loader = _build_loader(
        data.test_set["SMILES"].tolist(),
        data.test_set["label"].astype(float).tolist(),
        batch_size=bs,
    )
    return train_loader, val_loader, test_loader


def find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Sweep thresholds and return the one maximising F1.

    Args:
        labels(np.ndarray): ground-truth binary labels.
        probs(np.ndarray): predicted probabilities.

    Returns:
        Optimal classification threshold.
    """
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.01, 0.99, 200):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def train_loop(
    train_loader: GeoLoader,
    val_loader: GeoLoader,
    model: HIVPredictor,
    config: GNNConfig,
    device: str,
) -> Tuple[HIVPredictor, float, List[Dict]]:
    """Train the GNN with early stopping on PR-AUC.

    Args:
        train_loader(GeoLoader): training data loader.
        val_loader(GeoLoader): validation data loader.
        model(HIVPredictor): uninitialised or pre-initialised model.
        config(GNNConfig): full GNN configuration.
        device(str): torch device string.

    Returns:
        Tuple of (trained model, best threshold, training history).
    """

    # Class weighting.
    n_pos = sum(1 for d in train_loader.dataset if d.y.item() == 1)
    n_neg = len(train_loader.dataset) - n_pos
    if n_pos == 0:
        raise ValueError("No positive samples in training set.")
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(
        "  pos_weight = %.2f  (%d pos / %d neg)", pos_weight.item(), n_pos, n_neg
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    best_pr_auc = -1.0
    best_threshold = 0.5
    patience_counter = 0
    best_state = None
    history: List[Dict] = []

    for epoch in range(1, config.training.num_epochs + 1):
        # Training loop.
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch).view(-1)
            y = batch.y.view(-1).float()
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # Evaluate on validation set and compute metrics.
        model.eval()
        val_loss = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch).view(-1)
                y = batch.y.view(-1).float()
                val_loss += criterion(logits, y).item() * y.size(0)
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        avg_val_loss = val_loss / len(val_loader.dataset)
        logits_np = torch.cat(all_logits).numpy()
        labels_np = torch.cat(all_labels).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits_np))

        pr_auc = average_precision_score(labels_np, probs)
        roc_auc = roc_auc_score(labels_np, probs)
        current_threshold = find_best_threshold(labels_np, probs)
        preds = (probs >= current_threshold).astype(int)
        mcc = matthews_corrcoef(labels_np, preds)
        bal_acc = balanced_accuracy_score(labels_np, preds)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(avg_val_loss, 6),
                "pr_auc": round(pr_auc, 6),
                "roc_auc": round(roc_auc, 6),
                "mcc": round(mcc, 6),
                "balanced_acc": round(bal_acc, 6),
                "threshold": round(current_threshold, 4),
            }
        )

        logger.info(
            "  Epoch %3d  train_loss=%.4f  val_loss=%.4f  "
            "PR-AUC=%.4f  ROC-AUC=%.4f  MCC=%.4f  thr=%.3f",
            epoch,
            avg_train_loss,
            avg_val_loss,
            pr_auc,
            roc_auc,
            mcc,
            current_threshold,
        )

        # Early stopping on PR-AUC.
        if pr_auc > best_pr_auc + 1e-4:
            best_pr_auc = pr_auc
            best_threshold = current_threshold
            patience_counter = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if patience_counter >= config.training.patience:
            logger.info(
                "  Early stopping at epoch %d (no PR-AUC improvement for %d epochs)",
                epoch,
                config.training.patience,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(
        "Best PR-AUC: %.4f  |  Best threshold: %.3f", best_pr_auc, best_threshold
    )
    return model, best_threshold, history


# Public entry point.
def train(
    config: GNNConfig,
    device: Optional[str] = None,
) -> Path:
    """Train a GNN on HIV activity and save checkpoint + metadata.

    Returns the path to the saved checkpoint.
    """
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 60)
    logger.info("  GNN TRAINING - seed %d  (device: %s)", seed, device)
    logger.info("=" * 60)

    # data
    train_loader, val_loader, _ = prepare_dataloaders(config)
    in_dim = next(iter(train_loader)).x.size(-1)
    logger.info(
        "  Feature dim: %d  |  Train: %d  Val: %d",
        in_dim,
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    # model
    model = HIVPredictor(
        in_channels=in_dim,
        edge_dim=config.model.edge_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  Parameters: %s", f"{n_params:,}")

    # train
    model, best_threshold, history = train_loop(
        train_loader, val_loader, model, config, device
    )

    # save checkpoint
    out_dir = config.checkpoint_outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_gnn.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "best_threshold": best_threshold,
            "in_dim": in_dim,
            "model_params": {
                "in_channels": in_dim,
                "edge_dim": config.model.edge_dim,
                "hidden_dim": config.model.hidden_dim,
                "num_layers": config.model.num_layers,
                "dropout": config.model.dropout,
            },
        },
        ckpt_path,
    )
    logger.info("Checkpoint saved to %s", ckpt_path)

    # save meta.json
    meta = {
        "seed": seed,
        "device": device,
        "best_pr_auc": history[-1]["pr_auc"] if history else None,
        "best_threshold": best_threshold,
        "config": {
            "training": {
                k: getattr(config.training, k) for k in vars(GNNConfig().training)
            },
            "model": {k: getattr(config.model, k) for k in vars(GNNConfig().model)},
        },
        "history": history,
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    return ckpt_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train GNN for HIV activity prediction"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(GNNConfig(seed=args.seed), device=args.device)
