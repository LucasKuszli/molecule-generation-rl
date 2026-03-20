"""GNN evaluation on the HIV test set.

Loads a trained GNN checkpoint, evaluates on the held-out test split, and
saves metrics + visualisations following the same conventions as the
pretrain / transfer evaluation modules.

Usage::

    uv run python -m src.gnn.evaluation
    uv run python -m src.gnn.evaluation --seed 67
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config.training import GNNConfig
from src.gnn.model import HIVPredictor
from src.gnn.train import prepare_dataloaders

logger = logging.getLogger(__name__)


def run_evaluation(
    checkpoint_path: Optional[Path] = None,
    device: str = "cpu",
    seed: int = 42,
) -> Dict:
    """Evaluate a trained GNN on the HIV test set.

    Args:
        checkpoint_path(Path | None): model checkpoint; inferred from seed
            when *None*.
        device(str): torch device string.
        seed(int): random seed (selects checkpoint directory).

    Returns:
        Dict of evaluation metrics.
    """

    config = GNNConfig(seed=seed)
    if checkpoint_path is None:
        checkpoint_path = config.checkpoint_outdir / "best_gnn.pt"

    logger.info("Checkpoint: %s", checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train first with: "
            "uv run python -m src.gnn.train"
        )

    # Load model for evaluation.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_params = ckpt.get("model_params", {})
    in_dim = ckpt.get("in_dim", model_params.get("in_channels"))
    best_threshold = ckpt.get("best_threshold", 0.5)

    model = HIVPredictor(
        in_channels=in_dim,
        edge_dim=model_params.get("edge_dim", config.model.edge_dim),
        hidden_dim=model_params.get("hidden_dim", config.model.hidden_dim),
        num_layers=model_params.get("num_layers", config.model.num_layers),
        dropout=model_params.get("dropout", config.model.dropout),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    logger.info(
        "  Params: %s  |  Threshold: %.3f",
        f"{sum(p.numel() for p in model.parameters()):,}",
        best_threshold,
    )

    # Prepare dataloaders.
    _, val_loader, test_loader = prepare_dataloaders(config)

    # Predict on test set.
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch).view(-1)
            all_logits.append(logits.cpu())
            all_labels.append(batch.y.view(-1).cpu())

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (probs >= best_threshold).astype(int)

    pr_auc = average_precision_score(labels_np, probs)
    roc_auc = roc_auc_score(labels_np, probs)
    mcc = matthews_corrcoef(labels_np, preds)
    bal_acc = balanced_accuracy_score(labels_np, preds)
    f1 = f1_score(labels_np, preds, zero_division=0)

    n_test = len(labels_np)
    n_pos = int(labels_np.sum())
    n_pred_pos = int(preds.sum())

    results = {
        "n_test": n_test,
        "n_positive": n_pos,
        "n_predicted_positive": n_pred_pos,
        "threshold": round(best_threshold, 4),
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc_auc, 6),
        "mcc": round(mcc, 6),
        "balanced_acc": round(bal_acc, 6),
        "f1": round(f1, 6),
    }

    logger.info("=" * 60)
    logger.info("GNN EVALUATION (test set)")
    logger.info("=" * 60)
    logger.info("  Test size:          %s  (%d positives)", f"{n_test:,}", n_pos)
    logger.info("  Predicted positive: %d", n_pred_pos)
    logger.info("  PR-AUC:             %.4f", pr_auc)
    logger.info("  ROC-AUC:            %.4f", roc_auc)
    logger.info("  MCC:                %.4f", mcc)
    logger.info("  Balanced Acc:       %.4f", bal_acc)
    logger.info("  F1:                 %.4f", f1)
    logger.info("=" * 60)

    # Plots.
    img_dir = config.img_outdir
    img_dir.mkdir(parents=True, exist_ok=True)

    # PR curve.
    precision, recall, _ = precision_recall_curve(labels_np, probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"GNN Precision-Recall Curve (seed {seed})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(img_dir / "gnn_pr_curve.png"), dpi=150)
    plt.close(fig)

    # ROC curve.
    fpr, tpr, _ = roc_curve(labels_np, probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"GNN ROC Curve (seed {seed})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(img_dir / "gnn_roc_curve.png"), dpi=150)
    plt.close(fig)

    # Training curves (from meta.json if available).
    meta_path = config.checkpoint_outdir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        history = meta.get("history", [])
        if history:
            epochs = [h["epoch"] for h in history]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(
                epochs,
                [h["train_loss"] for h in history],
                label="Train Loss",
                linewidth=2,
            )
            ax1.plot(
                epochs, [h["val_loss"] for h in history], label="Val Loss", linewidth=2
            )
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("GNN Training Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(
                epochs, [h["pr_auc"] for h in history], label="PR-AUC", linewidth=2
            )
            ax2.plot(
                epochs, [h["roc_auc"] for h in history], label="ROC-AUC", linewidth=2
            )
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Score")
            ax2.set_title("GNN Validation Metrics")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(str(img_dir / "gnn_training_curves.png"), dpi=150)
            plt.close(fig)
            logger.info(
                "Training curves saved to %s", img_dir / "gnn_training_curves.png"
            )

    # Save eval results as json.
    eval_dir = config.eval_outdir
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / "gnn_eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", eval_path)

    return results


# CLI.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained GNN on HIV test set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    run_evaluation(checkpoint_path=ckpt, device=args.device, seed=args.seed)
