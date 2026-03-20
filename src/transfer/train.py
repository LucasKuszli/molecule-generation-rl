"""Transfer learning: fine-tune pre-trained SMILES LSTM on HIV actives.

Usage::

    uv run python -m src.transfer.train
    uv run python -m src.transfer.train --seed 123
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.training import TransferLearningConfig
from src.data.loader import get_transfer_dataset
from src.pretrain.model import SmilesLSTM
from src.pretrain.train import (
    SmilesDataset,
    collate_fn,
    evaluate,
    rdkit_valid_fraction,
)
from src.pretrain.vocabulary import SMILESTokenizer, Vocabulary

logger = logging.getLogger(__name__)


def train(config: TransferLearningConfig | None = None) -> Path:
    """Fine-tune a pre-trained LSTM on HIV actives.

    Args:
        config(TransferLearningConfig | None): training configuration;
            uses defaults when *None*.

    Returns:
        Path to the best checkpoint file.
    """

    if config is None:
        config = TransferLearningConfig()
    cfg_t = config.training

    # Set seeds and logging.
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seed: %d", seed)

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load pre-trained checkpoint.
    pretrain_path = Path(config.checkpoint_indir) / "best_prior.pt"
    logger.info("Loading pre-trained model from %s", pretrain_path)
    if not pretrain_path.exists():
        raise FileNotFoundError(
            f"No pre-trained checkpoint at {pretrain_path}. "
            "Run pre-training first: uv run python -m src.pretrain.train"
        )

    ckpt = torch.load(pretrain_path, map_location=device, weights_only=False)
    vocab = Vocabulary(ckpt["vocab"])
    model = SmilesLSTM(**ckpt["model_params"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    tokenizer = SMILESTokenizer()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Vocabulary: %d tokens  |  Parameters: %s", len(vocab), f"{n_params:,}")

    # Load HIV actives dataset.
    logger.info("Loading HIV actives for transfer learning …")
    data = get_transfer_dataset()
    train_smiles = data.train_set["SMILES"].tolist()
    val_smiles = data.val_set["SMILES"].tolist()

    train_ds = SmilesDataset(train_smiles, vocab, tokenizer)
    val_ds = SmilesDataset(val_smiles, vocab, tokenizer)
    logger.info(
        "Train sequences: %s  |  Val sequences: %s",
        f"{len(train_ds):,}",
        f"{len(val_ds):,}",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_t.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_t.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # AdamW optimizer with validation-based LR scheduler and early stopping.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_t.learning_rate, weight_decay=cfg_t.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    ckpt_dir = Path(config.checkpoint_outdir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_transfer.pt"

    best_val_nll = float("inf")
    patience_counter = 0
    history = []

    # Training loop with validation and checkpointing.
    for epoch in tqdm(range(1, cfg_t.num_epochs + 1)):
        model.train()
        epoch_nll = 0.0
        epoch_seqs = 0
        t0 = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            nll = model.nll(batch, pad_id=vocab.pad_id)
            loss = nll.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg_t.grad_clip)
            optimizer.step()

            epoch_nll += nll.sum().item()
            epoch_seqs += batch.size(0)

        train_nll = epoch_nll / max(epoch_seqs, 1)
        val_nll = evaluate(model, val_loader, vocab.pad_id, device)
        scheduler.step(val_nll)

        sampled, _ = model.sample(vocab, tokenizer, n=config.sample_n, device=device)
        validity = rdkit_valid_fraction(sampled)
        unique = len(set(sampled)) / max(len(sampled), 1)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %3d/%d | train NLL %7.2f | val NLL %7.2f | "
            "validity %.1f%% | unique %.1f%% | lr %.1e | %.0fs",
            epoch,
            cfg_t.num_epochs,
            train_nll,
            val_nll,
            validity * 100,
            unique * 100,
            lr,
            elapsed,
        )

        history.append(
            {
                "epoch": epoch,
                "train_nll": round(train_nll, 4),
                "val_nll": round(val_nll, 4),
                "validity": round(validity, 4),
                "uniqueness": round(unique, 4),
                "lr": lr,
                "elapsed_s": round(elapsed, 1),
            }
        )

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_params": model.get_params(),
                    "vocab": vocab._token_to_id,
                    "epoch": epoch,
                    "val_nll": val_nll,
                    "validity": validity,
                },
                best_path,
            )
            logger.info("Saved best checkpoint (val NLL %.2f)", val_nll)
        else:
            patience_counter += 1
            if patience_counter >= cfg_t.patience:
                logger.info(
                    "Early stopping after %d epochs without improvement",
                    cfg_t.patience,
                )
                break

    logger.info("Transfer learning complete. Best val NLL: %.2f", best_val_nll)
    logger.info("Checkpoint: %s", best_path)

    # Save training metadata and history.
    meta = {
        "seed": seed,
        "pretrain_checkpoint": str(pretrain_path),
        "best_val_nll": round(best_val_nll, 4),
        "total_epochs": len(history),
        "n_params": n_params,
        "vocab_size": len(vocab),
        "train_sequences": len(train_ds),
        "val_sequences": len(val_ds),
        "device": str(device),
        "config": {
            "batch_size": cfg_t.batch_size,
            "learning_rate": cfg_t.learning_rate,
            "weight_decay": cfg_t.weight_decay,
            "grad_clip": cfg_t.grad_clip,
            "patience": cfg_t.patience,
            "num_epochs": cfg_t.num_epochs,
        },
        "history": history,
    }
    meta_path = ckpt_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Training metadata saved to %s", meta_path)

    return best_path


# Run the training function if this script is executed directly.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transfer-learn SMILES LSTM on HIV actives"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(TransferLearningConfig(seed=args.seed))
