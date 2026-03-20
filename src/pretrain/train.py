"""Pre-training loop for the SMILES LSTM language model.

Usage::

    uv run python -m src.pretrain.train
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config.training import PretrainingConfig
from src.data.loader import get_pretrain_datasplit
from src.pretrain.model import SmilesLSTM
from src.pretrain.vocabulary import SMILESTokenizer, Vocabulary, build_vocabulary

logger = logging.getLogger(__name__)


class SmilesDataset(Dataset):
    """Encodes SMILES into padded token-id sequences for LM training.

    Args:
        smiles_list(List[str]): raw SMILES strings.
        vocab(Vocabulary): token-to-id mapping.
        tokenizer(SMILESTokenizer): SMILES tokenizer.
        max_len(int): maximum allowed sequence length (longer are dropped).
    """

    def __init__(
        self,
        smiles_list: List[str],
        vocab: Vocabulary,
        tokenizer: SMILESTokenizer,
        max_len: int = 140,
    ):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encoded = self._encode_all(smiles_list)

    def _encode_all(self, smiles_list: List[str]) -> List[List[int]]:
        """Tokenize and encode every SMILES, discarding those exceeding *max_len*."""
        encoded = []
        for smi in smiles_list:
            tokens = self.tokenizer.tokenize(smi)  # includes ^ and $
            ids = self.vocab.encode(tokens)
            if len(ids) <= self.max_len:
                encoded.append(ids)
        return encoded

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.encoded[idx], dtype=torch.long)


def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    """Right-pad sequences with zeros to equal length within a batch.

    Args:
        batch(List[torch.Tensor]): variable-length encoded sequences.

    Returns:
        Padded (batch_size, max_len) long tensor.
    """
    max_len = max(seq.size(0) for seq in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, : seq.size(0)] = seq
    return padded


def rdkit_valid_fraction(smiles_list: List[str]) -> float:
    """Compute the fraction of SMILES parseable by RDKit.

    Args:
        smiles_list(List[str]): generated SMILES to validate.

    Returns:
        Fraction in [0, 1].
    """
    if not smiles_list:
        return 0.0
    valid = sum(1 for s in smiles_list if s and Chem.MolFromSmiles(s) is not None)
    return valid / len(smiles_list)


@torch.no_grad()
def evaluate(
    model: SmilesLSTM,
    dataloader: DataLoader,
    pad_id: int,
    device: torch.device,
) -> float:
    """Compute mean NLL per sequence on a validation set.

    Args:
        model(SmilesLSTM): language model to evaluate.
        dataloader(DataLoader): validation data loader.
        pad_id(int): padding token id to ignore in NLL.
        device(torch.device): computation device.

    Returns:
        Mean negative log-likelihood.
    """
    model.eval()
    total_nll = 0.0
    total_seqs = 0
    for batch in dataloader:
        batch = batch.to(device)
        nll = model.nll(batch, pad_id=pad_id)
        total_nll += nll.sum().item()
        total_seqs += batch.size(0)
    return total_nll / max(total_seqs, 1)


def train(config: PretrainingConfig | None = None) -> Path:
    """Run the full pre-training loop.

    Args:
        config(PretrainingConfig | None): training configuration; uses defaults
            when *None*.

    Returns:
        Path to the best checkpoint file.
    """

    if config is None:
        config = PretrainingConfig()
    cfg_t = config.training
    cfg_m = config.model

    # Set reproducible random seeds and deterministic CuDNN for consistent results.
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seed: %d", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading data …")
    data = get_pretrain_datasplit()
    train_smiles = data.train_set["SMILES"].tolist()
    val_smiles = data.val_set["SMILES"].tolist()

    tokenizer = SMILESTokenizer()
    vocab_path = Path(config.vocab_path)
    if vocab_path.exists():
        logger.info("Loading vocabulary from %s", vocab_path)
        vocab = Vocabulary.load(vocab_path)
    else:
        logger.info("Building vocabulary from training data …")
        vocab = build_vocabulary(train_smiles, tokenizer)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(vocab_path)
    logger.info("Vocabulary: %d tokens", len(vocab))

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
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_t.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    model = SmilesLSTM(
        vocab_size=len(vocab),
        embedding_dim=cfg_m.embedding_size,
        hidden_dim=cfg_m.hidden_size,
        num_layers=cfg_m.num_layers,
        dropout=cfg_m.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_t.learning_rate, weight_decay=cfg_t.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    ckpt_dir = Path(config.checkpoint_outdir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_prior.pt"

    best_val_nll = float("inf")
    patience_counter = 0
    history = []

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
                    "  Early stopping after %d epochs without improvement",
                    cfg_t.patience,
                )
                break

    logger.info("Training complete. Best val NLL: %.2f", best_val_nll)
    logger.info("Checkpoint: %s", best_path)

    # Save training metadata and history to JSON for later analysis.
    meta = {
        "seed": seed,
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
            "embedding_size": cfg_m.embedding_size,
            "hidden_size": cfg_m.hidden_size,
            "num_layers": cfg_m.num_layers,
            "dropout": cfg_m.dropout,
        },
        "history": history,
    }
    meta_path = ckpt_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Training metadata saved to %s", meta_path)

    return best_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-train SMILES LSTM")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(PretrainingConfig(seed=args.seed))
