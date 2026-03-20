"""REINFORCE training for SMILES LSTM with predicted HIV activity as reward.

Loads a pre-trained (or transfer-learned) SMILES LSTM checkpoint and
fine-tunes it with REINFORCE, using a **multi-objective composite reward**
(HIV activity + validity + SA) and a **prior KL penalty** to prevent
mode collapse.

Usage::

    uv run python -m src.reinforce.train
    uv run python -m src.reinforce.train --stage transfer_rl
    uv run python -m src.reinforce.train --stage pretrain_rl --seed 67
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors

from src.config.training import RLConfig
from src.pretrain.evaluation import load_hiv_predictor, load_model_from_checkpoint
from src.pretrain.model import SmilesLSTM
from src.pretrain.vocabulary import SMILESTokenizer, Vocabulary

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


def _sa_score(mol) -> float:
    """Approximate synthetic accessibility score (1 = easy, 10 = hard).

    Attempts to use the ``sascorer`` package first. Falls back to a
    lightweight heuristic based on ring count, stereo centres and MW.

    Args:
        mol: RDKit molecule object.

    Returns:
        float: Score in the range [1, 10].
    """
    try:
        from sascorer import calculateScore

        return calculateScore(mol)
    except ImportError:
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        mw = Descriptors.MolWt(mol)
        score = 1 + 0.5 * num_rings + num_stereo + 0.005 * mw
        return min(10.0, max(1.0, score))


def compute_composite_reward(
    smiles_list: List[str],
    predictor,
    w_activity: float = 0.6,
    w_validity: float = 0.2,
    w_sa: float = 0.2,
) -> np.ndarray:
    """Multi-objective reward for each SMILES string.

    For each molecule the reward is:
        w_activity * P(active)   (GNN prediction, 0 for invalid)
      + w_validity * 1.0         (if parseable by RDKit, else 0)
      + w_sa       * (1 - SA/10) (normalised so 1 = easy, 0 = hard)

    Invalid SMILES receive reward 0.
    """
    rewards = np.zeros(len(smiles_list), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # reward stays 0

        # Validity reward is binary (1 for valid, 0 for invalid).
        r_valid = 1.0

        # --- SA component (normalised to [0, 1]) ---
        try:
            sa = _sa_score(mol)
        except Exception:
            sa = 5.0  # neutral fallback
        r_sa = 1.0 - sa / 10.0

        # GNN activity component: P(active) in [0, 1], or 0 if predictor fails.
        r_activity = 0.0
        try:
            label, prob = predictor.predict(smi)
            if prob is not None:
                r_activity = float(prob)
        except Exception:
            pass

        rewards[i] = w_activity * r_activity + w_validity * r_valid + w_sa * r_sa
    return rewards


# Reinforcement learning step: sample, reward, compute loss, return stats.
def _encode_smiles(
    smiles_list: List[str],
    vocab: Vocabulary,
    tokenizer: SMILESTokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Tokenize and pad a list of SMILES into a ``(batch, max_len)`` tensor.

    Args:
        smiles_list(list[str]): Raw SMILES strings.
        vocab(Vocabulary): Token-to-id mapping.
        tokenizer(SMILESTokenizer): Character-level tokenizer.
        device(torch.device): Target device for the tensor.

    Returns:
        torch.Tensor: Padded integer tensor of shape ``(B, T)``.
    """
    encoded = []
    for smi in smiles_list:
        tokens = [vocab.START] + tokenizer.tokenize(smi) + [vocab.END]
        ids = vocab.encode(tokens)
        encoded.append(ids)
    max_len = max(len(seq) for seq in encoded)
    padded = [seq + [vocab.pad_id] * (max_len - len(seq)) for seq in encoded]
    return torch.tensor(padded, dtype=torch.long, device=device)


def reinforce_step(
    model: SmilesLSTM,
    prior: SmilesLSTM,
    vocab: Vocabulary,
    predictor,
    batch_size: int,
    sigma: float,
    prior_weight: float,
    w_activity: float,
    w_validity: float,
    w_sa: float,
    device: torch.device,
) -> Tuple[float, Dict]:
    """Sample a batch, compute multi-objective rewards, return loss + stats.

    The loss combines two terms:
      1. REINFORCE:  E[NLL_agent * advantage] / sigma   (minimised -> maximise reward)
      2. Prior KL:   prior_weight * max(0, KL(agent || prior))   (regulariser, >= 0)
    """
    tokenizer = SMILESTokenizer()

    # Sample (no grad, policy rollout)
    smiles, _ = model.sample(vocab, tokenizer, n=batch_size, device=device)

    # Multi-objective reward
    rewards = compute_composite_reward(
        smiles,
        predictor,
        w_activity=w_activity,
        w_validity=w_validity,
        w_sa=w_sa,
    )
    rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)

    # Validity & uniqueness.
    valid_smiles = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid_smiles.append(Chem.MolToSmiles(mol))
    n_valid = len(valid_smiles)
    unique_smiles = set(valid_smiles)
    n_unique = len(unique_smiles)
    uniqueness = n_unique / n_valid if n_valid > 0 else 0.0

    # Fast diversity estimate (Tanimoto on a subsample)
    diversity = 0.0
    if n_unique >= 2:
        sample_smi = list(unique_smiles)[:200]
        fps = []
        for s in sample_smi:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
        if len(fps) >= 2:
            dists = []
            n_fp = min(100, len(fps))
            for i in range(n_fp):
                for j in range(i + 1, n_fp):
                    dists.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            diversity = float(np.mean(dists))

    # Re-encode sampled SMILES and compute NLLs with gradients.
    model.train()
    sequences = _encode_smiles(smiles, vocab, tokenizer, device)
    agent_nlls = model.nll(sequences, pad_id=vocab.pad_id)  # (batch,) with grad_fn

    # Prior NLLs (frozen, no grad)
    with torch.no_grad():
        prior_nlls = prior.nll(sequences, pad_id=vocab.pad_id)  # (batch,)

    #    REINFORCE loss with prior KL penalty
    #    REINFORCE: minimise E[advantage * NLL] / sigma
    #    When advantage > 0 (good molecule): gradient pushes NLL down -> more likely
    #    When advantage < 0 (bad molecule):  gradient pushes NLL up   -> less likely
    baseline = rewards_tensor.mean()
    advantage = rewards_tensor - baseline
    reinforce_loss = (agent_nlls * advantage).mean() / sigma

    #    Prior KL penalty: keeps agent close to pre-trained distribution
    #    KL(agent || prior) = E_agent[log agent - log prior]
    #                       = E_agent[NLL_prior - NLL_agent]   (>= 0)
    #    Clamped per sample to >= 0 before averaging.
    kl_penalty = torch.clamp(prior_nlls - agent_nlls, min=0.0).mean()
    loss = reinforce_loss + prior_weight * kl_penalty

    stats = {
        "mean_reward": float(rewards_tensor.mean()),
        "max_reward": float(rewards_tensor.max()),
        "std_reward": float(rewards_tensor.std()),
        "validity": n_valid / batch_size,
        "uniqueness": uniqueness,
        "diversity": diversity,
        "mean_nll": float(agent_nlls.mean().item()),
        "reinforce_loss": float(reinforce_loss.item()),
        "kl_penalty": float(kl_penalty.item()),
    }
    return loss, stats


def train_loop(
    model: SmilesLSTM,
    prior: SmilesLSTM,
    vocab: Vocabulary,
    predictor,
    config: RLConfig,
    device: torch.device,
) -> Tuple[SmilesLSTM, List[Dict]]:
    """Run REINFORCE training loop, return (best_model, history).

    Args:
        model(SmilesLSTM): Agent model (updated in place).
        prior(SmilesLSTM): Frozen prior for KL regularisation.
        vocab(Vocabulary): Shared vocabulary.
        predictor: GNN HIV-activity predictor.
        config(RLConfig): Hyper-parameter container.
        device(torch.device): Compute device.

    Returns:
        tuple[SmilesLSTM, list[dict]]: Best model and per-epoch history.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
    )

    history: List[Dict] = []
    best_mean_reward = -1.0
    best_state = None
    patience_counter = 0
    batch_size = config.training.batch_size
    sigma = config.training.sigma
    prior_weight = config.training.prior_weight
    w_activity = config.training.w_activity
    w_validity = config.training.w_validity
    w_sa = config.training.w_sa

    for epoch in range(1, config.training.num_epochs + 1):
        model.train()

        loss, stats = reinforce_step(
            model,
            prior,
            vocab,
            predictor,
            batch_size=batch_size,
            sigma=sigma,
            prior_weight=prior_weight,
            w_activity=w_activity,
            w_validity=w_validity,
            w_sa=w_sa,
            device=device,
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()

        record = {"epoch": epoch, "loss": round(float(loss), 6), **stats}
        history.append(record)

        logger.info(
            "Epoch %3d  loss=%.4f  mean_R=%.4f  max_R=%.4f  "
            "valid=%.2f%%  uniq=%.2f%%  div=%.3f  NLL=%.1f  KL=%.2f",
            epoch,
            float(loss),
            stats["mean_reward"],
            stats["max_reward"],
            stats["validity"] * 100,
            stats["uniqueness"] * 100,
            stats["diversity"],
            stats["mean_nll"],
            stats["kl_penalty"],
        )

        # Early stopping on mean reward.
        if stats["mean_reward"] > best_mean_reward + 1e-4:
            best_mean_reward = stats["mean_reward"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.training.patience:
            logger.info(
                "Early stopping at epoch %d (no reward improvement for %d epochs)",
                epoch,
                config.training.patience,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info("Best mean reward: %.4f", best_mean_reward)
    return model, history


# Public entry point.
def train(
    config: RLConfig,
    device: Optional[str] = None,
) -> Path:
    """REINFORCE fine-tune a SMILES LSTM and save checkpoint + metadata.

    Args:
        config(RLConfig): Full configuration including stage, seed, and
            hyper-parameters.
        device(str | None): Torch device string.  Defaults to CUDA when
            available, otherwise CPU.

    Returns:
        Path: Path to the saved best checkpoint.
    """
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    stage_name = config.stage.replace("_", " ").upper()
    logger.info("=" * 60)
    logger.info("%s TRAINING - seed %d  (device: %s)", stage_name, seed, device)
    logger.info("=" * 60)

    # Determine source checkpoint
    if config.stage == "pretrain_rl":
        ckpt_path = config.checkpoint_indir / "best_prior.pt"
    elif config.stage == "transfer_rl":
        ckpt_path = config.checkpoint_indir / "best_transfer.pt"
    else:
        raise ValueError(f"Unknown stage: {config.stage}")

    logger.info("Source checkpoint: %s", ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    # Load SMILES model.
    model, vocab = load_model_from_checkpoint(ckpt_path, device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Vocab: %d tokens | Params: %s", len(vocab), f"{n_params:,}")

    # Load frozen prior model for KL regularisation.
    logger.info("Prior checkpoint: %s", config.prior_checkpoint)
    if not config.prior_checkpoint.exists():
        raise FileNotFoundError(
            f"Prior checkpoint not found at {config.prior_checkpoint}. "
            "Train the base LSTM first."
        )
    prior, _ = load_model_from_checkpoint(config.prior_checkpoint, device)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad_(False)

    # Load GNN predictor (frozen, eval-only).
    logger.info("GNN checkpoint: %s", config.gnn_checkpoint)
    predictor = load_hiv_predictor(config.gnn_checkpoint, device=device)
    if predictor is None:
        raise FileNotFoundError(
            f"GNN checkpoint not found at {config.gnn_checkpoint}. Train the GNN first."
        )

    # Train
    model, history = train_loop(model, prior, vocab, predictor, config, device_t)

    # Save checkpoint
    out_dir = config.checkpoint_outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"best_{config.stage}.pt"
    ckpt_out = out_dir / ckpt_name
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_params": ckpt["model_params"],
            "vocab": ckpt["vocab"],
            "stage": config.stage,
            "seed": seed,
        },
        ckpt_out,
    )
    logger.info("Checkpoint saved to %s", ckpt_out)

    # Save meta.json
    meta = {
        "seed": seed,
        "stage": config.stage,
        "source_checkpoint": str(ckpt_path),
        "best_mean_reward": max(h["mean_reward"] for h in history) if history else 0.0,
        "training_params": {
            "num_epochs": config.training.num_epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "sigma": config.training.sigma,
            "prior_weight": config.training.prior_weight,
            "w_activity": config.training.w_activity,
            "w_validity": config.training.w_validity,
            "w_sa": config.training.w_sa,
        },
        "history": history,
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    return ckpt_out


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="REINFORCE fine-tuning of SMILES LSTM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--stage",
        type=str,
        default="pretrain_rl",
        choices=["pretrain_rl", "transfer_rl"],
    )
    args = parser.parse_args()

    config = RLConfig(seed=args.seed, stage=args.stage)
    train(config, device=args.device)
