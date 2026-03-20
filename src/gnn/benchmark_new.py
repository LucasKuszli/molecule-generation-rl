"""Multi-seed GNN benchmark with a Random Forest baseline.

Trains the GNN for several seeds, evaluates on the validation set, and
compares against a simple RF baseline using averaged atom features.

.. note::

    The ``__main__`` block at the bottom still contains hardcoded Windows
    paths from early development.  Use the pipeline runners in
    ``src.pipeline`` for reproducible runs.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader as GeoLoader

from src.config.training import GNNConfig
from src.gnn.dataset import HIVGraphDataset, mol_to_graph
from src.gnn.model import HIVPredictor
from src.gnn.train import train as train_pipeline

logger = logging.getLogger(__name__)


def set_seed(seed):
    """Seed all random number generators for reproducibility.

    Args:
        seed(int): seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, probs):
    """Compute classification metrics at a fixed 0.5 threshold.

    Args:
        y_true: ground-truth binary labels.
        probs: predicted probabilities.

    Returns:
        Dict of PR-AUC, ROC-AUC, precision, balanced accuracy, MCC.
    """
    preds = (probs >= 0.5).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, probs),
        "roc_auc": roc_auc_score(y_true, probs),
        "precision": precision_score(y_true, preds, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, preds),
        "mcc": matthews_corrcoef(y_true, preds),
    }


def load_model_from_checkpoint(ckpt_path, device):
    """Load an ``HIVPredictor`` from a checkpoint file.

    Args:
        ckpt_path: path to the ``.pt`` file.
        device: torch device string.

    Returns:
        Loaded model in eval mode.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    model = HIVPredictor(
        in_channels=ckpt["in_dim"],
        edge_dim=ckpt["model_params"]["edge_dim"],
        hidden_dim=ckpt["model_params"]["hidden_dim"],
        num_layers=ckpt["model_params"]["num_layers"],
        dropout=ckpt["model_params"]["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model


def predict_gnn(model, graphs, device):
    """Run GNN inference on a list of graph objects.

    Args:
        model: trained ``HIVPredictor``.
        graphs: list of PyG Data objects.
        device: torch device string.

    Returns:
        Tuple of (y_true, probs) NumPy arrays.
    """
    loader = GeoLoader(graphs, batch_size=32)

    all_logits = []
    all_y = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch).view(-1)
            all_logits.append(logits.cpu())
            all_y.append(batch.y.cpu())

    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_y).numpy()
    probs = 1 / (1 + np.exp(-logits))

    return y_true, probs


def smiles_to_features(smiles_list):
    """Convert SMILES to averaged atom features for the RF baseline.

    Args:
        smiles_list: iterable of SMILES strings.

    Returns:
        NumPy array of shape ``(n_molecules, 6)``.
    """
    feats = []
    for s in smiles_list:
        g = mol_to_graph(s)
        if g is None:
            feats.append(np.zeros(6))
        else:
            feats.append(g.x.mean(dim=0).numpy())
    return np.array(feats)


def run_experiment(train_df, val_df, seeds=(42, 52, 62)):
    """Run multi-seed GNN + RF benchmark.

    Args:
        train_df: training DataFrame with *SMILES* and *label* columns.
        val_df: validation DataFrame.
        seeds: tuple of random seeds for each run.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path("./benchmark")
    base_dir.mkdir(exist_ok=True)

    summary_rows = []

    s_train = train_df["SMILES"].values
    y_train = train_df["label"].values

    s_val = val_df["SMILES"].values
    y_val = val_df["label"].values

    for run_idx, seed in enumerate(seeds, start=1):
        logger.info("===== RUN %d | SEED %d =====", run_idx, seed)
        set_seed(seed)

        run_dir = base_dir / f"run_{run_idx}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Training.
        config = GNNConfig(seed=seed)
        config.checkpoint_outdir = run_dir / "checkpoints"

        ckpt_path = train_pipeline(config, device=device)

        # Load training history.
        meta_path = config.checkpoint_outdir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        history = meta["history"]
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(run_dir / "gnn_history.csv", index=False)

        best_epoch = max(history, key=lambda x: x["pr_auc"])

        # GNN Evaluation.
        val_graphs = [HIVGraphDataset([s], [y])[0] for s, y in zip(s_val, y_val)]

        gnn_model = load_model_from_checkpoint(ckpt_path, device)
        y_true, probs = predict_gnn(gnn_model, val_graphs, device)

        gnn_metrics = compute_metrics(y_true, probs)

        logger.info("GNN: %s", gnn_metrics)

        # RF Baseline.
        X_train = smiles_to_features(s_train)
        X_val = smiles_to_features(s_val)

        rf = RandomForestClassifier(n_estimators=200, random_state=seed)
        rf.fit(X_train, y_train)

        probs = rf.predict_proba(X_val)[:, 1]
        rf_metrics = compute_metrics(y_val, probs)

        logger.info("RF: %s", rf_metrics)

        # Save metrics as CSV.
        row = {
            "seed": seed,
            # Best training metrics
            "gnn_best_pr_auc": best_epoch["pr_auc"],
            "gnn_best_roc_auc": best_epoch["roc_auc"],
            "gnn_best_epoch": best_epoch["epoch"],
            # Validation metrics
            **{f"gnn_{k}": v for k, v in gnn_metrics.items()},
            **{f"rf_{k}": v for k, v in rf_metrics.items()},
        }

        pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)

        # Save config.
        with open(run_dir / "config.json", "w") as f:
            json.dump(
                {
                    "training": vars(config.training),
                    "model": vars(config.model),
                    "seed": seed,
                },
                f,
                indent=2,
            )

        summary_rows.append(row)

    # Summary statistics across runs.
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(base_dir / "summary.csv", index=False)
    summary_df.describe().to_csv(base_dir / "stats.csv")

    logger.info("Benchmark complete. Results saved to: %s", base_dir)


# Entry point for standalone runs.  For reproducible experiments, use the
# pipeline runners in src.pipeline for reproducible runs.
if __name__ == "__main__":
    train_df = pd.read_csv(
        "C:\\Users\\alyck\\Downloads\\molecules_and_ai\\data\\hiv_train.csv"
    )
    val_df = pd.read_csv(
        "C:\\Users\\alyck\\Downloads\\molecules_and_ai\\data\\hiv_val.csv"
    )

    run_experiment(train_df, val_df)
