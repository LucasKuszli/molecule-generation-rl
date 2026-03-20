"""Inference wrapper for the trained GNN HIV activity predictor."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch_geometric.data import Batch

from src.gnn.dataset import mol_to_graph
from src.gnn.model import HIVPredictor

logger = logging.getLogger(__name__)


class HIVActivityPredictor:
    """Load a trained GNN checkpoint and score SMILES strings.

    Returns (label, probability) per molecule.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        """Load a trained GNN from a checkpoint.

        Args:
            checkpoint_path(str | Path): path to the ``.pt`` checkpoint.
            device(str): torch device string.
        """
        self.device = device
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        in_dim = checkpoint.get("in_dim")
        model_params = checkpoint.get("model_params", {})
        if in_dim is None:
            for tensor in checkpoint["model_state"].values():
                if hasattr(tensor, "shape") and len(tensor.shape) >= 2:
                    in_dim = tensor.shape[1]
                    break
        if in_dim is None:
            raise ValueError("Unable to infer input dimension from checkpoint")

        self.model = HIVPredictor(
            in_channels=in_dim,
            edge_dim=model_params.get("edge_dim", 3),
            hidden_dim=model_params.get("hidden_dim", 128),
            num_layers=model_params.get("num_layers", 3),
            dropout=model_params.get("dropout", 0.2),
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(device)
        self.model.eval()

        self.threshold = checkpoint.get("best_threshold", 0.5)

    @torch.no_grad()
    def predict(self, smiles: str):
        """Score a single SMILES string.

        Args:
            smiles(str): input SMILES.

        Returns:
            Tuple of (label, probability) or (None, None) when parsing fails.
        """
        graph = mol_to_graph(smiles)
        if graph is None:
            return None, None

        batch = Batch.from_data_list([graph]).to(self.device)
        logits = self.model(batch)
        prob = torch.sigmoid(logits).item()
        label = int(prob >= self.threshold)
        return label, prob
