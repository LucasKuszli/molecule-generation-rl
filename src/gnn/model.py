"""GINE-based graph neural network for HIV activity prediction."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

logger = logging.getLogger(__name__)


class HIVPredictor(torch.nn.Module):
    """Edge-aware graph classification model using GINE layers."""

    def __init__(
        self,
        in_channels: int,
        edge_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        """Initialise GINE model.

        Args:
            in_channels(int): number of atom-level input features.
            edge_dim(int): dimensionality of edge features.
            hidden_dim(int): hidden representation size.
            num_layers(int): number of GINE message-passing layers.
            dropout(float): dropout probability between layers.
        """
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)
        self.convs = torch.nn.ModuleList()

        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINEConv(nn1, edge_dim=edge_dim))

        for _ in range(num_layers - 1):
            nn_k = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn_k, edge_dim=edge_dim))

        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """Forward pass through GINE layers, global mean pool and classifier.

        Args:
            data: PyG ``Data`` or ``Batch`` object with ``x``, ``edge_index``,
                ``edge_attr`` and ``batch`` attributes.

        Returns:
            Logit tensor of shape ``(batch_size,)``.
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze(-1)
