"""SMILES-to-graph conversion and PyTorch dataset for GNN training."""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def mol_to_graph(smiles: str) -> Optional[Data]:
    """Convert a SMILES string to a ``torch_geometric.data.Data`` object.

    Nodes represent atoms and edges represent bonds (undirected).

    Node features (6-dim):
        ``[atomic_num, is_aromatic, formal_charge, hybridization,
        total_Hs, is_in_ring]``

    Edge features (3-dim):
        ``[bond_type_norm, is_conjugated, is_in_ring]``

    Args:
        smiles(str): input SMILES string.

    Returns:
        ``Data`` object or *None* when the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(
            [
                atom.GetAtomicNum(),
                int(atom.GetIsAromatic()),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                atom.GetTotalNumHs(),
                int(atom.IsInRing()),
            ]
        )
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondTypeAsDouble() / 3.0
        feat = [bt, int(bond.GetIsConjugated()), int(bond.IsInRing())]
        edge_indices.extend([[a1, a2], [a2, a1]])
        edge_attrs.extend([feat, feat])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class HIVGraphDataset(torch.utils.data.Dataset):
    """PyTorch dataset converting SMILES + labels into graph objects.

    Args:
        smiles_list(List[str]): SMILES strings.
        labels(List[float]): binary activity labels.
    """

    def __init__(self, smiles_list: List[str], labels: List[float]):
        """Store SMILES and labels after asserting equal lengths."""
        assert len(smiles_list) == len(labels)
        self.smiles = smiles_list
        self.labels = labels

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        graph = mol_to_graph(self.smiles[idx])
        if graph is None:
            graph = Data(
                x=torch.zeros((1, 6)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 3), dtype=torch.float),
            )
        graph.y = torch.tensor(self.labels[idx], dtype=torch.float)
        return graph
