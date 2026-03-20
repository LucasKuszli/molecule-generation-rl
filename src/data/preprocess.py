"""Sanitise raw SMILES and split datasets by Bemis-Murcko scaffolds.

Pipeline: canonicalise -> desalt -> neutralise -> deduplicate -> scaffold split.

Usage::

    uv run python -m src.data.preprocess
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from src.config.data import DataConfig

logger = logging.getLogger(__name__)


def scaffold_smiles(smiles: str) -> str:
    """Return the Bemis-Murcko generic scaffold SMILES for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or ""


def scaffold_split_indices(
    smiles_list: List[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Partition into train / val / test by Murcko scaffold.

    All molecules sharing the same scaffold end up in the same split.
    Groups are sorted by size (largest first) and then shuffled
    deterministically with seed before the greedy bin-packing assignment.
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    scaffold_to_rows: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles_list):
        scaffold_to_rows.setdefault(scaffold_smiles(smi), []).append(idx)

    groups = list(scaffold_to_rows.values())
    groups.sort(key=len, reverse=True)
    random.Random(seed).shuffle(groups)

    n = len(smiles_list)
    target_train = int(round(train_frac * n))
    target_val = int(round(val_frac * n))

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for g in groups:
        if len(train_idx) + len(g) <= target_train:
            train_idx.extend(g)
        elif len(val_idx) + len(g) <= target_val:
            val_idx.extend(g)
        else:
            test_idx.extend(g)

    return train_idx, val_idx, test_idx


def standardize_smiles(smiles: str) -> Optional[str]:
    """Canonicalise a SMILES string: parse, desalt, neutralise.

    Args:
        smiles(str): raw SMILES string.

    Returns:
        str | None: canonical SMILES, or ``None`` if the molecule is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Remove salts/solvents - keep largest fragment.
        remover = rdMolStandardize.LargestFragmentChooser()
        mol = remover.choose(mol)

        # Uncharge molecule.
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        # Get canonical SMILES.
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def sanitize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise SMILES column, drop invalid entries, and deduplicate.

    Args:
        df(pd.DataFrame): dataframe with a ``SMILES`` or ``smiles`` column.

    Returns:
        pd.DataFrame: cleaned dataframe.
    """
    # Normalize column name to 'SMILES'.
    if "smiles" in df.columns and "SMILES" not in df.columns:
        df = df.rename(columns={"smiles": "SMILES"})

    df["SMILES"] = [
        standardize_smiles(s) for s in tqdm(df["SMILES"], desc="Standardizing SMILES")
    ]
    df = (
        df.dropna(subset=["SMILES"])
        .drop_duplicates(subset=["SMILES"])
        .reset_index(drop=True)
    )
    return df


def preprocess_datasets(config: Optional[DataConfig] = None) -> None:
    """Sanitise raw CSVs, enforce disjointness, and scaffold-split.

    HIV is cleaned first, then MOSES is cleaned and any overlapping SMILES
    are removed so that the two datasets share no molecules.

    Args:
        config(DataConfig | None): data configuration; defaults are used when ``None``.
    """
    if config is None:
        config = DataConfig()

    cleaned_datasets: Dict[str, pd.DataFrame] = {}

    for dataset in [config.hiv_source, config.moses_source]:
        try:
            raw_data = pd.read_csv(dataset.raw_outpath)
            cleaned = sanitize_dataset(raw_data)
            cleaned_datasets[dataset.source_name] = cleaned
        except Exception as e:
            logger.error("Error processing %s: %s", dataset.raw_outpath, e)

    # Remove HIV SMILES from Moses to keep datasets disjoint.
    if "hiv" in cleaned_datasets and "moses" in cleaned_datasets:
        hiv_smiles = set(cleaned_datasets["hiv"]["SMILES"])
        moses = cleaned_datasets["moses"]
        n_before = len(moses)
        moses = moses[~moses["SMILES"].isin(hiv_smiles)].reset_index(drop=True)
        n_removed = n_before - len(moses)
        logger.info(
            "Removed %d HIV-overlapping SMILES from Moses (%d -> %d)",
            n_removed,
            n_before,
            len(moses),
        )
        cleaned_datasets["moses"] = moses

    for dataset in [config.hiv_source, config.moses_source]:
        cleaned = cleaned_datasets.get(dataset.source_name)
        if cleaned is None:
            continue
        try:
            train_idx, val_idx, test_idx = scaffold_split_indices(
                cleaned["SMILES"].tolist(),
                train_frac=config.train_size,
                val_frac=config.val_size,
                test_frac=config.test_size,
            )

            dataset.sanitized_outpath.parent.mkdir(parents=True, exist_ok=True)
            cleaned.to_csv(dataset.sanitized_outpath, index=False)
            cleaned.iloc[train_idx].to_csv(dataset.train_outpath, index=False)
            cleaned.iloc[val_idx].to_csv(dataset.val_outpath, index=False)
            cleaned.iloc[test_idx].to_csv(dataset.test_outpath, index=False)

            logger.info(
                "Saved %s: %d total -> %d train / %d val / %d test",
                dataset.source_name,
                len(cleaned),
                len(train_idx),
                len(val_idx),
                len(test_idx),
            )
        except Exception as e:
            logger.error("Error splitting %s: %s", dataset.source_name, e)


if __name__ == "__main__":
    preprocess_datasets()
