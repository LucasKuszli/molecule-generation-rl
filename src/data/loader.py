"""Convenience loaders for the pre-split CSV datasets.

Each ``get_*`` function reads the on-disk CSVs and returns a typed
``DataSplit`` container so that downstream code never handles raw paths.
"""

from pathlib import Path

import pandas as pd

from src.config.data import DataConfig
from src.data.preprocess import scaffold_split_indices


class DataSplit:
    """Base class for data splits."""

    def __init__(
        self,
        train_set: pd.DataFrame,
        val_set: pd.DataFrame,
        test_set: pd.DataFrame | None = None,
    ):
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set


class PreTrainDataSplit(DataSplit):
    """Data split for pretraining.
    Contains the zinc dataset split into train, val and test sets,
    default split is 80/10/10."""

    def __init__(
        self, train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame
    ):
        super().__init__(train_set=train_set, test_set=test_set, val_set=val_set)


class TransferDataSplit(DataSplit):
    """Data split for transfer learning. Contains the HIV actives subset of the HIV dataset
    split into 90% train and a 10% holdout set for validation."""

    def __init__(self, train_set: pd.DataFrame, val_set: pd.DataFrame):
        super().__init__(train_set=train_set, val_set=val_set)


class GNNDataSplit(DataSplit):
    """Data split for the GNN training, full HIV set split into train, test, validation. Fractions
    are 80/10/10."""

    def __init__(
        self, train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame
    ):
        super().__init__(train_set=train_set, val_set=val_set, test_set=test_set)


class ReinforcementDataSplit(DataSplit):
    """
    Data split for the reinforcement learning stage, contains the full HIV dataset split into train, test and validation sets,
    fractions are 80/10/10.
    """

    def __init__(
        self, train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame
    ):
        super().__init__(train_set=train_set, val_set=val_set, test_set=test_set)


def get_pretrain_datasplit() -> PreTrainDataSplit:
    """Load the MOSES pretrain split from disk.

    Returns:
        PreTrainDataSplit: train / val / test DataFrames.
    """
    config = DataConfig()
    train_df = pd.read_csv(config.moses_source.train_outpath)
    validation_df = pd.read_csv(config.moses_source.val_outpath)
    test_df = pd.read_csv(config.moses_source.test_outpath)
    return PreTrainDataSplit(train_df, validation_df, test_df)


def get_transfer_dataset() -> TransferDataSplit:
    """Load the HIV actives transfer split (random 90/10).

    Returns:
        TransferDataSplit: train / val DataFrames of HIV actives.
    """
    config = DataConfig()
    train_df = pd.read_csv(config.hiv_source.train_outpath)
    hiv_actives = train_df[train_df["label"] == 1]
    train_split = hiv_actives.sample(frac=0.9, random_state=42)
    val_split = hiv_actives.drop(train_split.index)
    return TransferDataSplit(train_split, val_split)


def get_transfer_dataset_scaffold(seed: int = 42):
    """Scaffold-aware version of :func:`get_transfer_dataset`.

    All molecules sharing the same Bemis-Murcko scaffold land in the
    same split, so the validation set contains genuinely unseen scaffolds.
    Uses 90 / 10 train / val fractions (no test set for transfer learning).
    """
    config = DataConfig()
    train_df = pd.read_csv(config.hiv_source.train_outpath)
    hiv_actives = train_df[train_df["label"] == 1].reset_index(drop=True)

    train_idx, val_idx, _ = scaffold_split_indices(
        hiv_actives["SMILES"].tolist(),
        train_frac=0.9,
        val_frac=0.1,
        test_frac=0.0,
        seed=seed,
    )
    return TransferDataSplit(
        hiv_actives.iloc[train_idx],
        hiv_actives.iloc[val_idx],
    )


def get_gnn_datasplit() -> GNNDataSplit:
    """Load the HIV GNN classification split from disk.

    Returns:
        GNNDataSplit: train / val / test DataFrames.
    """
    config = DataConfig()
    train_df = pd.read_csv(config.hiv_source.train_outpath)
    val_df = pd.read_csv(config.hiv_source.val_outpath)
    test_df = pd.read_csv(config.hiv_source.test_outpath)
    return GNNDataSplit(train_df, val_df, test_df)


def get_reinforcement_datasplit() -> ReinforcementDataSplit:
    """Load the HIV RL split from disk.

    Returns:
        ReinforcementDataSplit: train / val / test DataFrames.
    """
    config = DataConfig()
    train_df = pd.read_csv(config.hiv_source.train_outpath)
    val_df = pd.read_csv(config.hiv_source.val_outpath)
    test_df = pd.read_csv(config.hiv_source.test_outpath)
    return ReinforcementDataSplit(train_df, val_df, test_df)
