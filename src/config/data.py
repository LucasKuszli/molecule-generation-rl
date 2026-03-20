"""Data source and split configuration for the pipeline.

Defines dataset locations, split ratios, and file paths for the
MOSES (pretraining) and HIV (classification / transfer) datasets.
"""

import dataclasses
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent.resolve()

# Split defs


class DataSplit:
    """Base descriptor for a train / val / test partition."""

    split_type: str = ""
    target: str = ""
    train_size: float = 0.0
    val_size: float = 0.0
    test_size: float = 0.0


class PreTrainSplit(DataSplit):
    """MOSES pretrain split (80 / 10 / 10)."""

    split_type = "pretrain"
    target_set = "zinc"
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1


class GNNSplit(DataSplit):
    """HIV GNN classification split (80 / 10 / 10)."""

    split_type = "gnn"
    target_set = "hiv"
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1


class TransferSplit(DataSplit):
    """Transfer learning split on HIV actives (90 / 10)."""

    split_type = "transfer"
    target = "hiv_training_actives"
    train_size = 0.9
    val_size = 0.1


# Data sources


class DataSetSource:
    """HuggingFace dataset identifier with associated file paths."""

    source_name: str = ""
    source_url: str = ""
    splits: list[DataSplit] = []
    raw_outpath: Path = Path()
    sanitized_outpath: Path = Path()
    train_outpath: Path = Path()
    val_outpath: Path = Path()
    test_outpath: Path = Path()


class Moses(DataSetSource):
    """MOSES dataset - ~1.9M drug-like SMILES for language-model pretraining."""

    source_name = "moses"
    source_url = "antoinebcx/smiles-molecules-moses"
    splits = [PreTrainSplit()]
    raw_outpath = root_dir / "data" / "raw_downloads" / "moses.csv"
    sanitized_outpath = root_dir / "data" / "training_data" / "moses.csv"
    train_outpath = root_dir / "data" / "training_data" / "moses_train.csv"
    val_outpath = root_dir / "data" / "training_data" / "moses_val.csv"
    test_outpath = root_dir / "data" / "training_data" / "moses_test.csv"


class HIV(DataSetSource):
    """MoleculeNet HIV dataset - ~41k molecules with binary activity labels."""

    source_name = "hiv"
    source_url = "scikit-fingerprints/MoleculeNet_HIV"
    splits = [GNNSplit(), TransferSplit()]
    raw_outpath = root_dir / "data" / "raw_downloads" / "hiv.csv"
    sanitized_outpath = root_dir / "data" / "training_data" / "hiv.csv"
    train_outpath = root_dir / "data" / "training_data" / "hiv_train.csv"
    val_outpath = root_dir / "data" / "training_data" / "hiv_val.csv"
    test_outpath = root_dir / "data" / "training_data" / "hiv_test.csv"


# Main project config


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Top-level data configuration combining both dataset sources."""

    source_names = ["moses", "hiv"]
    moses_source: Moses = Moses()
    hiv_source: HIV = HIV()
    split_type: str = "scaffold"
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
