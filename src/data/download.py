"""Download raw datasets from HuggingFace and save as local CSV files.

Usage::

    uv run python -m src.data.download
"""

import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset

from src.config.data import DataConfig

logger = logging.getLogger(__name__)


def load_and_save_dataset(dataset_id: str, outpath: Path) -> None:
    """Download all splits of a HuggingFace dataset and save as a single CSV.

    Args:
        dataset_id(str): HuggingFace dataset identifier.
        outpath(Path): destination CSV path.
    """
    try:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        ds_dict = load_dataset(dataset_id)
        from datasets import concatenate_datasets

        ds = concatenate_datasets(list(ds_dict.values()))
        ds.to_csv(str(outpath))
        logger.info("Dataset saved to %s (%d rows)", outpath, len(ds))
    except Exception as e:
        logger.error("Error loading or saving dataset: %s", e)


def load_and_save_datasets(config: Optional[DataConfig] = None) -> None:
    """Download and save all datasets defined in *config*.

    Args:
        config(DataConfig | None): data configuration; defaults are used when ``None``.
    """
    if config is None:
        config = DataConfig()
    load_and_save_dataset(
        config.moses_source.source_url,
        config.moses_source.raw_outpath,
    )
    load_and_save_dataset(
        config.hiv_source.source_url,
        config.hiv_source.raw_outpath,
    )


if __name__ == "__main__":
    config = DataConfig()
    load_and_save_datasets(config)
