"""End-to-end data setup: download from HuggingFace, then preprocess.

Usage::

    uv run python -m src.data.setup
"""

from src.config.data import DataConfig
from src.data.download import load_and_save_datasets
from src.data.preprocess import preprocess_datasets


def setup_data(config: DataConfig) -> None:
    """Run the full data pipeline: download -> sanitise -> scaffold-split.

    Args:
        config(DataConfig): data configuration.
    """
    load_and_save_datasets(config)
    preprocess_datasets(config)


if __name__ == "__main__":
    setup_data(DataConfig())
