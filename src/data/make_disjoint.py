"""Remove HIV SMILES from already-processed Moses CSV files.

Useful when the raw preprocessing has already been run but the disjointness
constraint was not enforced at that time.

Usage::

    uv run python -m src.data.make_disjoint
"""

import logging

import pandas as pd

from src.config.data import DataConfig

logger = logging.getLogger(__name__)


def main() -> None:
    """Filter HIV SMILES out of every Moses split file."""
    config = DataConfig()
    hiv = pd.read_csv(config.hiv_source.sanitized_outpath)
    hiv_smiles = set(hiv["SMILES"])
    logger.info("HIV SMILES: %d", len(hiv_smiles))

    for path in [
        config.moses_source.sanitized_outpath,
        config.moses_source.train_outpath,
        config.moses_source.val_outpath,
        config.moses_source.test_outpath,
    ]:
        df = pd.read_csv(path)
        n_before = len(df)
        df = df[~df["SMILES"].isin(hiv_smiles)].reset_index(drop=True)
        removed = n_before - len(df)
        df.to_csv(path, index=False)
        logger.info("%s: %d -> %d (removed %d)", path.name, n_before, len(df), removed)

    logger.info("Done - Moses and HIV are now disjoint.")


if __name__ == "__main__":
    main()
