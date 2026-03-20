"""Configuration sub-package for the molecular generation pipeline."""

from pathlib import Path

config_dir = Path(__file__).parent.resolve()

__all__ = ["data", "training", "seed"]
