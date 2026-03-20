"""Independent fingerprint-based HIV activity evaluator.

Uses Morgan fingerprints + Logistic Regression as an orthogonal evaluator
that is *not* the GNN used as the RL reward signal.  This lets us check
whether molecules flagged as active by the GNN are also flagged by a
structurally different classifier, surfacing potential reward hacking.

Usage (standalone training)::

    uv run python -m src.gnn.fingerprint_evaluator
    uv run python -m src.gnn.fingerprint_evaluator --seed 42
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression

from src.config.data import DataConfig, root_dir

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


def smiles_to_fp(
    smiles: str, radius: int = 2, n_bits: int = 2048
) -> Optional[np.ndarray]:
    """Convert a SMILES string to a Morgan fingerprint bit-vector.

    Args:
        smiles(str): input SMILES.
        radius(int): Morgan fingerprint radius.
        n_bits(int): fingerprint length.

    Returns:
        NumPy float32 array of shape ``(n_bits,)`` or *None* on parse failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class FingerprintEvaluator:
    """Morgan-FP + Logistic Regression HIV activity classifier.

    Provides the same ``predict(smiles) -> (label, prob)`` interface as
    :class:`src.gnn.predictor.HIVActivityPredictor` so that both can be
    used interchangeably during evaluation.
    """

    def __init__(
        self, model=None, threshold: float = 0.5, radius: int = 2, n_bits: int = 2048
    ):
        self.model = model
        self.threshold = threshold
        self.radius = radius
        self.n_bits = n_bits

    @classmethod
    def train_from_dataframe(
        cls,
        df,
        smiles_col: str = "SMILES",
        label_col: str = "label",
        radius: int = 2,
        n_bits: int = 2048,
        seed: int = 42,
    ) -> "FingerprintEvaluator":
        """Fit a Logistic Regression on Morgan fingerprints from a DataFrame.

        Parameters
        ----------
        df : pandas DataFrame with *smiles_col* and *label_col* columns.
        """

        fps, labels = [], []
        for _, row in df.iterrows():
            fp = smiles_to_fp(row[smiles_col], radius=radius, n_bits=n_bits)
            if fp is not None:
                fps.append(fp)
                labels.append(int(row[label_col]))

        X = np.stack(fps)
        y = np.array(labels)

        lr = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        lr.fit(X, y)

        return cls(model=lr, radius=radius, n_bits=n_bits)

    def predict(self, smiles: str) -> Tuple[Optional[int], Optional[float]]:
        """Score a single SMILES. Returns ``(label, prob)`` or ``(None, None)``."""
        fp = smiles_to_fp(smiles, radius=self.radius, n_bits=self.n_bits)
        if fp is None or self.model is None:
            return None, None
        prob = float(self.model.predict_proba(fp.reshape(1, -1))[0, 1])
        label = int(prob >= self.threshold)
        return label, prob

    def predict_batch(self, smiles_list: List[str]) -> List[dict]:
        """Score a list of SMILES. Returns list of ``{smiles, label, prob}``."""
        results = []
        for smi in smiles_list:
            label, prob = self.predict(smi)
            if label is not None:
                results.append({"smiles": smi, "label": label, "prob": prob})
        return results

    def save(self, path: Path) -> None:
        """Pickle the evaluator to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "threshold": self.threshold,
            "radius": self.radius,
            "n_bits": self.n_bits,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> "FingerprintEvaluator":
        """Load a saved evaluator from *path*."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls(
            model=state["model"],
            threshold=state.get("threshold", 0.5),
            radius=state.get("radius", 2),
            n_bits=state.get("n_bits", 2048),
        )


def fp_evaluator_path(seed: int) -> Path:
    """Return the default save path for the fingerprint evaluator."""
    return root_dir / "artifacts" / "gnn" / f"seed_{seed}" / "fp_evaluator.pkl"


def load_fp_evaluator(seed: int) -> Optional[FingerprintEvaluator]:
    """Load the fingerprint evaluator for *seed*, or return ``None``."""
    path = fp_evaluator_path(seed)
    if not path.exists():
        logger.warning("FP evaluator not found at %s", path)
        return None
    try:
        return FingerprintEvaluator.load(path)
    except Exception as e:
        logger.warning("Could not load FP evaluator: %s", e)
        return None


def train_and_save(seed: int = 42) -> Path:
    """Train a fingerprint evaluator on the HIV training set and save it.

    Args:
        seed(int): random seed for the logistic regression.

    Returns:
        Path to the saved evaluator pickle.
    """

    config = DataConfig()
    train_df = pd.read_csv(config.hiv_source.train_outpath)

    logger.info(
        "Training FP evaluator on %d molecules (seed %d) …", len(train_df), seed
    )
    evaluator = FingerprintEvaluator.train_from_dataframe(train_df, seed=seed)

    out_path = fp_evaluator_path(seed)
    evaluator.save(out_path)
    logger.info("Saved to %s", out_path)

    # Quick self-check.
    val_df = pd.read_csv(config.hiv_source.val_outpath)
    fps, labels = [], []
    for _, row in val_df.iterrows():
        fp = smiles_to_fp(row["SMILES"])
        if fp is not None:
            fps.append(fp)
            labels.append(int(row["label"]))
    if fps:
        X = np.stack(fps)
        y = np.array(labels)
        preds = evaluator.model.predict(X)
        from sklearn.metrics import balanced_accuracy_score

        logger.info(
            "Validation balanced accuracy: %.3f", balanced_accuracy_score(y, preds)
        )

    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Morgan-FP + LR HIV evaluator")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_and_save(seed=args.seed)
