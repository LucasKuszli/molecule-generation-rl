"""SMILES vocabulary construction and tokenization utilities.

Provides a regex-based tokenizer, a bidirectional token-id mapping, and a
helper class for inspecting token distributions in the pre-training dataset.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt

from src.data.loader import get_pretrain_datasplit

logger = logging.getLogger(__name__)

SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|se|@@|@|%[0-9]{2}"
    r"|[BCNOPSFIncos]|[bcnops]|[0-9]"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|\*)"
)


class SMILESTokenizer:
    """Regex-based SMILES tokenizer.

    Handles multi-character tokens (Br, Cl, Si, Se, bracketed atoms,
    two-digit ring numbers) and wraps sequences with start/end tokens.
    """

    START = "^"
    END = "$"

    def tokenize(self, smiles: str, with_start_end: bool = True) -> List[str]:
        """Tokenize a SMILES string into a list of tokens."""
        tokens = SMILES_REGEX.findall(smiles)
        if with_start_end:
            tokens = [self.START] + tokens + [self.END]
        return tokens

    def untokenize(self, tokens: List[str]) -> str:
        """Reconstruct a SMILES string from tokens."""
        parts = []
        for t in tokens:
            if t == self.END:
                break
            if t != self.START:
                parts.append(t)
        return "".join(parts)


class Vocabulary:
    """Bidirectional token <-> id mapping for SMILES tokens.

    Special tokens are always present at fixed positions:
        0: <pad>   1: ^   2: $   3: <unk>
    """

    PAD = "<pad>"
    START = "^"
    END = "$"
    UNK = "<unk>"

    SPECIAL_TOKENS = [PAD, START, END, UNK]

    def __init__(self, token_to_id: Optional[dict] = None):
        if token_to_id is not None:
            self._token_to_id = dict(token_to_id)
        else:
            self._token_to_id = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self._id_to_token = {i: t for t, i in self._token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self._token_to_id[self.PAD]

    @property
    def start_id(self) -> int:
        return self._token_to_id[self.START]

    @property
    def end_id(self) -> int:
        return self._token_to_id[self.END]

    @property
    def unk_id(self) -> int:
        return self._token_to_id[self.UNK]

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def __getitem__(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_id)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert a list of tokens to a list of vocabulary ids."""
        return [self._token_to_id.get(t, self.unk_id) for t in tokens]

    def decode(self, ids) -> List[str]:
        """Convert a sequence of ids back to tokens."""
        return [self._id_to_token.get(int(i), self.UNK) for i in ids]

    def add(self, token: str) -> int:
        """Add a token and return its id. No-op if already present."""
        if token in self._token_to_id:
            return self._token_to_id[token]
        idx = len(self._token_to_id)
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        return idx

    def save(self, path: str | Path) -> None:
        """Save vocabulary to a JSON file."""
        Path(path).write_text(json.dumps(self._token_to_id, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load vocabulary from a JSON file."""
        token_to_id = json.loads(Path(path).read_text())
        return cls(token_to_id)

    def tokens(self) -> List[str]:
        """Return all tokens in id order."""
        return [self._id_to_token[i] for i in range(len(self._id_to_token))]


def build_vocabulary(
    smiles_list: List[str], tokenizer: Optional[SMILESTokenizer] = None
) -> Vocabulary:
    """Build a Vocabulary from a list of SMILES strings.

    Args:
        smiles_list(List[str]): raw SMILES to extract tokens from.
        tokenizer(SMILESTokenizer | None): tokenizer instance; created if *None*.

    Returns:
        Vocabulary populated with every token found in the corpus.
    """
    if tokenizer is None:
        tokenizer = SMILESTokenizer()
    vocab = Vocabulary()  # starts with special tokens.
    for smi in smiles_list:
        for token in tokenizer.tokenize(smi, with_start_end=False):
            vocab.add(token)
    return vocab


class PretrainVocabulary:
    """Utility for inspecting the pre-training vocabulary."""

    def __init__(self, pretrain_train: List[str]):
        """Initialise with the pre-training SMILES list.

        Args:
            pretrain_train(List[str]): list of SMILES from the training split.
        """
        assert isinstance(pretrain_train, list) and all(
            isinstance(s, str) for s in pretrain_train
        )
        self.smiles_list = pretrain_train
        self.tokenizer = SMILESTokenizer()
        self.all_tokens: Counter = Counter()
        self.vocabulary: Optional[Vocabulary] = None

    def build_vocabulary(self) -> Vocabulary:
        """Tokenize all SMILES and build the vocabulary.

        Returns:
            vocab(Vocabulary): vocabulary built from the corpus.
        """
        for smi in self.smiles_list:
            tokens = self.tokenizer.tokenize(smi, with_start_end=False)
            self.all_tokens.update(tokens)

        self.vocabulary = build_vocabulary(self.smiles_list, self.tokenizer)
        logger.info(
            "Vocabulary size: %d tokens (%d unique corpus tokens + %d special)",
            len(self.vocabulary),
            len(self.all_tokens),
            len(Vocabulary.SPECIAL_TOKENS),
        )
        logger.info("Most common tokens:")
        for token, count in self.all_tokens.most_common(15):
            logger.info("  '%s': %s", token, f"{count:,}")
        return self.vocabulary

    def plot_smiles_length_distribution(self) -> None:
        """Plot histogram of SMILES token lengths and save to disk."""
        tokenizer = self.tokenizer
        lengths = [
            len(tokenizer.tokenize(smi, with_start_end=False))
            for smi in self.smiles_list
        ]

        plt.figure(figsize=(10, 4))
        plt.hist(lengths, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
        plt.xlabel("SMILES Length (tokens)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(
            "Distribution of SMILES Token Lengths in Pre-training Data", fontsize=14
        )
        plt.axvline(
            np.mean(lengths),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(lengths):.1f}",
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig("img/pretrain/smiles_length_distribution.png", dpi=150)
        plt.close()

    def show_rare_tokens(self) -> None:
        """Log tokens that occur fewer than 100 000 times in the corpus."""
        rare_tokens = [(t, c) for t, c in self.all_tokens.items() if c < 100000]
        logger.info("Rare tokens (count < 100000): %d", len(rare_tokens))
        for token, count in sorted(rare_tokens, key=lambda x: x[1])[:10]:
            logger.info("  '%s': %d", token, count)


if __name__ == "__main__":
    pretrain_split = get_pretrain_datasplit()
    pretrain_train = pretrain_split.train_set["SMILES"].tolist()
    builder = PretrainVocabulary(pretrain_train)
    vocab = builder.build_vocabulary()
    builder.plot_smiles_length_distribution()
    builder.show_rare_tokens()
    vocab.save("data/training_data/vocab.json")
    logger.info("Vocabulary saved (%d tokens)", len(vocab))
