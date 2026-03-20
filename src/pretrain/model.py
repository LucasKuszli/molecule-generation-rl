"""SMILES LSTM language model for molecular generation.

Embedding -> multi-layer LSTM -> linear decoder.
Trained with next-token prediction (CrossEntropyLoss, ignoring ``<pad>``).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.pretrain.vocabulary import SMILESTokenizer, Vocabulary

logger = logging.getLogger(__name__)


class SmilesLSTM(nn.Module):
    """Character-level SMILES language model.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    embedding_dim : int
        Dimension of the token embedding layer.
    hidden_dim : int
        LSTM hidden state size.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers (applied when num_layers > 1).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        x : (batch, seq_len) long tensor of token ids.
        hidden : optional (h_0, c_0) tuple.

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        hidden : (h_n, c_n) tuple
        """
        emb = self.embedding(x)  # (B, T, E)
        out, hidden = self.lstm(emb, hidden)  # (B, T, H)
        logits = self.linear(out)  # (B, T, V)
        return logits, hidden

    def get_params(self) -> dict:
        """Return a dict of constructor kwargs (useful for checkpointing)."""
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    # Likelihood calculation for training/evaluation.
    def nll(self, sequences: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        """Compute per-sequence negative log-likelihood.

        Parameters
        ----------
        sequences : (batch, seq_len) - full sequences including start/end tokens.
        pad_id : token id used for padding (ignored in loss).

        Returns
        -------
        (batch,) tensor of NLL values.
        """
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        logits, _ = self.forward(inputs)
        log_probs = logits.log_softmax(dim=-1)
        token_nll = nn.functional.nll_loss(
            log_probs.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=pad_id,
            reduction="none",
        ).view(targets.shape)
        return token_nll.sum(dim=1)

    # Sampling from the model autoregressively.
    @torch.no_grad()
    def sample(
        self,
        vocab: Vocabulary,
        tokenizer: SMILESTokenizer,
        n: int = 128,
        max_len: int = 140,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[List[str], torch.Tensor]:
        """Sample SMILES strings autoregressively.

        Returns
        -------
        smiles : list of n SMILES strings
        nlls   : (n,) tensor of negative log-likelihoods
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()

        start_id = vocab.start_id
        end_id = vocab.end_id

        input_ids = torch.full((n, 1), start_id, dtype=torch.long, device=device)
        hidden = None
        sequences = [input_ids]
        nlls = torch.zeros(n, device=device)

        for _ in range(max_len - 1):
            logits, hidden = self.forward(input_ids, hidden)
            logits = logits[:, -1, :] / temperature  # (n, V)
            probs = logits.softmax(dim=-1)
            log_probs = logits.log_softmax(dim=-1)

            input_ids = torch.multinomial(probs, 1)  # (n, 1)
            nlls -= log_probs.gather(1, input_ids).squeeze(1)
            sequences.append(input_ids)

            # Stop early if every sequence has produced END.
            if (input_ids.squeeze(1) == end_id).all():
                break

        sequences = torch.cat(sequences, dim=1)  # (n, T)

        # Decode to SMILES.
        smiles = []
        for seq in sequences.cpu().numpy():
            tokens = vocab.decode(seq)
            smiles.append(tokenizer.untokenize(tokens))

        return smiles, nlls
