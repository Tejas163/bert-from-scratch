"""
Embedding implementations for BERT.
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (from "Attention is All You Need")."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings (used in BERT)."""

    def __init__(self, d_model: int, max_len: int = 512):
        """
        Initialize learned positional embeddings.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings to input."""
        batch_size, seq_len = x.shape[:2]
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        return x + self.position_embeddings(positions)


class BERTEmbeddings(nn.Module):
    """BERT embedding layer combining token, segment, and position embeddings."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 512,
        segment_vocab_size: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize BERT embeddings.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            max_len: Maximum sequence length
            segment_vocab_size: Number of segment types (2 for BERT)
            dropout: Dropout probability
        """
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(segment_vocab_size, d_model)
        self.position_embedding = LearnedPositionalEmbedding(d_model, max_len)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, token_ids: torch.Tensor, segment_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            token_ids: Token IDs (batch, seq_len)
            segment_ids: Segment IDs (batch, seq_len), optional

        Returns:
            Embeddings (batch, seq_len, d_model)
        """
        token_embeds = self.token_embedding(token_ids)

        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)
        segment_embeds = self.segment_embedding(segment_ids)

        embeddings = token_embeds + segment_embeds
        embeddings = self.position_embedding(embeddings)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
