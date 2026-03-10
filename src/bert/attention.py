"""
Self-Attention and Multi-Head Attention implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor (batch, seq_len_q, d_k)
        K: Key tensor (batch, seq_len_k, d_k)
        V: Value tensor (batch, seq_len_k, d_v)
        mask: Optional attention mask

    Returns:
        output: Attention output (batch, seq_len_q, d_v)
        attention_weights: Attention weights (batch, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1))
    scaled_scores = scores / np.sqrt(d_k)

    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scaled_scores, dim=-1)
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


class SelfAttention(nn.Module):
    """Self-attention module with learned projections."""

    def __init__(
        self, embed_dim: int, d_k: int = None, d_v: int = None, dropout: float = 0.1
    ):
        """
        Initialize Self-Attention.

        Args:
            embed_dim: Input embedding dimension
            d_k: Dimension for keys (defaults to embed_dim)
            d_v: Dimension for values (defaults to embed_dim)
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.d_k = d_k if d_k is not None else embed_dim
        self.d_v = d_v if d_v is not None else embed_dim

        self.W_q = nn.Linear(embed_dim, self.d_k)
        self.W_k = nn.Linear(embed_dim, self.d_k)
        self.W_v = nn.Linear(embed_dim, self.d_v)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_v)
            attention_weights: (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize Multi-Head Attention.

        Args:
            embed_dim: Input embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_len, embed_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        Q = (
            self.W_q(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_k(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_v(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        output = self.W_o(context)

        return output, attn_weights
