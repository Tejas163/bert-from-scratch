"""
Transformer Encoder implementations.
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):
    """Single Transformer Encoder Block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize Transformer Encoder Block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function ("gelu" or "relu")
        """
        super().__init__()

        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        if activation == "gelu":
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, embed_dim),
                nn.Dropout(dropout),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, embed_dim),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        attn_output, _ = self.self_attention(x, mask)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer Encoder Blocks."""

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        """
        Initialize Transformer Encoder.

        Args:
            num_layers: Number of encoder layers
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through all layers.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
