"""
Complete BERT model implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .embeddings import BERTEmbeddings
from .transformer import TransformerEncoder
from .attention import scaled_dot_product_attention


class Pooler(nn.Module):
    """Pooler layer for extracting [CLS] token representation."""

    def __init__(self, d_model: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token = hidden_states[:, 0]
        return self.activation(self.dense(first_token))


class BERTModel(nn.Module):
    """Base BERT model returning sequence and pooler outputs."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize BERT model.

        Args:
            vocab_size: Vocabulary size
            d_model: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.embeddings = BERTEmbeddings(vocab_size, d_model, max_len, dropout=dropout)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.pooler = Pooler(d_model)

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        """
        Forward pass.

        Args:
            token_ids: Token IDs (batch, seq_len)
            segment_ids: Segment IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            sequence_output: (batch, seq_len, d_model)
            pooled_output: (batch, d_model)
        """
        embeddings = self.embeddings(token_ids, segment_ids)

        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = extended_mask.to(dtype=embeddings.dtype)
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None

        sequence_output = self.encoder(embeddings, extended_mask)
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output


class MLMHead(nn.Module):
    """Masked Language Modeling head."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        return self.decoder(x)


class NSPHead(nn.Module):
    """Next Sentence Prediction head."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 2)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        return self.linear(pooled_output)


class BERTForPretraining(nn.Module):
    """BERT model for pretraining (MLM + NSP)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = BERTModel(
            vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout
        )
        self.mlm_head = MLMHead(d_model, vocab_size)
        self.nsp_head = NSPHead(d_model)

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        mlm_labels: torch.Tensor = None,
        nsp_labels: torch.Tensor = None,
    ):
        """
        Forward pass for pretraining.

        Args:
            token_ids: Token IDs
            segment_ids: Segment IDs
            attention_mask: Attention mask
            mlm_labels: MLM labels (-100 for ignored)
            nsp_labels: NSP labels

        Returns:
            loss: Combined MLM + NSP loss
            mlm_logits: MLM predictions
            nsp_logits: NSP predictions
        """
        sequence_output, pooled_output = self.bert(
            token_ids, segment_ids, attention_mask
        )

        mlm_logits = self.mlm_head(sequence_output)
        nsp_logits = self.nsp_head(pooled_output)

        loss = None
        if mlm_labels is not None and nsp_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(
                mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1)
            )
            nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
            loss = mlm_loss + nsp_loss

        return loss, mlm_logits, nsp_logits


class BERTForClassification(nn.Module):
    """BERT for sequence classification."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int = 2,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = BERTModel(
            vocab_size, d_model, num_heads, num_layers, d_ff, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        Forward pass for classification.

        Args:
            token_ids: Token IDs
            attention_mask: Attention mask
            labels: Classification labels

        Returns:
            loss: Cross-entropy loss
            logits: Classification logits
        """
        _, pooled = self.bert(token_ids, attention_mask=attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return loss, logits


class BERTForQuestionAnswering(nn.Module):
    """BERT for question answering (SQuAD-style)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = BERTModel(
            vocab_size, d_model, num_heads, num_layers, d_ff, dropout=dropout
        )
        self.qa_outputs = nn.Linear(d_model, 2)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        start_positions: torch.Tensor = None,
        end_positions: torch.Tensor = None,
    ):
        """
        Forward pass for QA.

        Args:
            token_ids: Token IDs
            attention_mask: Attention mask
            start_positions: Start position labels
            end_positions: End position labels

        Returns:
            loss: SQuAD loss
            start_logits: Start position logits
            end_logits: End position logits
        """
        sequence_output, _ = self.bert(token_ids, attention_mask=attention_mask)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return loss, start_logits, end_logits
