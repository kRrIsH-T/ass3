"""Shared building blocks for the sequence models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


def validate_tokens(tokens: torch.Tensor, padding_mask: torch.Tensor | None) -> None:
    """Check the common input contract used by all encoders."""
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, sequence_length)")
    if padding_mask is not None:
        if padding_mask.shape != tokens.shape:
            raise ValueError("padding_mask must have the same shape as tokens")
        if padding_mask.dtype is not torch.bool:
            raise TypeError("padding_mask must be a boolean tensor")


class SequenceEncoder(nn.Module, ABC):
    """Interface shared by all token sequence encoders."""

    output_size: int

    @abstractmethod
    def forward(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return one hidden vector per input token."""


class SequenceClassifier(nn.Module):
    """Turn any :class:`SequenceEncoder` into a sequence classifier.

    ``pooling='last'`` selects the final non-padding representation.  Mean
    pooling is useful when every position should contribute to the decision.
    """

    def __init__(
        self,
        encoder: SequenceEncoder,
        num_classes: int,
        pooling: str = "last",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be 'last' or 'mean'")
        self.encoder = encoder
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.output_size, num_classes)

    def forward(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self.encoder(tokens, padding_mask)
        valid = (
            torch.ones_like(tokens, dtype=torch.bool)
            if padding_mask is None
            else ~padding_mask
        )
        lengths = valid.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("each sequence must contain at least one non-padding token")

        if self.pooling == "mean":
            weights = valid.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / lengths.unsqueeze(-1)
        else:
            batch = torch.arange(tokens.size(0), device=tokens.device)
            positions = torch.arange(tokens.size(1), device=tokens.device)
            last_valid = positions.expand_as(tokens).masked_fill(~valid, -1).max(dim=1).values
            pooled = hidden[batch, last_valid]
        return self.head(self.dropout(pooled))
