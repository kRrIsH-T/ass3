"""Transformer encoder components with attention written from primitives."""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import SequenceEncoder, validate_tokens


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_length: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / hidden_size)
        )
        encoding = torch.zeros(max_length, hidden_size)
        encoding[:, 0::2] = torch.sin(position * scale)
        if hidden_size > 1:
            encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.size(1) > self.encoding.size(0):
            raise ValueError("sequence is longer than the configured max_length")
        return inputs + self.encoding[: inputs.size(1)].to(inputs.dtype)


class ManualSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_size).transpose(1, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        query = self._heads(self.query(inputs))
        key = self._heads(self.key(inputs))
        value = self._heads(self.value(inputs))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)

        blocked = None
        if padding_mask is not None:
            blocked = padding_mask[:, None, None, :]
        if causal:
            future = torch.triu(
                torch.ones(inputs.size(1), inputs.size(1), dtype=torch.bool, device=inputs.device),
                diagonal=1,
            )[None, None]
            blocked = future if blocked is None else (blocked | future)
        if blocked is not None:
            scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)

        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = weights @ value
        context = context.transpose(1, 2).contiguous().view_as(inputs)
        return self.output(context)


class TransformerBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, feedforward_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = ManualSelfAttention(hidden_size, num_heads, dropout)
        self.feedforward_norm = nn.LayerNorm(hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, inputs: torch.Tensor, padding_mask: torch.Tensor | None, causal: bool
    ) -> torch.Tensor:
        inputs = inputs + self.dropout(
            self.attention(self.attention_norm(inputs), padding_mask, causal)
        )
        return inputs + self.dropout(self.feedforward(self.feedforward_norm(inputs)))


class ManualTransformer(SequenceEncoder):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int = 2,
        feedforward_size: int | None = None,
        max_length: int = 2048,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        feedforward_size = feedforward_size or 4 * hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positions = SinusoidalPositionEncoding(hidden_size, max_length)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, feedforward_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.output_size = hidden_size

    def forward(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        validate_tokens(tokens, padding_mask)
        hidden = self.dropout(self.positions(self.embedding(tokens)))
        for layer in self.layers:
            hidden = layer(hidden, padding_mask, self.causal)
        hidden = self.final_norm(hidden)
        if padding_mask is not None:
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return hidden


TransformerEncoder = ManualTransformer
