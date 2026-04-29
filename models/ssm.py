"""A compact input-selective state-space (Mamba-style) encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .base import SequenceEncoder, validate_tokens


class SelectiveSSM(SequenceEncoder):
    """Diagonal selective SSM followed by a gated output projection.

    The transition matrix ``A`` is stable and learned.  The step size, input
    coefficient, and readout coefficient depend on the current token, which
    is the central selective idea used by modern Mamba-like models.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        state_size: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_projection = nn.Linear(embedding_dim, hidden_size)
        self.parameters_projection = nn.Linear(hidden_size, 1 + 2 * state_size)
        self.gate_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.skip = nn.Parameter(torch.ones(hidden_size))
        self.log_a = nn.Parameter(torch.zeros(hidden_size, state_size))
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.state_size = state_size
        self.output_size = hidden_size

    def forward(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        validate_tokens(tokens, padding_mask)
        projected = self.input_projection(self.embedding(tokens))
        state = projected.new_zeros(tokens.size(0), self.output_size, self.state_size)
        outputs = []
        # Negative diagonal entries guarantee a decaying continuous transition.
        a = -torch.exp(self.log_a)

        for time_step in range(tokens.size(1)):
            current = projected[:, time_step]
            delta, b, c = torch.split(
                self.parameters_projection(current), [1, self.state_size, self.state_size], -1
            )
            delta = F.softplus(delta) + 1e-4
            transition = torch.exp(delta.unsqueeze(-1) * a.unsqueeze(0))
            next_state = transition * state + delta.unsqueeze(-1) * (
                current.unsqueeze(-1) * b.unsqueeze(1)
            )
            selective_readout = (next_state * c.unsqueeze(1)).sum(dim=-1)
            value = selective_readout + self.skip * current
            value = value * torch.sigmoid(self.gate_projection(current))
            output = self.norm(current + self.dropout(self.output_projection(value)))

            if padding_mask is not None:
                keep_old = padding_mask[:, time_step, None, None]
                state = torch.where(keep_old, state, next_state)
                output = torch.where(
                    padding_mask[:, time_step, None],
                    outputs[-1] if outputs else torch.zeros_like(output),
                    output,
                )
            else:
                state = next_state
            outputs.append(output)

        if not outputs:
            return projected.new_empty(tokens.size(0), 0, self.output_size)
        return torch.stack(outputs, dim=1)


MambaEncoder = SelectiveSSM
