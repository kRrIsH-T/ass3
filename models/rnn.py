"""A vanilla recurrent network implemented one time step at a time."""

from __future__ import annotations

import torch
from torch import nn

from .base import SequenceEncoder, validate_tokens


class VanillaRNN(SequenceEncoder):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_projection = nn.Linear(embedding_dim, hidden_size)
        self.hidden_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_size = hidden_size

    def forward(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        validate_tokens(tokens, padding_mask)
        inputs = self.embedding(tokens)
        hidden = inputs.new_zeros(tokens.size(0), self.output_size)
        outputs = []

        for time_step in range(tokens.size(1)):
            candidate = torch.tanh(
                self.input_projection(inputs[:, time_step])
                + self.hidden_projection(hidden)
            )
            if padding_mask is not None:
                keep_old = padding_mask[:, time_step].unsqueeze(-1)
                hidden = torch.where(keep_old, hidden, candidate)
            else:
                hidden = candidate
            outputs.append(hidden)

        if not outputs:
            return inputs.new_empty(tokens.size(0), 0, self.output_size)
        return torch.stack(outputs, dim=1)


RNNEncoder = VanillaRNN
