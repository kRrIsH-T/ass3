"""An LSTM whose gates and recurrence are written explicitly."""

from __future__ import annotations

import torch
from torch import nn

from .base import SequenceEncoder, validate_tokens


class ManualLSTM(SequenceEncoder):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gates = nn.Linear(embedding_dim + hidden_size, 4 * hidden_size)
        self.output_size = hidden_size

    def forward(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        validate_tokens(tokens, padding_mask)
        inputs = self.embedding(tokens)
        hidden = inputs.new_zeros(tokens.size(0), self.output_size)
        cell = torch.zeros_like(hidden)
        outputs = []

        for time_step in range(tokens.size(1)):
            joined = torch.cat((inputs[:, time_step], hidden), dim=-1)
            input_gate, forget_gate, candidate, output_gate = self.gates(joined).chunk(4, -1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            candidate = torch.tanh(candidate)
            output_gate = torch.sigmoid(output_gate)

            next_cell = forget_gate * cell + input_gate * candidate
            next_hidden = output_gate * torch.tanh(next_cell)
            if padding_mask is not None:
                keep_old = padding_mask[:, time_step].unsqueeze(-1)
                cell = torch.where(keep_old, cell, next_cell)
                hidden = torch.where(keep_old, hidden, next_hidden)
            else:
                cell, hidden = next_cell, next_hidden
            outputs.append(hidden)

        if not outputs:
            return inputs.new_empty(tokens.size(0), 0, self.output_size)
        return torch.stack(outputs, dim=1)


LSTMEncoder = ManualLSTM
