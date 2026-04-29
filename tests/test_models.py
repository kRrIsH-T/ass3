import pytest
import torch
from torch import nn

from models import ManualLSTM, ManualTransformer, SelectiveSSM, SequenceClassifier, VanillaRNN


@pytest.mark.parametrize(
    "encoder",
    [
        VanillaRNN(31, 8, 12),
        ManualLSTM(31, 8, 12),
        SelectiveSSM(31, 8, 12, state_size=5),
        ManualTransformer(31, 12, num_heads=3, num_layers=2),
    ],
)
def test_encoder_shape_padding_and_gradients(encoder):
    tokens = torch.tensor([[2, 4, 6, 0, 0], [1, 3, 5, 7, 9]])
    padding = torch.tensor([[False, False, False, True, True], [False] * 5])
    output = encoder(tokens, padding)

    assert output.shape == (2, 5, 12)
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in encoder.parameters())


@pytest.mark.parametrize(
    "encoder",
    [VanillaRNN(20, 6, 10), ManualLSTM(20, 6, 10), SelectiveSSM(20, 6, 10, 4)],
)
def test_recurrent_padding_does_not_change_state(encoder):
    tokens = torch.tensor([[3, 8, 0, 0]])
    padding = torch.tensor([[False, False, True, True]])
    output = encoder(tokens, padding)
    torch.testing.assert_close(output[:, 1], output[:, 2])
    torch.testing.assert_close(output[:, 2], output[:, 3])


def test_classifier_ignores_right_padding():
    model = SequenceClassifier(VanillaRNN(30, 7, 11), num_classes=4)
    short = model(torch.tensor([[2, 5, 9]]))
    padded = model(
        torch.tensor([[2, 5, 9, 12, 13]]),
        torch.tensor([[False, False, False, True, True]]),
    )
    torch.testing.assert_close(short, padded)


def test_causal_transformer_cannot_see_future_tokens():
    model = ManualTransformer(30, 12, 3, num_layers=2, causal=True)
    model.eval()
    first = model(torch.tensor([[1, 2, 3, 4]]))
    changed = model(torch.tensor([[1, 2, 17, 19]]))
    torch.testing.assert_close(first[:, :2], changed[:, :2], atol=1e-6, rtol=1e-6)


def test_no_forbidden_torch_sequence_modules_are_used():
    forbidden = (nn.RNNBase, nn.MultiheadAttention, nn.Transformer)
    models = [
        VanillaRNN(10, 4, 6),
        ManualLSTM(10, 4, 6),
        SelectiveSSM(10, 4, 6),
        ManualTransformer(10, 6, 2),
    ]
    assert not any(isinstance(module, forbidden) for model in models for module in model.modules())
