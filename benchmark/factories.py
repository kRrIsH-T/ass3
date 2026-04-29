"""Small composition helpers connecting encoders to the common classifier."""

from models import ManualLSTM, ManualTransformer, SelectiveSSM, SequenceClassifier, VanillaRNN


def rnn_classifier(vocab_size: int, num_classes: int, embedding_dim=128, hidden_size=192, dropout=0.2):
    return SequenceClassifier(
        VanillaRNN(vocab_size, embedding_dim, hidden_size),
        num_classes,
        pooling="mean",
        dropout=dropout,
    )


def lstm_classifier(vocab_size: int, num_classes: int, embedding_dim=128, hidden_size=192, dropout=0.2):
    return SequenceClassifier(
        ManualLSTM(vocab_size, embedding_dim, hidden_size),
        num_classes,
        pooling="mean",
        dropout=dropout,
    )


def ssm_classifier(
    vocab_size: int, num_classes: int, embedding_dim=128, hidden_size=192,
    state_size=16, dropout=0.2,
):
    encoder = SelectiveSSM(vocab_size, embedding_dim, hidden_size, state_size, dropout)
    return SequenceClassifier(encoder, num_classes, pooling="mean", dropout=dropout)


def transformer_classifier(
    vocab_size: int, num_classes: int, hidden_size=128, num_heads=4,
    num_layers=2, feedforward_size=512, max_length=128, dropout=0.2,
):
    encoder = ManualTransformer(
        vocab_size, hidden_size, num_heads, num_layers, feedforward_size, max_length, dropout
    )
    return SequenceClassifier(encoder, num_classes, pooling="mean", dropout=dropout)
