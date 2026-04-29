from pathlib import Path

import torch
from torch import nn

from benchmark.config import BenchmarkConfig, DataConfig, TrainingConfig
from benchmark.data import AGNewsDataModule, Vocabulary
from benchmark.runner import run_benchmark


class TinyClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 8, padding_idx=padding_idx)
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, tokens, lengths=None):
        mask = tokens.ne(0).unsqueeze(-1)
        pooled = (self.embedding(tokens) * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(pooled)


class MaskClassifier(TinyClassifier):
    def forward(self, tokens, padding_mask=None):
        assert padding_mask is not None and padding_mask.dtype == torch.bool
        embedded = self.embedding(tokens).masked_fill(padding_mask.unsqueeze(-1), 0)
        return self.classifier(embedded.sum(1) / (~padding_mask).sum(1, keepdim=True).clamp_min(1))


def test_vocabulary_is_deterministic():
    texts = ["zebra apple", "apple zebra", "banana"]
    assert Vocabulary.build(texts, 1, 10).itos == ["<pad>", "<unk>", "apple", "zebra", "banana"]


def test_offline_data_is_stratified_and_encoded(tmp_path: Path):
    config = DataConfig(output_dir=str(tmp_path), force_fallback=True, batch_size=4, min_frequency=1)
    module = AGNewsDataModule(config, seed=7)
    info = module.prepare()
    assert info.train_samples == 4
    assert info.validation_samples == 4
    assert info.test_samples == 4
    inputs, labels, lengths = next(iter(module.loaders()["train"]))
    assert inputs.shape == (4, config.max_length)
    assert set(labels.tolist()) == {0, 1, 2, 3}
    assert torch.all(lengths > 0)


def test_end_to_end_smoke(tmp_path: Path):
    config = BenchmarkConfig(
        data=DataConfig(
            output_dir=str(tmp_path), force_fallback=True, batch_size=4,
            max_length=12, min_frequency=1,
        ),
        training=TrainingConfig(epochs=1, patience=1, inference_warmup_batches=0, inference_batches=1),
    )
    results = run_benchmark(config, registry={"tiny": TinyClassifier})
    assert len(results) == 1
    assert 0 <= results[0].test_accuracy <= 1
    assert (tmp_path / "tables" / "benchmark.csv").exists()
    assert (tmp_path / "plots" / "tiny_curves.png").exists()


def test_padding_mask_model_contract(tmp_path: Path):
    config = BenchmarkConfig(
        data=DataConfig(output_dir=str(tmp_path), force_fallback=True, batch_size=4, max_length=12, min_frequency=1),
        training=TrainingConfig(epochs=1, patience=1, inference_warmup_batches=0, inference_batches=1),
    )
    assert run_benchmark(config, registry={"mask": MaskClassifier})[0].epochs_completed == 1
