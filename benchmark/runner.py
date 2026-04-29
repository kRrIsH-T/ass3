from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Callable

from torch import nn

from .config import BenchmarkConfig
from .data import AGNewsDataModule
from .engine import ExperimentResult, Trainer, resolve_device, seed_everything
from .reporting import write_reports

ModelFactory = Callable[..., nn.Module]


def import_factory(specification: str) -> ModelFactory:
    try:
        module_name, attribute = specification.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Model target must use 'module:attribute': {specification}") from exc
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"Model target is not callable: {specification}")
    return factory


def _build(
    factory: ModelFactory, parameters: dict, vocabulary_size: int,
    num_classes: int, max_length: int,
) -> nn.Module:
    values = dict(parameters)
    signature = inspect.signature(factory)
    aliases = {
        "vocab_size": vocabulary_size,
        "vocabulary_size": vocabulary_size,
        "num_classes": num_classes,
        "n_classes": num_classes,
        "pad_idx": 0,
        "padding_idx": 0,
        "max_length": max_length,
    }
    for name, value in aliases.items():
        if name in signature.parameters and name not in values:
            values[name] = value
    model = factory(**values)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Factory {factory!r} did not produce a torch.nn.Module")
    return model


def run_benchmark(
    config: BenchmarkConfig,
    registry: dict[str, ModelFactory] | None = None,
    selected: list[str] | None = None,
) -> list[ExperimentResult]:
    """Run models against one data definition and a reset loader order per model."""
    seed_everything(config.seed)
    data = AGNewsDataModule(config.data, config.seed)
    info = data.prepare()
    output_dir = Path(config.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    factories = dict(registry or {})
    definitions = config.models
    names = selected or list(definitions or factories)
    if not names:
        raise ValueError("No models configured; add config.models or provide a registry")
    results = []
    for name in names:
        definition = definitions.get(name, {})
        factory = factories.get(name)
        if factory is None:
            target = definition.get("target")
            if not target:
                raise ValueError(f"No target or registered factory for model '{name}'")
            factory = import_factory(target)
        seed_everything(config.seed)
        model = _build(
            factory, definition.get("params", {}), info.vocabulary_size,
            info.num_classes, config.data.max_length,
        )
        trainer = Trainer(config.training, resolve_device(config.device), output_dir)
        # A new generator gives every architecture exactly the same minibatch order.
        results.append(trainer.fit(name, model, data.loaders()))
        write_reports(results, output_dir)
    return results
