from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    output_dir: str = "outputs"
    max_length: int = 128
    min_frequency: int = 2
    max_vocab_size: int = 30000
    validation_fraction: float = 0.1
    batch_size: int = 64
    num_workers: int = 0
    max_train_samples: int | None = 20000
    max_validation_samples: int | None = None
    max_test_samples: int | None = None
    offline_fallback: bool = True
    force_fallback: bool = False

    def __post_init__(self) -> None:
        if self.max_length < 1 or self.batch_size < 1 or self.max_vocab_size < 2:
            raise ValueError("max_length/batch_size must be positive and max_vocab_size must be at least 2")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        if self.min_frequency < 1 or self.num_workers < 0:
            raise ValueError("min_frequency must be positive and num_workers non-negative")


@dataclass
class TrainingConfig:
    epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    patience: int = 3
    min_delta: float = 1e-4
    scheduler_patience: int = 1
    scheduler_factor: float = 0.5
    amp: bool = True
    inference_warmup_batches: int = 2
    inference_batches: int = 20

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0 or self.patience < 1:
            raise ValueError("epochs, learning_rate, and patience must be positive")
        if self.gradient_clip <= 0 or not 0 < self.scheduler_factor < 1:
            raise ValueError("gradient_clip must be positive and scheduler_factor between zero and one")


@dataclass
class BenchmarkConfig:
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type, values: dict[str, Any]):
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} options: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load YAML while keeping configuration validation explicit."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency error is actionable
        raise RuntimeError("PyYAML is required to load benchmark configuration") from exc
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Benchmark config must be a mapping")
    allowed = {"seed", "device", "data", "training", "models"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown BenchmarkConfig options: {sorted(unknown)}")
    return BenchmarkConfig(
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        data=_construct(DataConfig, raw.get("data", {})),
        training=_construct(TrainingConfig, raw.get("training", {})),
        models=raw.get("models", {}),
    )
