from __future__ import annotations

import contextlib
import inspect
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainingConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    learning_rate: float
    seconds: float


@dataclass
class ExperimentResult:
    model: str
    parameter_count: int
    trainable_parameter_count: int
    epochs_completed: int
    training_seconds: float
    best_validation_loss: float
    best_validation_accuracy: float
    test_loss: float
    test_accuracy: float
    inference_examples_per_second: float
    peak_gpu_memory_mb: float
    device: str
    checkpoint: str
    history: list[EpochMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["history"] = [asdict(item) for item in self.history]
        return result


def _model_forward(model: nn.Module, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Adapt the shared batch to common classifier forward conventions."""
    parameters = inspect.signature(model.forward).parameters
    if "padding_mask" in parameters:
        positions = torch.arange(inputs.size(1), device=inputs.device).unsqueeze(0)
        padding_mask = positions >= lengths.unsqueeze(1)
        output = model(inputs, padding_mask=padding_mask)
    elif "lengths" in parameters:
        output = model(inputs, lengths=lengths)
    else:
        output = model(inputs)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("logits")
    if not isinstance(output, torch.Tensor) or output.ndim != 2:
        raise ValueError("Model forward must return [batch, classes] logits")
    return output


class Trainer:
    def __init__(self, config: TrainingConfig, device: torch.device, output_dir: str | Path):
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss()
        self.amp_enabled = bool(config.amp and device.type == "cuda")

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.amp_enabled else contextlib.nullcontext()

    def _run_epoch(self, model: nn.Module, loader: DataLoader, optimizer=None, scaler=None):
        training = optimizer is not None
        model.train(training)
        total_loss = total_correct = total = 0
        for inputs, labels, lengths in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), self._autocast():
                logits = _model_forward(model, inputs, lengths)
                loss = self.criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.detach().item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        if total == 0:
            raise ValueError("Loader contains no examples")
        return total_loss / total, total_correct / total

    @torch.inference_mode()
    def _inference_speed(self, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        batches = list()
        for index, batch in enumerate(loader):
            batches.append(batch)
            if index + 1 >= self.config.inference_warmup_batches + self.config.inference_batches:
                break
        if not batches:
            return 0.0

        def run(batch):
            inputs, _, lengths = batch
            inputs, lengths = inputs.to(self.device), lengths.to(self.device)
            with self._autocast():
                _model_forward(model, inputs, lengths)

        warmups = batches[: self.config.inference_warmup_batches]
        measured = batches[self.config.inference_warmup_batches :] or batches
        for batch in warmups:
            run(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        examples = 0
        for batch in measured:
            run(batch)
            examples += batch[0].size(0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return examples / max(time.perf_counter() - start, 1e-12)

    def fit(self, name: str, model: nn.Module, loaders: dict[str, DataLoader]) -> ExperimentResult:
        required = {"train", "validation", "test"}
        if not required.issubset(loaders):
            raise ValueError(f"Missing loaders: {sorted(required - set(loaders))}")
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.config.scheduler_factor, patience=self.config.scheduler_patience
        )
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        except AttributeError:  # PyTorch < 2.3
            scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"{name}.pt"
        temporary = checkpoint.with_suffix(".tmp")
        best_loss = math.inf
        best_accuracy = 0.0
        stale_epochs = 0
        history: list[EpochMetrics] = []
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        training_start = time.perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_accuracy = self._run_epoch(model, loaders["train"], optimizer, scaler)
            validation_loss, validation_accuracy = self._run_epoch(model, loaders["validation"])
            scheduler.step(validation_loss)
            history.append(
                EpochMetrics(
                    epoch, train_loss, train_accuracy, validation_loss, validation_accuracy,
                    optimizer.param_groups[0]["lr"], time.perf_counter() - epoch_start,
                )
            )
            if validation_loss < best_loss - self.config.min_delta:
                best_loss, best_accuracy, stale_epochs = validation_loss, validation_accuracy, 0
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "validation_loss": best_loss}, temporary)
                temporary.replace(checkpoint)
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break
        training_seconds = time.perf_counter() - training_start
        if not checkpoint.exists():
            raise RuntimeError("Training completed without producing a checkpoint")
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        model.load_state_dict(saved["model_state_dict"])
        test_loss, test_accuracy = self._run_epoch(model, loaders["test"])
        speed = self._inference_speed(model, loaders["test"])
        peak_memory = (
            torch.cuda.max_memory_allocated(self.device) / (1024**2) if self.device.type == "cuda" else 0.0
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        result = ExperimentResult(
            name, total_parameters, trainable_parameters, len(history), training_seconds,
            best_loss, best_accuracy, test_loss, test_accuracy, speed, peak_memory,
            str(self.device), str(checkpoint), history,
        )
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"{name}.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result
