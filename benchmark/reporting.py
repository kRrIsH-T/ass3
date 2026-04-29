from __future__ import annotations

import csv
import json
from pathlib import Path

from .engine import ExperimentResult


SUMMARY_FIELDS = (
    "model", "parameter_count", "epochs_completed", "training_seconds",
    "best_validation_loss", "best_validation_accuracy", "test_loss", "test_accuracy",
    "inference_examples_per_second", "peak_gpu_memory_mb", "device",
)


def write_reports(results: list[ExperimentResult], output_dir: str | Path) -> None:
    root = Path(output_dir)
    table_dir, plot_dir = root / "tables", root / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = [{field: result.to_dict()[field] for field in SUMMARY_FIELDS} for result in results]
    with (table_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    header = "| " + " | ".join(SUMMARY_FIELDS) + " |"
    separator = "| " + " | ".join("---" for _ in SUMMARY_FIELDS) + " |"
    markdown = [header, separator]
    for row in rows:
        markdown.append("| " + " | ".join(_format(row[field]) for field in SUMMARY_FIELDS) + " |")
    (table_dir / "benchmark.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (root / "benchmark.json").write_text(json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")
    _plot(results, plot_dir)


def _format(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _plot(results: list[ExperimentResult], plot_dir: Path) -> None:
    try:
        import matplotlib

        # Avoid GUI backends on headless servers and cloud runners.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # tables remain useful in minimal/offline environments
        return
    for result in results:
        epochs = [metric.epoch for metric in result.history]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, [m.train_loss for m in result.history], label="train")
        axes[0].plot(epochs, [m.validation_loss for m in result.history], label="validation")
        axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{result.model}: loss")
        axes[0].legend()
        axes[1].plot(epochs, [m.train_accuracy for m in result.history], label="train")
        axes[1].plot(epochs, [m.validation_accuracy for m in result.history], label="validation")
        axes[1].set(xlabel="Epoch", ylabel="Accuracy", title=f"{result.model}: accuracy", ylim=(0, 1))
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(plot_dir / f"{result.model}_curves.png", dpi=160)
        plt.close(figure)
    if results:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar([r.model for r in results], [r.test_accuracy for r in results])
        axis.set(ylabel="Test accuracy", title="Architecture comparison", ylim=(0, 1))
        figure.tight_layout()
        figure.savefig(plot_dir / "test_accuracy.png", dpi=160)
        plt.close(figure)
