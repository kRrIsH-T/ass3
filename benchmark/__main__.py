from __future__ import annotations

import argparse
import json

from .config import load_config
from .runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fair AG News sequence-model benchmark")
    parser.add_argument("--config", default="configs/benchmark.yaml", help="YAML experiment configuration")
    parser.add_argument("--models", nargs="*", help="Optional subset of configured model names")
    parser.add_argument("--smoke", action="store_true", help="Use the tiny offline data and one epoch")
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.smoke:
        config.data.output_dir = "results/smoke"
        config.data.force_fallback = True
        config.data.batch_size = 4
        config.training.epochs = 1
        config.training.patience = 1
        config.training.inference_batches = 1
    results = run_benchmark(config, selected=args.models)
    print(json.dumps([{k: v for k, v in result.to_dict().items() if k != "history"} for result in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
