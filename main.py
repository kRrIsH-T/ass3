"""Project command-line entry point.

Subcommands are deliberately thin dispatchers so training remains importable from
notebooks and tests without CLI side effects.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="From-scratch sequence model benchmark")
    subcommands = parser.add_subparsers(dest="command", required=True)
    benchmark = subcommands.add_parser("benchmark", help="train and compare configured models")
    benchmark.add_argument("--config", default="configs/benchmark.yaml")
    benchmark.add_argument("--models", nargs="*")
    benchmark.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "benchmark":
        from benchmark.__main__ import cli

        forwarded = ["--config", args.config]
        if args.models is not None:
            forwarded.extend(["--models", *args.models])
        if args.smoke:
            forwarded.append("--smoke")
        return cli(forwarded)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
