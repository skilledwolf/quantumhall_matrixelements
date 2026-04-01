#!/usr/bin/env python3
"""Generate the frozen symmetric-gauge reference dataset.

Run from the repo root:
  PYTHONPATH=src python validation/generate_symmetric_reference_data.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from symmetric_reference import REFERENCE_PATH, write_reference_data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REFERENCE_PATH,
        help="Where to write the JSON reference dataset.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    path = write_reference_data(args.output)
    print(f"Wrote symmetric reference dataset to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
