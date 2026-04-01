#!/usr/bin/env python3
"""Validate production symmetric-gauge outputs against frozen reference data.

Run from the repo root:
  PYTHONPATH=src python validation/validate_symmetric_reference.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from symmetric_reference import REFERENCE_PATH, validate_reference_data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REFERENCE_PATH,
        help="Path to the frozen JSON reference dataset.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    results = validate_reference_data(args.input)

    print("Symmetric frozen-reference validation:")
    failed = False
    for item in results:
        status = "PASS" if item["ok"] else "FAIL"
        print(
            f"  [{status}] {item['section']}:{item['label']}  "
            f"abs_err={item['abs_err']:.3e}  rel_err={item['rel_err']:.3e}"
        )
        failed = failed or (not item["ok"])

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
