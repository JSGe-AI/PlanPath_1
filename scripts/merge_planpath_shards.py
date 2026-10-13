#!/usr/bin/env python3
"""Merge disjoint patient-level evaluation shards into one prediction file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from planpath.core.io_utils import read_jsonl, write_jsonl_atomic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("inputs", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_key: dict[tuple[str, str], dict] = {}
    methods: set[str] = set()
    for path in args.inputs:
        if not path.is_file():
            raise FileNotFoundError(f"missing shard output: {path}")
        for row in read_jsonl(path):
            key = (str(row.get("patient_id")), str(row.get("qa_id")))
            if key in by_key:
                raise ValueError(f"duplicate task across shards: {key}")
            by_key[key] = row
            methods.add(str(row.get("method")))
    if len(methods) > 1:
        raise ValueError(f"shards contain multiple methods: {sorted(methods)}")
    if args.expected_count is not None and len(by_key) != args.expected_count:
        raise ValueError(
            f"merged task count {len(by_key)} != expected {args.expected_count}"
        )
    rows = [by_key[key] for key in sorted(by_key)]
    write_jsonl_atomic(args.output, rows)
    print(
        f"merged {len(rows)} predictions from {len(args.inputs)} shards into "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
