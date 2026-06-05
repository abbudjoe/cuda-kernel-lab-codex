#!/usr/bin/env python3
"""Append a benchmark row to a lesson CSV file."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

FIELDS = [
    "date", "gpu", "cuda", "kernel", "version", "input", "time_ms",
    "throughput", "correct", "notes"
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV path, e.g. lessons/06_softmax/benchmarks.csv")
    for field in FIELDS[1:]:
        parser.add_argument(f"--{field}", default="")
    args = parser.parse_args()

    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    row = {"date": datetime.now().date().isoformat()}
    for field in FIELDS[1:]:
        row[field] = getattr(args, field)

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended benchmark row to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
