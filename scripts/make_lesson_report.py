#!/usr/bin/env python3
"""Combine a lesson's README, notes, and benchmarks into one markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lesson_dir", help="Path to lesson directory")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    lesson = Path(args.lesson_dir)
    if not lesson.exists():
        raise SystemExit(f"Lesson directory not found: {lesson}")

    parts = []
    for name in ["README.md", "NOTES.md", "BENCHMARKS.md"]:
        path = lesson / name
        if path.exists():
            parts.append(f"\n\n<!-- {name} -->\n\n" + path.read_text(encoding="utf-8"))

    output = Path(args.output) if args.output else lesson / "REPORT.md"
    output.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
