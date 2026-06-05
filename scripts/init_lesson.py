#!/usr/bin/env python3
"""Create a new lesson folder from lessons/_template."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lesson", help="Lesson folder name, e.g. 14_my_kernel")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    template = root / "lessons" / "_template"
    dest = root / "lessons" / args.lesson

    if dest.exists():
        raise SystemExit(f"Destination already exists: {dest}")

    shutil.copytree(template, dest)

    if args.title:
        readme = dest / "README.md"
        text = readme.read_text(encoding="utf-8")
        text = text.replace("Lesson NN: Title", f"Lesson {args.lesson}: {args.title}")
        readme.write_text(text, encoding="utf-8")

    print(f"Created {dest}")
    print("Next: add_subdirectory(...) in root CMakeLists.txt when ready to build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
