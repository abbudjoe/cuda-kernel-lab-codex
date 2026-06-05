#!/usr/bin/env python3
"""Capture CUDA/GPU/build environment metadata for benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str]) -> dict:
    if shutil.which(cmd[0]) is None:
        return {"available": False, "command": cmd, "stdout": "", "stderr": f"{cmd[0]} not found", "returncode": None}
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return {
        "available": True,
        "command": cmd,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="profiler_reports/environment.json")
    args = parser.parse_args()

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "commands": {
            "nvidia_smi": run(["nvidia-smi"]),
            "nvcc_version": run(["nvcc", "--version"]),
            "cmake_version": run(["cmake", "--version"]),
            "gpp_version": run(["g++", "--version"]),
            "git_commit": run(["git", "rev-parse", "HEAD"]),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
