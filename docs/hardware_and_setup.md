# Hardware and Setup Guide

## Minimum practical setup

You need an NVIDIA GPU to run CUDA kernels locally. You can still study the repo and write code without one, but you cannot validate CUDA performance without CUDA-capable hardware.

Recommended:

- Linux or WSL2;
- NVIDIA GPU;
- compatible NVIDIA driver;
- CUDA Toolkit;
- CMake 3.24+;
- C++17-capable compiler;
- Python 3.10+;
- Nsight Compute;
- Nsight Systems.

## First environment commands

```bash
nvidia-smi
nvcc --version
cmake --version
g++ --version
python3 --version
```

Capture a machine-readable environment report:

```bash
python3 scripts/record_env.py --output profiler_reports/environment.json
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run starter lesson

```bash
./build/bin/lesson_01_vector_add 1000000
```

## Common setup blockers

| Symptom | Possible cause |
|---|---|
| `nvcc: command not found` | CUDA toolkit not installed or not on PATH |
| `nvidia-smi` fails | driver issue or no NVIDIA GPU visible |
| CMake cannot find CUDA compiler | `CUDACXX` not set or toolkit path issue |
| kernel launch fails | driver/runtime mismatch, invalid device code, or bad launch parameters |

## Codex setup prompt

```text
Help me debug my CUDA setup. I ran these commands:

[PASTE OUTPUT]

Explain what is working, what is missing, and the next smallest setup step. Do not assume I need to reinstall everything unless the output supports it.
```
