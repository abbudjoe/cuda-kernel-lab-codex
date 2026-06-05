# CUDA Kernel Lab with Codex

A standalone, repo-driven course for learning CUDA kernel development for AI inference and robotics-oriented workloads with Codex acting as tutor, lab partner, profiler coach, and reviewer.

This project is intentionally structured as a **learning lab**, not a polished library. Each lesson should produce evidence of skill:

- a CPU reference implementation;
- a naive CUDA implementation;
- at least one optimized CUDA implementation after you understand the bottleneck;
- correctness tests;
- benchmark results;
- profiler notes when relevant;
- a short technical explanation you can show in a portfolio.

## Intended career direction

This repo is designed for someone moving toward:

- scientific software / lab automation / LIMS engineering;
- ML systems and inference engineering;
- GPU performance engineering;
- robotics perception, simulation, and motion-planning acceleration;
- research engineering support roles.

The near-term career wedge can be lab automation or scientific software. The long-term moat is GPU-accelerated AI and robotics systems.

## How to use this repo with Codex

1. Open this folder in your terminal or IDE.
2. Start Codex from the repo root.
3. Codex should read `AGENTS.md` automatically if your Codex environment supports project instructions.
4. Begin with `COURSE_SYLLABUS.md`.
5. For each lesson, use the matching prompts in `CODEX_PROMPT_LIBRARY.md` and the lesson-level `PROMPTS.md`.
6. Do not let Codex write the final optimized solution before you attempt the naive implementation.

Suggested first Codex prompt:

```text
Read AGENTS.md, COURSE_SYLLABUS.md, RUBRIC.md, and docs/codex_workflow.md.
Then act as my CUDA tutor for Stage 0. Do not implement anything yet.
First inspect the repo structure, summarize how this course works, and give me my first setup task.
```

## Repo structure

```text
.
├── AGENTS.md                         # durable project instructions for Codex
├── COURSE_SYLLABUS.md                # 24-week course plan
├── RUBRIC.md                         # grading rubric and stage gates
├── CODEX_PROMPT_LIBRARY.md           # prompts for tutoring, implementation, profiling, review, and portfolio writing
├── CMakeLists.txt                    # starter CUDA/CMake project
├── Makefile                          # convenience commands
├── .agents/skills/                   # optional Codex skills for repeatable workflows
├── .codex/config.toml                # optional project-scoped Codex config placeholder
├── docs/                             # methodology and templates
├── include/common/                   # shared CUDA utility headers
├── lessons/                          # lesson workspaces
└── scripts/                          # helper scripts for environment capture, lesson creation, and reports
```

## Hardware and software assumptions

You will eventually need an NVIDIA GPU and CUDA toolkit for the CUDA parts. The repo can still be read and edited without CUDA, but compilation and profiling require a CUDA-capable machine.

Recommended baseline:

- Linux or WSL2 on Windows;
- NVIDIA GPU;
- recent NVIDIA driver;
- CUDA Toolkit;
- CMake 3.24+;
- C++17 compiler;
- Python 3.10+;
- Nsight Compute and Nsight Systems for profiling.

See `docs/hardware_and_setup.md` for details.

## Course rule

The main rule is simple:

> No performance claim without a correctness test and a benchmark.

A lesson is not done until you can explain the bottleneck in your own words.

## Quickstart

```bash
# From repo root
python3 scripts/record_env.py --output profiler_reports/environment.json
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/lesson_01_vector_add 1000000
```

If this fails because CUDA is not installed, that is okay for day one. Ask Codex to help you interpret the error and set up your environment.
