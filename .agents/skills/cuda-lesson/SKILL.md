---
name: cuda-lesson
description: Use for creating or completing a CUDA course lesson with CPU reference, naive CUDA, tests, benchmarks, profiling notes, and portfolio-safe documentation. Do not use for non-CUDA tasks.
---

# CUDA Lesson Skill

Use this skill when the user asks to create, plan, review, or complete a CUDA lesson in this repo.

## Workflow

1. Read `AGENTS.md`, `RUBRIC.md`, and the target lesson README.
2. Identify the lesson objective and current state.
3. Create or update a staged plan:
   - concept;
   - CPU reference;
   - naive CUDA;
   - correctness tests;
   - benchmark;
   - profiling question;
   - optimization experiment;
   - notes/report.
4. Do not implement the final optimized kernel before the user attempts the naive version unless explicitly asked.
5. Keep naive and optimized versions side by side.
6. Require benchmarks and correctness checks before performance claims.
7. End by updating `NOTES.md` or giving the user a note update prompt.

## Lesson deliverables

- CPU/reference implementation;
- naive CUDA kernel;
- correctness test path;
- benchmark command;
- benchmark table or CSV;
- profiler notes if applicable;
- README update;
- final quiz questions.

## Review questions

Before marking done, ask:

1. What is the thread/block mapping?
2. What are the main memory reads/writes?
3. What can race?
4. What is the likely bottleneck?
5. What did the benchmark include and exclude?
6. What limitation should a portfolio reader know?
