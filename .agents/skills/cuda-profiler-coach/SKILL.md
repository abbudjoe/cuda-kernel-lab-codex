---
name: cuda-profiler-coach
description: Use when interpreting CUDA benchmark or Nsight profiler output and proposing measured optimization experiments.
---

# CUDA Profiler Coach Skill

Use this skill for CUDA performance analysis.

## Required inputs

Ask for or inspect:

- kernel name;
- input size/shape;
- benchmark command;
- GPU model;
- timing method;
- profiler output or metrics;
- user's bottleneck prediction.

## Analysis steps

1. Verify correctness was checked before profiling.
2. Clarify what the timing includes and excludes.
3. Classify likely bottleneck:
   - memory-bound;
   - compute-bound;
   - launch-bound;
   - synchronization-bound;
   - occupancy-limited;
   - divergence-heavy;
   - atomics/contention-limited.
4. Cite the specific metric or observation supporting the classification.
5. State one alternative explanation.
6. Propose exactly two optimization experiments.
7. For each experiment, define:
   - hypothesis;
   - code change;
   - metric expected to move;
   - risk/tradeoff;
   - benchmark needed.
8. Do not claim success until results are measured.

## Output format

Use:

```text
Likely bottleneck:
Evidence:
Alternative explanation:
Experiment 1:
Experiment 2:
What to measure next:
```
