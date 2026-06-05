# Rubric and Stage Gates

This rubric is for self-assessment, Codex review, and portfolio readiness. Use it at the end of every lesson and stage.

## Scoring scale

| Score | Meaning |
|---:|---|
| 0 | Missing or not attempted |
| 1 | Attempted but unreliable or mostly copied |
| 2 | Works partially; limited explanation or weak tests |
| 3 | Correct and reasonably explained; basic benchmarks |
| 4 | Strong implementation; good tests, benchmarks, and bottleneck analysis |
| 5 | Portfolio-quality; reproducible, profiled, well-explained, limitations documented |

## Lesson rubric

| Category | 0-1 | 2-3 | 4-5 |
|---|---|---|---|
| Correctness | no reliable reference or tests | basic CPU reference and simple tests | robust reference, edge cases, tolerances, deterministic checks |
| CUDA mental model | cannot explain mapping | explains threads/blocks for this kernel | explains memory, sync, occupancy, divergence, and tradeoffs |
| Benchmarking | no timing or misleading timing | CUDA events and one input size | warmups, sweeps, metadata, throughput, fair comparisons |
| Profiling | no profiler use | profiler output captured but shallow interpretation | metrics interpreted and tied to optimization hypotheses |
| Optimization | random changes | one plausible optimization | measured experiments with clear before/after and explanation |
| Code quality | hard to build/read | organized but rough | clean structure, comments, error handling, reproducible build |
| Documentation | notes absent | README explains basic usage | portfolio-safe writeup with limitations and benchmark context |
| Codex usage | copied final answer | asked for help/review | used Codex as tutor/reviewer; can explain code independently |

## Stage gates

### Stage 0 gate: setup

Pass when:

- environment captured or setup blockers documented;
- starter project inspected;
- you can describe host/device separation;
- you can explain the course loop.

Minimum score: 3 in documentation and Codex usage.

### Stage 1 gate: fundamentals

Pass when:

- vector add complete;
- reduction complete;
- transpose complete or attempted with notes;
- one scan/histogram lesson attempted;
- you can explain coalescing, shared memory, and synchronization.

Minimum score: average 3 across correctness, CUDA mental model, and benchmarking.

### Stage 2 gate: profiling

Pass when:

- at least one Nsight Compute profile captured;
- you can classify at least two kernels by bottleneck type;
- you have updated benchmark methodology;
- you have run one optimization experiment based on profiler evidence.

Minimum score: 3 in profiling and 3 in benchmarking.

### Stage 3 gate: inference kernels

Pass when:

- softmax complete;
- LayerNorm or RMSNorm complete;
- one fused op complete;
- at least one inference kernel has a portfolio-quality README.

Minimum score: 4 in correctness and 3 in numerical-stability explanation.

### Stage 4 gate: framework integration

Pass when:

- custom op scaffold exists;
- tests compare against framework reference where possible;
- limitations are explicit;
- you can explain Python wrapper, C++ registration, CUDA kernel, and build system.

Minimum score: 3 in code quality and documentation.

### Stage 5 gate: robotics kernels

Pass when:

- at least two robotics-oriented kernels attempted;
- CPU reference exists for each;
- one has benchmark sweep over robotics-relevant dimensions;
- limitations are documented.

Minimum score: 3 in correctness and 3 in problem framing.

### Stage 6 gate: capstone

Pass when:

- capstone builds or setup limitations are documented;
- benchmark results are reproducible;
- profiler notes or performance analysis are included;
- README is portfolio-safe;
- you can answer interview questions about tradeoffs and limitations.

Minimum score: 4 average across categories.

## Portfolio-readiness checklist

A project is portfolio-ready only if all are true:

- [ ] It has a clear problem statement.
- [ ] It explains why the kernel/workload matters.
- [ ] It includes build and run instructions.
- [ ] It includes a CPU or trusted reference implementation.
- [ ] It includes correctness tests.
- [ ] It includes benchmark methodology.
- [ ] It records hardware/software environment.
- [ ] It states input sizes.
- [ ] It avoids unsupported generalizations.
- [ ] It has a limitations section.
- [ ] It includes at least one “what I would improve next” section.

## Codex independence check

Before you mark a lesson complete, ask Codex:

```text
Quiz me on this lesson. Ask one question at a time. Focus on whether I can explain the code and performance results without relying on you. After each answer, grade it and tell me what to review.
```

You pass only when you can answer without reading the code line by line.
