# Portfolio Guide

## What makes a CUDA project credible

A credible project is not just fast code. It shows that you can reason across correctness, hardware, measurement, and tradeoffs.

Include:

- problem statement;
- why the kernel/workload matters;
- architecture or data-flow diagram if helpful;
- CPU/reference implementation;
- CUDA implementation versions;
- correctness tests;
- benchmark methodology;
- profiler findings;
- measured results;
- limitations;
- next steps.

## Avoid overclaiming

Bad:

> My kernel is 100x faster.

Better:

> On [GPU], for [input shape], under [benchmark method], the optimized kernel measured [X] ms versus [Y] ms for the naive CUDA version. This comparison excludes host-device transfer time.

## Career-positioning language

For scientific software / lab automation:

> Built reproducible GPU benchmarking workflows and CUDA kernels for data-processing workloads, emphasizing correctness, measurement, and technical documentation.

For ML systems:

> Implemented and benchmarked inference-relevant CUDA kernels such as softmax/RMSNorm, with CPU references, correctness tests, and profiler-guided optimization notes.

For robotics software:

> Developed CUDA prototypes for robotics-oriented workloads including point-cloud preprocessing and batched trajectory scoring, with CPU baselines and latency-focused benchmarks.

## Demo script structure

1. State the problem.
2. Explain why CPU or naive GPU is insufficient.
3. Show correctness test.
4. Show benchmark result.
5. Explain bottleneck and optimization.
6. State limitations honestly.
7. Explain what you would improve next.
