# Benchmarking Methodology

## Rule

No performance claim without context.

Always record:

- GPU model;
- CUDA version;
- driver version;
- compiler flags;
- build type;
- git commit if available;
- input shape and dtype;
- warmup iterations;
- timed iterations;
- timing method;
- correctness status.

## Timing CUDA kernels

Use CUDA events for kernel timing when possible. Avoid timing asynchronous kernels with host wall-clock time unless you synchronize correctly and explain what you are measuring.

Basic pattern:

1. allocate and initialize inputs;
2. run correctness check;
3. warm up;
4. create CUDA start/stop events;
5. record start;
6. launch kernel repeatedly or once, depending on the benchmark design;
7. record stop;
8. synchronize stop event;
9. compute elapsed time;
10. report average and, if possible, min/median.

## Pitfalls

- Timing includes host-device copies when you only intended kernel time.
- Kernel is asynchronous and wall-clock timing omits work.
- First launch includes initialization overhead.
- Input size is too small and launch overhead dominates.
- Compiler optimizes away unused CPU work.
- Baseline comparison is unfair.
- Benchmarks use different data layouts.
- Result is not checked before timing.
- Only one input size is tested.
- Environment metadata is missing.

## Suggested benchmark table

| Date | GPU | Kernel | Version | Input | Dtype | Time ms | Throughput | Correct | Notes |
|---|---|---|---|---|---|---:|---:|---|---|

## Codex benchmark review prompt

```text
Review this benchmark table for misleading comparisons or missing context.
Tell me what I can and cannot claim from these results.
```
