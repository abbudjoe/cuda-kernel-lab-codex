# Lesson 01: Vector Add

## Purpose

Vector add is the “hello world” of CUDA. It teaches the host/device workflow, memory allocation, kernel launch syntax, thread indexing, bounds checks, and CUDA event timing.

## Build and run

From repo root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/lesson_01_vector_add 1000000
```

## What to study

In `vector_add.cu`, identify:

- CPU reference implementation;
- device kernel;
- host-side CUDA memory allocation;
- host-to-device copies;
- grid/block size calculation;
- kernel launch;
- correctness check;
- CUDA event timing.

## Questions

1. Why is there a bounds check inside the kernel?
2. What happens if `n` is not divisible by `block_size`?
3. What does the benchmark include and exclude?
4. Is vector add likely memory-bound or compute-bound? Why?
5. Why do we run warmup iterations?

## Done criteria

- [ ] I ran the lesson at multiple input sizes.
- [ ] I recorded results in `BENCHMARKS.md`.
- [ ] I explained the bottleneck prediction in `NOTES.md`.
- [ ] I can explain every CUDA API call used in this file.
