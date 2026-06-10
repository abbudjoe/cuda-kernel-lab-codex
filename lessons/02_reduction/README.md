# Lesson 02: Reduction

## Purpose

A reduction collapses many input values into one output value. In this lesson, the first target is sum reduction:

```text
input[0] + input[1] + ... + input[n - 1] -> one scalar sum
```

This is harder than vector add because many threads contribute to the same result. If multiple threads update the same output location without coordination, the kernel has a race.

## Main concepts

- CPU reference implementation;
- one-thread-per-element indexing;
- data races;
- atomics as a simple correctness tool;
- shared memory and synchronization in the optimized version;
- benchmark timing scope.

## Starting point

The starter file is `reduction.cu`.

It already gives you:

- input generation;
- GPU memory allocation;
- host-to-device copy;
- one output scalar on the GPU;
- kernel launch shape;
- device-to-host copy;
- correctness comparison.

It intentionally leaves the two important pieces as TODOs:

1. `reduce_cpu_reference`
2. `reduce_naive_atomic_kernel`

Do those in order.

## Mental model

Vector add maps one thread to one independent output:

```text
thread idx -> c[idx] = a[idx] + b[idx]
```

Reduction maps many threads to one output:

```text
thread idx -> contribute input[idx] to output[0]
```

That shared output creates the central correctness problem.

## First implementation task

Implement only the CPU reference first:

```cpp
float reduce_cpu_reference(const std::vector<float>& input) {
    float sum = 0.0f;
    // TODO: loop over input and accumulate into sum
    return sum;
}
```

Then build and run:

```bash
cmake --build build -j
./build/bin/lesson_02_reduction 1000000
```

Expected behavior after only the CPU reference is implemented: the program should compile and run, but the CUDA result should still be wrong until you implement the kernel.

## Naive CUDA task

After the CPU reference works, implement the naive CUDA kernel:

```text
idx = blockIdx.x * blockDim.x + threadIdx.x
if idx < n:
    atomic add input[idx] into output[0]
```

This will not be the final optimized reduction. It is a deliberately simple first GPU version so we can get correctness before performance.

## Prediction before coding

Before writing the CUDA TODO, answer:

1. What race happens if many threads do `output[0] += input[idx]` at the same time?
2. Why might an atomic-based reduction be slow for large `n`?
3. What timing scope should we record: kernel-only, end-to-end, or both?

## First Codex prompt

```text
Read COURSE_SYLLABUS.md, RUBRIC.md, AGENTS.md, and this lesson README.
Create a staged plan for Reduction.
Do not implement the final optimized solution yet.
First explain the concept, then scaffold the smallest useful CPU reference and naive CUDA task.
```

## Done criteria

- [ ] CPU/reference implementation.
- [ ] Naive CUDA implementation.
- [ ] Correctness checks.
- [ ] Benchmark table.
- [ ] Profiler notes if applicable.
- [ ] Optimization experiment.
- [ ] Portfolio-safe explanation.
