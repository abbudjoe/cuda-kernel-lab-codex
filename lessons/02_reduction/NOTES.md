# Notes: Reduction

## Prediction

- Correctness risk: Many threads update `output[0]`; plain `+=` would lose updates due to a read-modify-write race.
- Expected bottleneck for naive atomic version: Contention on one global memory address.
- Why: Every valid thread calls `atomicAdd` on the same scalar output, so updates are serialized or heavily coordinated.
- Timing scope to record: both kernel-only and end-to-end later, after correctness is stable.

## Results
- On a Colab Tesla T4, the naive atomic reduction was correct across the tested sweep, but reached only about 2.0 GB/s input throughput at N=10,000,000 under a kernel+reset timing scope. This is expected because every thread contends on one global output scalar via atomicAdd.

## Benchmark Timing Contract

- Record `Kernel+reset ms` with CUDA events around each timed iteration's output reset and kernel launch.
- Record `End-to-end run ms` with CPU wall-clock timing around allocation, copies, warmups, timed iterations, result copy, and cleanup.
- Keep the reset in the kernel timing scope for the naive atomic benchmark because every reduction needs `output[0]` reset to zero before accumulation.

## What I learned

- How reduction differs from vector add:
- What can race:
- What `atomicAdd` protects:
- Why the naive version is not the final optimized version:

## Open questions

- How much slower is one global atomic per input than a shared-memory block reduction?

## Correctness Sweep

Command:

```bash
for n in 1 2 3 1000 1000000 10000000; do
  ./build/bin/lesson_02_reduction "$n"
done
```

Result:

All tested sizes passed after changing the tolerance to compare abs(cpu_sum - cuda_sum) against max(1e-4, 1e-7 * sum(abs(input))).

The tolerance needs to scale with sum(abs(input)) rather than only abs(cpu_sum) because positive and negative random values can cancel, making the final sum small even when many floating-point additions occurred.

## Block Partial Correctness

Implemented `reduce_block_partial_kernel`, where each block reduces up to 256 inputs in shared memory and writes one partial sum to `partial_sums[blockIdx.x]`. The host copies partial sums back and finishes the reduction on CPU.

Correctness passed for N = 1, 2, 3, 1,000, 1,000,000, and 10,000,000. The block partial version changes floating-point addition order, so its result can differ slightly from both the CPU reference and the naive atomic result.