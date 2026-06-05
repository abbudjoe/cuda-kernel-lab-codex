# Profiling Checklist

## Before profiling

- [ ] Correctness tests pass.
- [ ] Benchmark is reproducible.
- [ ] Input size is representative.
- [ ] Build uses release mode.
- [ ] Debug prints are disabled.
- [ ] You have a bottleneck prediction.

## Basic bottleneck categories

| Bottleneck | Symptoms | First questions |
|---|---|---|
| Memory-bound | low arithmetic intensity, high memory traffic | Are accesses coalesced? Are we rereading data? |
| Compute-bound | high arithmetic utilization | Are we using efficient math? Is precision appropriate? |
| Launch-bound | tiny kernels, many launches | Can work be fused or batched? |
| Synchronization-bound | barriers/atomics dominate | Can we reduce sync or contention? |
| Occupancy-limited | too few active warps | Are registers/shared memory/block size limiting occupancy? |
| Divergence-heavy | branches split warps | Can data/layout reduce branch divergence? |

## Nsight Compute questions

When reviewing Nsight Compute output, ask:

1. What is the achieved occupancy?
2. What is the memory throughput?
3. Are global loads/stores coalesced?
4. Are there shared-memory bank conflicts?
5. Is warp execution efficiency low?
6. Is register pressure high?
7. Are atomics or synchronization expensive?
8. Does the source view identify a hot line?
9. Does the result match my pre-profile prediction?

## Codex profiler prompt

```text
Act as my profiler coach.
Here is my Nsight output:

[PASTE]

Answer:
1. What bottleneck is most likely?
2. Which metric supports that?
3. What is one alternative explanation?
4. What experiment would distinguish them?
5. What should I change first?
```
