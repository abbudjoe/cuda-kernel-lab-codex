# Benchmarks: Vector Add

Record your results here.


| Date       | GPU | CUDA | N         | Block size | Time ms | Approx GB/s | Correct | Notes       |
| ---------- | --- | ---- | --------- | ---------- | ------- | ----------- | ------- | ----------- |
| 2026-06-08 | Tesla T4 | 12.8 | 1,000 | 256 | 0.0034 | 3.5511 | yes | launch/timing dominated, not bandwidth-representative |
| 2026-06-08 | Tesla T4 | 12.8 | 10,000 | 256 | 0.0041 | 29.5905 | yes | still too small, not bandwidth-representative |
| 2026-06-08 | Tesla T4 | 12.8 | 100,000 | 256 | 0.0058 | 205.8291 | yes | transitional/noisy; useful as a teaching datapoint, not a headline claim |
| 2026-06-08 | Tesla T4 | 12.8 | 1,000,000 | 256 | 0.0494 | 242.6980 | yes | much more representative for this harness |
| 2026-06-08 | Tesla T4 | 12.8 | 10,000,000 | 256 | 0.4578 | 262.1175 | yes | stable representative row |
| 2026-06-08 | Tesla T4 | 12.8 | 50,000,000 | 256 | 2.2723 | 264.0452 | yes | stable representative row |

## Notes

> 2026-06-08: Small inputs are not representative for bandwidth claims because fixed launch overhead, event timing granularity, and insufficient GPU occupancy dominate the measurement. The 100,000-element case is transitional: it is useful for showing the ramp-up behavior, but larger inputs provide a more stable estimate of kernel memory bandwidth.

## Claims allowed

Only make claims that match your measurement context.

Example:

> On a Colab Tesla T4, with N=50,000,000 float32 elements and CUDA event timing excluding host-device transfer, the naive vector add kernel averaged 2.2723 ms over 100 timed iterations.
