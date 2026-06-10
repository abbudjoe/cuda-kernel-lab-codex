# Benchmarks: Reduction

| Date       | GPU | CUDA | N         | Block size | Kernel+reset ms | End-to-end run ms | Approx input GB/s | Abs error | Tolerance | Correct | Notes       |
| ---------- | --- | ---- | --------- | ---------- | --------------- | ----------------: | ---------------: | --------: | --------: | ------- | ----------- |
| 2026-06-10 | Tesla T4 | 12.8 | 1 | 256 | 0.0048 | 189.2867 | 0.0008 | 0.000000 | 0.000100 | yes | naive atomic |
| 2026-06-10 | Tesla T4 | 12.8 | 2 | 256 | 0.0058 | 0.8901 | 0.0014 | 0.000000 | 0.000100 | yes | naive atomic |
| 2026-06-10 | Tesla T4 | 12.8 | 3 | 256 | 0.0055 | 0.8631 | 0.0022 | 0.000000 | 0.000100 | yes | naive atomic |
| 2026-06-10 | Tesla T4 | 12.8 | 1,000 | 256 | 0.0082 | 1.1597 | 0.4886 | 0.000004 | 0.000100 | yes | naive atomic |
| 2026-06-10 | Tesla T4 | 12.8 | 1,000,000 | 256 | 2.9422 | 330.9194 | 1.3595 | 0.002441 | 0.049986 | yes | naive atomic |
| 2026-06-10 | Tesla T4 | 12.8 | 10,000,000 | 256 | 20.0038 | 2210.3711 | 1.9996 | 0.021729 | 0.499940 | yes | naive atomic |

## Timing Scope

- `Kernel+reset ms` uses CUDA event timing around each timed iteration's `cudaMemset(d_output, 0)` plus the reduction kernel launch. The reset is included because each reduction requires a fresh zero output scalar.
- `End-to-end run ms` uses CPU wall-clock timing around allocation, host-to-device copy, warmups, timed iterations, result copy, and cleanup for the whole benchmark run.
- `Approx input GB/s` counts only input bytes read as `N * sizeof(float)` divided by `Kernel+reset ms`. It is a teaching metric, not a full accounting of atomic read-modify-write traffic.
