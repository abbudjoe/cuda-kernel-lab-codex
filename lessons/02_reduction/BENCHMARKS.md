# Benchmarks: Reduction

| Date       | GPU | CUDA | N         | Block size | Kernel+reset ms | End-to-end run ms | Approx input GB/s | Abs error | Tolerance | Correct | Notes       |
| ---------- | --- | ---- | --------- | ---------- | --------------- | ----------------: | ---------------: | --------: | --------: | ------- | ----------- |
| 2026-06-10 | Tesla T4 | 12.8 | 1 | 256 | 0.0048 | 189.2867 | 0.0008 | 0.000000 | 0.000100 | yes | naive atomic |
| 2026-06-10 | Tesla T4 | 12.8 | 2 | 256 | 0.0058 | 0.8901 | 0.0014 | 0.000000 | 0.000100 | yes | naive atomic |
| 2026-06-10 | Tesla T4 | 12.8 | 3 | 256 | 0.0055 | 0.8631 | 0.0022 | 0.000000 | 0.000100 | yes | naive atomic |
| 2026-06-10 | Tesla T4 | 12.8 | 1,000 | 256 | 0.0082 | 1.1597 | 0.4886 | 0.000004 | 0.000100 | yes | naive atomic |
| 2026-06-10 | Tesla T4 | 12.8 | 1,000,000 | 256 | 2.9422 | 330.9194 | 1.3595 | 0.002441 | 0.049986 | yes | naive atomic |
| 2026-06-10 | Tesla T4 | 12.8 | 10,000,000 | 256 | 20.0038 | 2210.3711 | 1.9996 | 0.021729 | 0.499940 | yes | naive atomic |
| 2026-06-11 | Tesla T4 | 12.8 | 1 | 256 | 0.0040 | 162.6149 | 0.0010 | 0.000000 | 0.000100 | yes | naive atomic, reset included |
| 2026-06-11 | Tesla T4 | 12.8 | 1 | 256 | 0.0033 | 0.5824 | 0.0012 | 0.000000 | 0.000100 | yes | block partial + CPU finish |
| 2026-06-11 | Tesla T4 | 12.8 | 2 | 256 | 0.0056 | 0.9109 | 0.0014 | 0.000000 | 0.000100 | yes | naive atomic, reset included |
| 2026-06-11 | Tesla T4 | 12.8 | 2 | 256 | 0.0034 | 0.6407 | 0.0023 | 0.000000 | 0.000100 | yes | block partial + CPU finish |
0 | yes | block partial + CPU finish |
| 2026-06-11 | Tesla T4 | 12.8 | 3 | 256 | 0.0052 | 0.8488 | 0.0023 | 0.000000 | 0.000100 | yes | naive atomic, reset included |
| 2026-06-11 | Tesla T4 | 12.8 | 3 | 256 | 0.0033 | 0.5408 | 0.0036 | 0.000000 | 0.000100 | yes | block partial + CPU finish |
| 2026-06-11 | Tesla T4 | 12.8 | 1,000 | 256 | 0.0070 | 1.0251 | 0.5700 | 0.000004 | 0.000100 | yes | naive atomic, reset included |
| 2026-06-11 | Tesla T4 | 12.8 | 1,000 | 256 | 0.0036 | 0.5817 | 1.1127 | 0.000002 | 0.000100 | yes | block partial + CPU finish |
| 2026-06-11 | Tesla T4 | 12.8 | 1,000,000 | 256 | 2.3488 | 265.7457 | 1.7030 | 0.009705 | 0.049986 | yes | naive atomic, reset included |
| 2026-06-11 | Tesla T4 | 12.8 | 1,000,000 | 256 | 0.0372 | 5.4624 | 107.6704 | 0.004211 | 0.049986 | yes | block partial + CPU finish |
| 2026-06-11 | Tesla T4 | 12.8 | 10,000,000 | 256 | 19.8054 | 2188.4688 | 2.0197 | 0.014893 | 0.499940 | yes | naive atomic, reset included |
| 2026-06-11 | Tesla T4 | 12.8 | 10,000,000 | 256 | 0.3805 | 52.1039 | 105.1254 | 0.070801 | 0.499940 | yes | block partial + CPU finish |

## Timing Scope

- `Kernel+reset ms` uses CUDA event timing around each timed iteration's `cudaMemset(d_output, 0)` plus the reduction kernel launch. The reset is included because each reduction requires a fresh zero output scalar.
- `End-to-end run ms` uses CPU wall-clock timing around allocation, host-to-device copy, warmups, timed iterations, result copy, and cleanup for the whole benchmark run.
- `Approx input GB/s` counts only input bytes read as `N * sizeof(float)` divided by `Kernel+reset ms`. It is a teaching metric, not a full accounting of atomic read-modify-write traffic.
