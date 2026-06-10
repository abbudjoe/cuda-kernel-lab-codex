# Benchmarks: Reduction

| Date       | GPU | CUDA | N         | Block size | Kernel+reset ms | End-to-end run ms | Approx input GB/s | Abs error | Tolerance | Correct | Notes       |
| ---------- | --- | ---- | --------- | ---------- | --------------- | ----------------: | ---------------: | --------: | --------: | ------- | ----------- |
| YYYY-MM-DD | TBD | TBD  | 1,000,000 | 256        | TBD             | TBD               | TBD              | TBD       | TBD       | yes/no  | naive atomic |

## Timing Scope

- `Kernel+reset ms` uses CUDA event timing around each timed iteration's `cudaMemset(d_output, 0)` plus the reduction kernel launch. The reset is included because each reduction requires a fresh zero output scalar.
- `End-to-end run ms` uses CPU wall-clock timing around allocation, host-to-device copy, warmups, timed iterations, result copy, and cleanup for the whole benchmark run.
- `Approx input GB/s` counts only input bytes read as `N * sizeof(float)` divided by `Kernel+reset ms`. It is a teaching metric, not a full accounting of atomic read-modify-write traffic.
