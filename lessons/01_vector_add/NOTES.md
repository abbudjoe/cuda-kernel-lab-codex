# Notes: Vector Add

## Prediction before benchmark

- Expected bottleneck: Small N will show unstable/low bandwidth because launch overhead and underfilled GPU dominate. Large N should stabilize because the kernel has enough work to saturate memory bandwidth better.
- Why: fixed launch overhead, event timing granularity, and insufficient GPU occupancy dominate the measurement
- Metric that would support this: time_ms

## Results

- Command:

```bash
for n in 1000 10000 100000 1000000 10000000 50000000; do
  ./build/bin/lesson_01_vector_add "$n"
done
```

- Output:

```text
kernel=vector_add, version=naive, n=1000, time_ms=0.0041, approx_GBps=2.9586, max_abs_error=0.0000
kernel=vector_add, version=naive, n=10000, time_ms=0.0038, approx_GBps=31.6109, max_abs_error=0.0000
kernel=vector_add, version=naive, n=100000, time_ms=0.0041, approx_GBps=295.5083, max_abs_error=0.0000
kernel=vector_add, version=naive, n=1000000, time_ms=0.0485, approx_GBps=247.5705, max_abs_error=0.0000
kernel=vector_add, version=naive, n=10000000, time_ms=0.4651, approx_GBps=258.0297, max_abs_error=0.0000
kernel=vector_add, version=naive, n=50000000, time_ms=2.2943, approx_GBps=261.5212, max_abs_error=0.0000
```

## What I learned

1. What data lives on the CPU vs GPU?
    Input vectors `a` and `b` are created in CPU memory. GPU buffers `d_a`, `d_b`, and `d_c` are allocated with `cudaMalloc`. The input data is copied into `d_a` and `d_b`, the GPU kernel writes results into `d_c`, and `d_c` is copied back into CPU vector `c_cuda`.
2. What does cudaMemcpyHostToDevice move?
    `cudaMemcpyHostToDevice` copies bytes from host memory, such as `h_a.data()`, into device/global GPU memory, such as `d_a`.
3. How does blockIdx.x * blockDim.x + threadIdx.x map a thread to an element?
    `blockIdx.x * blockDim.x + threadIdx.x` computes a global linear thread index. `blockIdx.x * blockDim.x` gives the starting element for the block, and `threadIdx.x` selects the thread's offset within that block.
4. Why is vector add memory-bound for large N?
    Vector add is memory bound because each element performs one floating-point add while reading two floats and writing one float. This is all done using global memory, and so global memory bandwidth dominates, as we are testing how fast we can move data to/from GPU DRAM, while the actual number of computations per data movement are on the lower side.
5. Why are small N benchmark rows not representative?
    - Too little work to fill many streaming multiprocessors (SMs)
    - fixed kernel launch overhead
    - event timing noise relative to the work
    - memory system never reaches steady-state streaming behavior
6. What does the reported time include?
    The reported `time_ms` measures only the timed kernel loop. Host-to-device copies happen before `timer.start()`, and the device-to-host copy happens after `timer.stop_ms()`.

## Open questions

- How should benchmark output distinguish kernel-only timing from end-to-end timing in future lessons?
