# Interview Prep Questions

## CUDA fundamentals

1. What is the difference between a thread, block, grid, and warp?
2. Why do CUDA kernels need bounds checks?
3. What does it mean for memory access to be coalesced?
4. When does shared memory help?
5. What can go wrong if `__syncthreads()` is missing?
6. What is warp divergence?
7. What is occupancy, and why is it not the only metric that matters?
8. Why can a faster-looking benchmark be misleading?

## Reductions

1. Why are reductions harder than embarrassingly parallel operations?
2. What race conditions can occur in a naive reduction?
3. How does a tree reduction reduce work?
4. Why might a reduction be memory-bound?

## Inference kernels

1. Why does softmax need numerical stabilization?
2. Why are LayerNorm/RMSNorm memory-bandwidth-sensitive?
3. When does kernel fusion help?
4. What are the risks of writing custom CUDA ops for inference?

## Robotics kernels

1. How do you map depth pixels to point-cloud points?
2. What makes point-cloud voxelization challenging on GPU?
3. How would you parallelize scoring thousands of trajectories?
4. What data-layout choices matter for robotics kernels?

## Portfolio defense

1. What did your benchmark include and exclude?
2. What is the biggest limitation of your implementation?
3. What did the profiler show?
4. What would you try next?
5. What did you implement yourself versus what Codex helped with?
