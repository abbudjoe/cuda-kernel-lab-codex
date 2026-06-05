#include "common/cuda_check.cuh"
#include "common/cuda_timer.cuh"
#include "common/vector_utils.hpp"

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, std::size_t n) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void vector_add_cpu(const std::vector<float>& a,
                    const std::vector<float>& b,
                    std::vector<float>& c) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i];
    }
}

float run_vector_add_cuda(const std::vector<float>& h_a,
                          const std::vector<float>& h_b,
                          std::vector<float>& h_c,
                          int warmup_iters,
                          int timed_iters) {
    const std::size_t n = h_a.size();
    const std::size_t bytes = n * sizeof(float);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    for (int i = 0; i < warmup_iters; ++i) {
        vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    }
    CUDA_KERNEL_CHECK();

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < timed_iters; ++i) {
        vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaGetLastError());
    float total_ms = timer.stop_ms();

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return total_ms / static_cast<float>(timed_iters);
}

int main(int argc, char** argv) {
    std::size_t n = 1'000'000;
    if (argc > 1) {
        n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    }

    const int warmup_iters = 10;
    const int timed_iters = 100;

    std::vector<float> a = make_random_vector(n, 1234);
    std::vector<float> b = make_random_vector(n, 5678);
    std::vector<float> c_cpu(n, 0.0f);
    std::vector<float> c_cuda(n, 0.0f);

    vector_add_cpu(a, b, c_cpu);
    float ms = run_vector_add_cuda(a, b, c_cuda, warmup_iters, timed_iters);

    float err = max_abs_error(c_cpu, c_cuda);
    print_result_line("vector_add", "naive", n, ms, err);

    if (err > 1e-6f) {
        std::cerr << "Correctness check failed. max_abs_error=" << err << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
