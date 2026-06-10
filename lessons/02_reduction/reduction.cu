#include "common/cuda_check.cuh"
#include "common/vector_utils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

constexpr int kBlockSize = 256;
constexpr float kAbsTolerance = 1e-2f;
constexpr float kRelTolerance = 1e-5f;

float reduce_cpu_reference(const std::vector<float>& input) {
    // TODO 1:
    //   Write the plain CPU reference first.
    //
    // Pseudocode:
    //   float sum = 0.0f
    //   for each x in input:
    //       sum += x
    //   return sum
    //
    // After implementing this, remove the throw below.
    (void)input;
    throw std::logic_error("TODO: implement reduce_cpu_reference");
}

__global__ void reduce_naive_atomic_kernel(const float* input, float* output, std::size_t n) {
    // TODO 2:
    //   Start with the simplest correct GPU reduction, even if it is slow.
    //
    // Pseudocode:
    //   idx = blockIdx.x * blockDim.x + threadIdx.x
    //   if idx < n:
    //       atomically add input[idx] into output[0]
    //
    // Question to answer before coding:
    //   What race would happen if multiple threads wrote output[0] without
    //   an atomic operation?
    (void)input;
    (void)output;
    (void)n;
}

float reduce_cuda_naive(const std::vector<float>& h_input) {
    const std::size_t n = h_input.size();
    const std::size_t bytes = n * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

    const int grid_size = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
    reduce_naive_atomic_kernel<<<grid_size, kBlockSize>>>(d_input, d_output, n);
    CUDA_KERNEL_CHECK();

    float h_output = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return h_output;
}

bool nearly_equal(float expected, float actual) {
    float abs_error = std::fabs(expected - actual);
    float tolerance = std::max(kAbsTolerance, kRelTolerance * std::fabs(expected));
    return abs_error <= tolerance;
}

int main(int argc, char** argv) {
    std::size_t n = 1'000'000;
    if (argc > 1) {
        n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    }

    std::vector<float> input = make_random_vector(n, 1234);

    try {
        float cpu_sum = reduce_cpu_reference(input);
        float cuda_sum = reduce_cuda_naive(input);
        float abs_error = std::fabs(cpu_sum - cuda_sum);
        bool correct = nearly_equal(cpu_sum, cuda_sum);

        std::cout << "kernel=reduction, version=naive_atomic"
                  << ", n=" << n
                  << ", cpu_sum=" << cpu_sum
                  << ", cuda_sum=" << cuda_sum
                  << ", abs_error=" << abs_error
                  << ", correct=" << (correct ? "yes" : "no")
                  << std::endl;

        return correct ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::logic_error& err) {
        std::cerr << err.what() << std::endl;
        return EXIT_FAILURE;
    }
}
