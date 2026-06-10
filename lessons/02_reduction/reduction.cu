#include "common/cuda_check.cuh"
#include "common/cuda_timer.cuh"
#include "common/vector_utils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
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
    float sum = 0.0f;
    for (std::size_t i = 0; i < input.size(); ++i) {
        sum += input[i];
    }
    return sum;
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

std::string today_yyyy_mm_dd() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
    localtime_r(&now_time, &local_time);

    std::ostringstream out;
    out << std::put_time(&local_time, "%Y-%m-%d");
    return out.str();
}

std::string current_gpu_name() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    return props.name;
}

std::string cuda_runtime_version() {
    int version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&version));

    std::ostringstream out;
    out << (version / 1000) << "." << ((version % 1000) / 10);
    return out.str();
}

std::string format_count(std::size_t value) {
    std::string out = std::to_string(value);
    for (int insert_at = static_cast<int>(out.size()) - 3; insert_at > 0; insert_at -= 3) {
        out.insert(static_cast<std::size_t>(insert_at), ",");
    }
    return out;
}

double approx_gbps(std::size_t n, float ms) {
    double bytes = static_cast<double>(n) * 3.0 * sizeof(float); // read a,b and write c
    return bytes / (ms / 1000.0) / 1e9;
}

void print_benchmark_header() {
    std::cout << "| Date       | GPU | CUDA | N         | Block size | Time ms | Approx GB/s | Correct | Notes       |\n"
              << "| ---------- | --- | ---- | --------- | ---------- | ------- | ----------- | ------- | ----------- |\n";
}

void print_benchmark_row(const std::string& date,
                         const std::string& gpu,
                         const std::string& cuda,
                         std::size_t n,
                         float ms,
                         bool correct) {
    std::cout << "| " << date
              << " | " << gpu
              << " | " << cuda
              << " | " << format_count(n)
              << " | " << kBlockSize
              << " | " << std::fixed << std::setprecision(4) << ms
              << " | " << std::fixed << std::setprecision(4) << approx_gbps(n, ms)
              << " | " << (correct ? "yes" : "no")
              << " | initial run |\n";
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
