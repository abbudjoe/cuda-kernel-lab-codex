#include "common/cuda_check.cuh"
#include "common/cuda_timer.cuh"
#include "common/vector_utils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

constexpr int kBlockSize = 256;
constexpr float kAbsTolerance = 1e-4f;
constexpr float kRelTolerance = 1e-7f;

struct ReductionResult {
    float sum;
    float kernel_ms;
    double end_to_end_ms;
};

float reduce_cpu_reference(const std::vector<float>& input) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < input.size(); ++i) {
        sum += input[i];
    }
    return sum;
}

__global__ void reduce_naive_atomic_kernel(const float* input, float* output, std::size_t n) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(output, input[idx]);
    }
}

__global__ void reduce_block_partial(const float* input, float* partial_sums, std::size_t n) {
    __shared__ float shared[kBlockSize];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        shared[tid] = input[idx];
    } else {
        shared[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

ReductionResult reduce_cuda_multipass(const std::vector<float>& h_input, 
                                      int warmup_iters, 
                                      int timed_iters) {
    auto end_to_end_start = std::chrono::steady_clock::now();
    float* d_input = nullptr;
    const std::size_t n = h_input.size();
    const std::size_t bytes = n * sizeof(float);
    const int max_blocks = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
    float* d_scratch_a = nullptr;
    float* d_scratch_b = nullptr;

    CUDA_CHECK(cudaMalloc(&d_scratch_a, max_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch_b, max_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));


    float* current_input = d_input;
    std::size_t current_n = n;
    float* current_output = d_scratch_a;

    while (current_n > 1) {
        int blocks = static_cast<int>((current_n + kBlockSize - 1) / kBlockSize);

        reduce_block_partial<<<blocks, kBlockSize>>>(
            current_input,
            current_output, 
            current_n
        );

        current_n = blocks;

        current_input = current_output;
        
        if (current_output == d_scratch_a) {
            current_output = d_scratch_b;
        } else {
            current_output = d_scratch_a;
        }
    }
    CUDA_KERNEL_CHECK();

    float h_output = 0.0f;
    
    CUDA_CHECK(cudaMemcpy(&h_output, current_input, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_scratch_a));
    CUDA_CHECK(cudaFree(d_scratch_b));

    return {h_output, 0.0f, 0.0};
}

ReductionResult reduce_cuda_block_partial_cpu_finish(const std::vector<float>& h_input,
                                                     int warmup_iters,
                                                     int timed_iters) {
    auto end_to_end_start = std::chrono::steady_clock::now();
    float* d_input = nullptr;
    const std::size_t n = h_input.size();
    const std::size_t bytes = n * sizeof(float);
    const int grid_size = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
    float* d_partial_sums = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_partial_sums, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    
    for (int i = 0; i < warmup_iters; ++i) {
        reduce_block_partial<<<grid_size, kBlockSize>>>(d_input, d_partial_sums, n);
    }
    CUDA_KERNEL_CHECK();


    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < timed_iters; ++i) {
        reduce_block_partial<<<grid_size, kBlockSize>>>(d_input, d_partial_sums, n);
    }
    CUDA_CHECK(cudaGetLastError());
    float kernel_ms = timer.stop_ms();

    std::vector<float> h_partial_sums(grid_size);
    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float sum = reduce_cpu_reference(h_partial_sums);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial_sums));

    auto end_to_end_stop = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> end_to_end_ms = end_to_end_stop - end_to_end_start;

    return {sum, kernel_ms / static_cast<float>(timed_iters), end_to_end_ms.count()};
}

ReductionResult reduce_cuda_naive(const std::vector<float>& h_input,
                                  int warmup_iters,
                                  int timed_iters) {
    auto end_to_end_start = std::chrono::steady_clock::now();
    const std::size_t n = h_input.size();
    const std::size_t bytes = n * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

    const int grid_size = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
    for (int i = 0; i < warmup_iters; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        reduce_naive_atomic_kernel<<<grid_size, kBlockSize>>>(d_input, d_output, n);
    }
    CUDA_KERNEL_CHECK();

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < timed_iters; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        reduce_naive_atomic_kernel<<<grid_size, kBlockSize>>>(d_input, d_output, n);
    }
    CUDA_CHECK(cudaGetLastError());
    float total_kernel_reset_ms = timer.stop_ms();

    float h_output = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    auto end_to_end_stop = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> end_to_end_ms = end_to_end_stop - end_to_end_start;

    return {h_output, total_kernel_reset_ms / static_cast<float>(timed_iters), end_to_end_ms.count()};
}

float sum_abs_cpu_reference(const std::vector<float>& input) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < input.size(); ++i) {
        sum += std::fabs(input[i]);
    }
    return sum;
}

bool nearly_equal(float expected, float actual, float scale) {
    float abs_error = std::fabs(expected - actual);
    float tolerance = std::max(kAbsTolerance, kRelTolerance * scale);
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

double approx_input_gbps(std::size_t n, float ms) {
    double bytes = static_cast<double>(n) * sizeof(float);
    return bytes / (ms / 1000.0) / 1e9;
}

void print_benchmark_header() {
    std::cout << "| Date       | GPU | CUDA | N         | Block size | Kernel+reset ms | End-to-end run ms | Approx input GB/s | Abs error | Tolerance | Correct | Notes       |\n"
              << "| ---------- | --- | ---- | --------- | ---------- | --------------- | ----------------: | ---------------: | --------: | --------: | ------- | ----------- |\n";
}

void print_benchmark_row(const std::string& date,
                         const std::string& gpu,
                         const std::string& cuda,
                         std::size_t n,
                         const ReductionResult& result,
                         bool correct,
                         float abs_error,
                         float tolerance,
                         const char* notes) {
    std::cout << "| " << date
              << " | " << gpu
              << " | " << cuda
              << " | " << format_count(n)
              << " | " << kBlockSize
              << " | " << std::fixed << std::setprecision(4) << result.kernel_ms
              << " | " << std::fixed << std::setprecision(4) << result.end_to_end_ms
              << " | " << std::fixed << std::setprecision(4) << approx_input_gbps(n, result.kernel_ms)
              << " | " << std::fixed << std::setprecision(6) << abs_error
              << " | " << std::fixed << std::setprecision(6) << tolerance
              << " | " << (correct ? "yes" : "no")
              << " | " << notes << " |\n";
}

int main(int argc, char** argv) {
    std::vector<std::size_t> sizes;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            sizes.push_back(static_cast<std::size_t>(std::strtoull(argv[i], nullptr, 10)));
        }
    } else {
        sizes.push_back(1'000'000);
    }

    const int warmup_iters = 10;
    const int timed_iters = 100;
    const std::string date = today_yyyy_mm_dd();
    const std::string gpu = current_gpu_name();
    const std::string cuda = cuda_runtime_version();

    print_benchmark_header();

    bool all_correct = true;
    for (std::size_t n : sizes) {
        std::vector<float> input = make_random_vector(n, 1234);
        float cpu_sum = reduce_cpu_reference(input);
        
        ReductionResult cuda_result_naive = reduce_cuda_naive(input, warmup_iters, timed_iters);
        float abs_error_naive = std::fabs(cpu_sum - cuda_result_naive.sum);

        ReductionResult cuda_result_partial = reduce_cuda_block_partial_cpu_finish(input, warmup_iters, timed_iters);
        float abs_error_partial = std::fabs(cpu_sum - cuda_result_partial.sum);

        ReductionResult cuda_result_multipass = reduce_cuda_multipass(input, warmup_iters, timed_iters);
        float abs_error_multipass = std::fabs(cpu_sum - cuda_result_multipass.sum);

        float cpu_sum_abs = sum_abs_cpu_reference(input);
        bool correct_naive = nearly_equal(cpu_sum, cuda_result_naive.sum, cpu_sum_abs);
        bool correct_partial = nearly_equal(cpu_sum, cuda_result_partial.sum, cpu_sum_abs);
        bool correct_multipass = nearly_equal(cpu_sum, cuda_result_multipass.sum, cpu_sum_abs);

        float tolerance = std::max(kAbsTolerance, kRelTolerance * cpu_sum_abs);

        print_benchmark_row(date, gpu, cuda, n, cuda_result_naive, correct_naive, abs_error_naive, tolerance, "naive atomic, reset included");
        print_benchmark_row(date, gpu, cuda, n, cuda_result_partial, correct_partial, abs_error_partial, tolerance, "block partial + CPU finish");
        all_correct = all_correct && correct_naive && correct_partial && correct_multipass;


        if (!correct_naive) {
            std::cerr << "Correctness check failed for n=" << n
                      << ". cpu_sum=" << cpu_sum
                      << ", cuda_sum=" << cuda_result_naive.sum
                      << ", abs_error=" << abs_error_naive
                      << ", tolerance=" << tolerance
                      << std::endl;
        }

        if (!correct_partial) {
            std::cout << "check=block_partial_cpu_finish"
                  << ", n=" << n
                  << ", cuda_sum=" << cuda_result_partial.sum
                  << ", abs_error=" << abs_error_partial
                  << ", correct=" << (correct_partial ? "yes" : "no")
                  << std::endl;
        }
        
        std::cout << "check=block_multipass"
                  << ", n=" << n
                  << ", cuda_sum=" << cuda_result_multipass.sum
                  << ", abs_error=" << abs_error_multipass
                  << ", correct=" << (correct_multipass ? "yes" : "no")
                  << std::endl;

    }

    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
