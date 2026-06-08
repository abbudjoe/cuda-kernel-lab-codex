#include "common/cuda_check.cuh"
#include "common/cuda_timer.cuh"
#include "common/vector_utils.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

constexpr int kBlockSize = 256;
constexpr float kErrorTolerance = 1e-6f;

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

    const int grid_size = static_cast<int>((n + kBlockSize - 1) / kBlockSize);

    for (int i = 0; i < warmup_iters; ++i) {
        vector_add_kernel<<<grid_size, kBlockSize>>>(d_a, d_b, d_c, n);
    }
    CUDA_KERNEL_CHECK();

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < timed_iters; ++i) {
        vector_add_kernel<<<grid_size, kBlockSize>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaGetLastError());
    float total_ms = timer.stop_ms();

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return total_ms / static_cast<float>(timed_iters);
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
        std::vector<float> a = make_random_vector(n, 1234);
        std::vector<float> b = make_random_vector(n, 5678);
        std::vector<float> c_cpu(n, 0.0f);
        std::vector<float> c_cuda(n, 0.0f);

        vector_add_cpu(a, b, c_cpu);
        float ms = run_vector_add_cuda(a, b, c_cuda, warmup_iters, timed_iters);

        float err = max_abs_error(c_cpu, c_cuda);
        bool correct = err <= kErrorTolerance;
        print_benchmark_row(date, gpu, cuda, n, ms, correct);
        all_correct = all_correct && correct;

        if (!correct) {
            std::cerr << "Correctness check failed for n=" << n
                      << ". max_abs_error=" << err << std::endl;
        }
    }

    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
