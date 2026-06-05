#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

inline void cuda_check_impl(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s returned %s\n",
                     file, line, expr, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) cuda_check_impl((expr), #expr, __FILE__, __LINE__)

#define CUDA_KERNEL_CHECK()                         \
    do {                                           \
        CUDA_CHECK(cudaGetLastError());            \
        CUDA_CHECK(cudaDeviceSynchronize());       \
    } while (0)
