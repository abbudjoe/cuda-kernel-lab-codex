#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

inline std::vector<float> make_random_vector(std::size_t n, unsigned seed = 1234) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(n);
    for (auto& x : out) x = dist(rng);
    return out;
}

inline float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
    float max_err = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_err = std::max(max_err, std::fabs(a[i] - b[i]));
    }
    return max_err;
}

inline void print_result_line(const char* kernel,
                              const char* version,
                              std::size_t n,
                              float ms,
                              float max_err) {
    double bytes = static_cast<double>(n) * 3.0 * sizeof(float); // read a,b and write c
    double gbps = bytes / (ms / 1000.0) / 1e9;
    std::cout << std::fixed << std::setprecision(4)
              << "kernel=" << kernel
              << ", version=" << version
              << ", n=" << n
              << ", time_ms=" << ms
              << ", approx_GBps=" << gbps
              << ", max_abs_error=" << max_err
              << std::endl;
}
