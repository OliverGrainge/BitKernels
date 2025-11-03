#include "bitkernels/bitlinear.h"
#include "bitkernels/types.h"
#include "../common/utils.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iostream>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace bitkernels {

// Cache line size
constexpr size_t CACHE_LINE = 64;

void prepare_weights(
    const float* weights_fp32,
    size_t N,
    size_t K,
    PackedWeights& packed_out,
    float eps
) {
    if (K % 4 != 0) {
        std::cerr << "ERROR: K must be divisible by 4. K=" << K << std::endl;
        return;
    }

    const size_t total = N * K;

    // 1. Compute abs-mean scale
    double sum_abs = 0.0;
    #ifdef USE_OPENMP
    #pragma omp parallel for reduction(+:sum_abs)
    #endif
    for (size_t i = 0; i < total; ++i) {
        sum_abs += std::abs(weights_fp32[i]);
    }
    float scale = static_cast<float>(sum_abs / double(total));
    if (scale < eps) scale = eps;

    // 2. Quantize to ternary
    int8_t* ternary = new int8_t[total];
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < total; ++i) {
        float s = weights_fp32[i] / scale;
        int r = static_cast<int>(std::lrintf(s));
        ternary[i] = static_cast<int8_t>(std::max(-1, std::min(1, r)));
    }

    // 3. Pack into 2-bit format (4:1 compression)
    const size_t packed_K = K / 4;
    const size_t packed_size = N * packed_K;
    
    // Align to cache line for better performance
    size_t aligned_size = ((packed_size + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE;
    uint8_t* packed_data = nullptr;
    
    // Use posix_memalign for better compatibility
    void* ptr = nullptr;
    if (posix_memalign(&ptr, CACHE_LINE, aligned_size) != 0) {
        std::cerr << "ERROR: Failed to allocate aligned memory" << std::endl;
        delete[] ternary;
        return;
    }
    packed_data = static_cast<uint8_t*>(ptr);
    std::memset(packed_data, 0, aligned_size);

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t n = 0; n < N; ++n) {
        const int8_t* row = ternary + n * K;
        uint8_t* dst = packed_data + n * packed_K;
        for (size_t k = 0; k < K; k += 4) {
            dst[k/4] = common::pack_ternary(row[k+0], row[k+1], row[k+2], row[k+3]);
        }
    }

    // 4. Set output
    packed_out.data = packed_data;
    packed_out.scale = scale;
    packed_out.N = N;
    packed_out.K = K;
    packed_out.packed_K = packed_K;
    packed_out._internal_data = nullptr;

    delete[] ternary;
}

}  // namespace bitkernels

