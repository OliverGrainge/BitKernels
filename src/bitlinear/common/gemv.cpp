#include "bitkernels/bitlinear.h"
#include "bitkernels/types.h"
#include "../../common/utils.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <array>

namespace bitkernels {
namespace common {

// ============================================================================
// Fast LUT for unpacking (same as GEMM)
// ============================================================================

static auto generate_ternary_lut_gemv() {
    std::array<std::array<int8_t, 4>, 256> lut{};
    
    for (int i = 0; i < 256; i++) {
        int8_t w0 = ((i >> 0) & 3) == 0 ? -1 : (((i >> 0) & 3) == 1 ? 0 : 1);
        int8_t w1 = ((i >> 2) & 3) == 0 ? -1 : (((i >> 2) & 3) == 1 ? 0 : 1);
        int8_t w2 = ((i >> 4) & 3) == 0 ? -1 : (((i >> 4) & 3) == 1 ? 0 : 1);
        int8_t w3 = ((i >> 6) & 3) == 0 ? -1 : (((i >> 6) & 3) == 1 ? 0 : 1);
        
        lut[i][0] = w0;
        lut[i][1] = w1;
        lut[i][2] = w2;
        lut[i][3] = w3;
    }
    
    return lut;
}

static const auto TERNARY_LUT_GEMV = generate_ternary_lut_gemv();

// ============================================================================
// Activation Quantization for single vector
// ============================================================================

static inline float quantize_activation_vector(
    const float* x_fp32,
    size_t K,
    int8_t* x_int8,
    float eps
) {
    // Find absolute maximum
    float abs_max = 0.0f;
    
    for (size_t k = 0; k < K; ++k) {
        abs_max = std::max(abs_max, std::abs(x_fp32[k]));
    }

    float scale = std::max(abs_max / 127.0f, eps);
    const float inv_scale = 1.0f / scale;

    // Quantize to int8
    for (size_t k = 0; k < K; ++k) {
        float fq = std::round(x_fp32[k] * inv_scale);
        // Clamp to [-127, 127]
        fq = std::max(-127.0f, std::min(127.0f, fq));
        x_int8[k] = static_cast<int8_t>(fq);
    }
    
    return scale;
}

// ============================================================================
// Weight unpacking helper
// ============================================================================

static inline void unpack_weight_row_gemv(
    const uint8_t* __restrict__ packed_row,
    int8_t* __restrict__ unpacked_row,
    size_t K
) {
    for (size_t k = 0; k < K; k += 4) {
        uint8_t byte = packed_row[k / 4];
        const auto& vals = TERNARY_LUT_GEMV[byte];
        
        size_t remaining = std::min(size_t(4), K - k);
        for (size_t i = 0; i < remaining; ++i) {
            unpacked_row[k + i] = vals[i];
        }
    }
}

// ============================================================================
// GEMV Kernel (optimized for single input vector)
// ============================================================================

static inline void compute_gemv_portable(
    const int8_t* __restrict__ x_int8,
    const uint8_t* __restrict__ packed_weights,
    float* __restrict__ Y_fp32,
    size_t N,
    size_t K,
    size_t packed_K,
    float scale_x,
    float scale_w
) {
    const float deq_scale = scale_x * scale_w;
    
    // Process 4 outputs at a time for better parallelism
    size_t n = 0;
    
    // Allocate weight buffers for unrolled computation
    int8_t* w0_buf = new int8_t[K];
    int8_t* w1_buf = new int8_t[K];
    int8_t* w2_buf = new int8_t[K];
    int8_t* w3_buf = new int8_t[K];
    int8_t* w_buf = new int8_t[K];
    
    for (; n + 4 <= N; n += 4) {
        // Unpack 4 weight rows
        const uint8_t* w0_packed = packed_weights + (n + 0) * packed_K;
        const uint8_t* w1_packed = packed_weights + (n + 1) * packed_K;
        const uint8_t* w2_packed = packed_weights + (n + 2) * packed_K;
        const uint8_t* w3_packed = packed_weights + (n + 3) * packed_K;
        
        // Unpack weights using LUT
        for (size_t k = 0; k < K; k += 4) {
            size_t pk = k / 4;
            const auto& vals0 = TERNARY_LUT_GEMV[w0_packed[pk]];
            const auto& vals1 = TERNARY_LUT_GEMV[w1_packed[pk]];
            const auto& vals2 = TERNARY_LUT_GEMV[w2_packed[pk]];
            const auto& vals3 = TERNARY_LUT_GEMV[w3_packed[pk]];
            
            size_t remaining = std::min(size_t(4), K - k);
            for (size_t j = 0; j < remaining; ++j) {
                w0_buf[k + j] = vals0[j];
                w1_buf[k + j] = vals1[j];
                w2_buf[k + j] = vals2[j];
                w3_buf[k + j] = vals3[j];
            }
        }
        
        // Compute dot products
        int32_t sum0 = 0;
        int32_t sum1 = 0;
        int32_t sum2 = 0;
        int32_t sum3 = 0;
        
        size_t k = 0;
        
        // Process in blocks of 16 for better cache utilization
        for (; k + 16 <= K; k += 16) {
            // Prefetch next iteration
            if (k + 32 <= K) {
                __builtin_prefetch(x_int8 + k + 16, 0, 1);
                __builtin_prefetch(w0_buf + k + 16, 0, 1);
                __builtin_prefetch(w1_buf + k + 16, 0, 1);
                __builtin_prefetch(w2_buf + k + 16, 0, 1);
                __builtin_prefetch(w3_buf + k + 16, 0, 1);
            }
            
            for (size_t kk = 0; kk < 16; ++kk) {
                int32_t xi = x_int8[k + kk];
                sum0 += xi * w0_buf[k + kk];
                sum1 += xi * w1_buf[k + kk];
                sum2 += xi * w2_buf[k + kk];
                sum3 += xi * w3_buf[k + kk];
            }
        }
        
        // Handle remainder
        for (; k < K; ++k) {
            int32_t xi = x_int8[k];
            sum0 += xi * w0_buf[k];
            sum1 += xi * w1_buf[k];
            sum2 += xi * w2_buf[k];
            sum3 += xi * w3_buf[k];
        }
        
        Y_fp32[n + 0] = sum0 * deq_scale;
        Y_fp32[n + 1] = sum1 * deq_scale;
        Y_fp32[n + 2] = sum2 * deq_scale;
        Y_fp32[n + 3] = sum3 * deq_scale;
    }
    
    // Handle remaining outputs
    for (; n < N; ++n) {
        const uint8_t* w_packed = packed_weights + n * packed_K;
        
        // Unpack weight row
        for (size_t k = 0; k < K; k += 4) {
            const auto& vals = TERNARY_LUT_GEMV[w_packed[k / 4]];
            size_t remaining = std::min(size_t(4), K - k);
            for (size_t j = 0; j < remaining; ++j) {
                w_buf[k + j] = vals[j];
            }
        }
        
        // Compute dot product
        int32_t sum = 0;
        size_t k = 0;
        
        // Process in blocks of 16
        for (; k + 16 <= K; k += 16) {
            for (size_t kk = 0; kk < 16; ++kk) {
                sum += int32_t(x_int8[k + kk]) * int32_t(w_buf[k + kk]);
            }
        }
        
        // Handle remainder
        for (; k < K; ++k) {
            sum += int32_t(x_int8[k]) * int32_t(w_buf[k]);
        }
        
        Y_fp32[n] = sum * deq_scale;
    }
    
    // Cleanup
    delete[] w0_buf;
    delete[] w1_buf;
    delete[] w2_buf;
    delete[] w3_buf;
    delete[] w_buf;
}

// ============================================================================
// GEMV Implementation (public)
// ============================================================================

void bitlinear_gemv_impl(
    const float* X_fp32,
    size_t K,
    const PackedWeights& packed_weights,
    float* Y_fp32,
    const float* bias,
    float eps
) {
    if (K != packed_weights.K) {
        std::cerr << "ERROR: Dimension mismatch. K=" << K 
                  << " but weights.K=" << packed_weights.K << std::endl;
        return;
    }

    const size_t N = packed_weights.N;
    const float scale_w = packed_weights.scale;
    const uint8_t* packed_data = packed_weights.data;
    const size_t packed_K = packed_weights.packed_K;
    
    // Allocate quantized input
    int8_t* x_int8 = new int8_t[K];
    
    // Quantize input vector
    float scale_x = quantize_activation_vector(X_fp32, K, x_int8, eps);
    
    // Compute Y = X @ W^T
    compute_gemv_portable(x_int8, packed_data, Y_fp32, N, K, packed_K, scale_x, scale_w);
    
    delete[] x_int8;
    
    // Add bias if provided
    if (bias != nullptr) {
        // Vectorized bias addition - compiler can auto-vectorize
        for (size_t n = 0; n < N; ++n) {
            Y_fp32[n] += bias[n];
        }
    }
}

}  // namespace common
}  // namespace bitkernels

