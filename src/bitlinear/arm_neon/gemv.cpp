#include "bitkernels/bitlinear.h"
#include "bitkernels/types.h"
#include "../../common/utils.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <array>
#include <arm_neon.h>

namespace bitkernels {
namespace arm_neon {

// ============================================================================
// Fast LUT for unpacking (same as GEMM)
// ============================================================================

static auto generate_ternary_lut_gemv() {
    std::array<int8x8_t, 256> lut{};
    
    for (int i = 0; i < 256; i++) {
        int8_t w0 = ((i >> 0) & 3) == 0 ? -1 : (((i >> 0) & 3) == 1 ? 0 : 1);
        int8_t w1 = ((i >> 2) & 3) == 0 ? -1 : (((i >> 2) & 3) == 1 ? 0 : 1);
        int8_t w2 = ((i >> 4) & 3) == 0 ? -1 : (((i >> 4) & 3) == 1 ? 0 : 1);
        int8_t w3 = ((i >> 6) & 3) == 0 ? -1 : (((i >> 6) & 3) == 1 ? 0 : 1);
        
        int8_t values[8] = {w0, w1, w2, w3, 0, 0, 0, 0};
        lut[i] = vld1_s8(values);
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
    // Find abs max with NEON
    float32x4_t vmax = vdupq_n_f32(0.0f);
    size_t k = 0;
    
    for (; k + 16 <= K; k += 16) {
        float32x4_t v0 = vabsq_f32(vld1q_f32(x_fp32 + k));
        float32x4_t v1 = vabsq_f32(vld1q_f32(x_fp32 + k + 4));
        float32x4_t v2 = vabsq_f32(vld1q_f32(x_fp32 + k + 8));
        float32x4_t v3 = vabsq_f32(vld1q_f32(x_fp32 + k + 12));
        vmax = vmaxq_f32(vmax, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
    }
    
    float abs_max = vmaxvq_f32(vmax);
    for (; k < K; ++k) {
        abs_max = std::max(abs_max, std::abs(x_fp32[k]));
    }

    float scale = std::max(abs_max / 127.0f, eps);
    const float inv_scale = 1.0f / scale;

    // Quantize with NEON
    float32x4_t vinv = vdupq_n_f32(inv_scale);
    k = 0;
    
    for (; k + 16 <= K; k += 16) {
        float32x4_t x0 = vmulq_f32(vld1q_f32(x_fp32 + k), vinv);
        float32x4_t x1 = vmulq_f32(vld1q_f32(x_fp32 + k + 4), vinv);
        float32x4_t x2 = vmulq_f32(vld1q_f32(x_fp32 + k + 8), vinv);
        float32x4_t x3 = vmulq_f32(vld1q_f32(x_fp32 + k + 12), vinv);

        int32x4_t i0 = vcvtnq_s32_f32(x0);
        int32x4_t i1 = vcvtnq_s32_f32(x1);
        int32x4_t i2 = vcvtnq_s32_f32(x2);
        int32x4_t i3 = vcvtnq_s32_f32(x3);

        int16x8_t i16_01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        int16x8_t i16_23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        int8x16_t i8 = vcombine_s8(vqmovn_s16(i16_01), vqmovn_s16(i16_23));
        
        vst1q_s8(x_int8 + k, i8);
    }
    
    for (; k < K; ++k) {
        float fq = std::round(x_fp32[k] * inv_scale);
        x_int8[k] = static_cast<int8_t>(std::clamp(fq, -127.0f, 127.0f));
    }
    
    return scale;
}

// ============================================================================
// GEMV Kernel (optimized for single input vector)
// ============================================================================

#if defined(__ARM_FEATURE_DOTPROD)

static inline void compute_gemv_sdot(
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
    
    // Process 4 outputs at a time
    size_t n = 0;
    
    // Allocate weight buffers
    int8_t* w0_buf = new int8_t[K];
    int8_t* w1_buf = new int8_t[K];
    int8_t* w2_buf = new int8_t[K];
    int8_t* w3_buf = new int8_t[K];
    int8_t* w_buf = new int8_t[K];
    
    for (; n + 4 <= N; n += 4) {
        // Unpack 4 weight rows on-the-fly
        
        const uint8_t* w0_packed = packed_weights + (n + 0) * packed_K;
        const uint8_t* w1_packed = packed_weights + (n + 1) * packed_K;
        const uint8_t* w2_packed = packed_weights + (n + 2) * packed_K;
        const uint8_t* w3_packed = packed_weights + (n + 3) * packed_K;
        
        // Unpack weights - manual scalar unpacking
        for (size_t k = 0; k < K; k += 4) {
            size_t pk = k / 4;
            uint8_t b0 = w0_packed[pk];
            uint8_t b1 = w1_packed[pk];
            uint8_t b2 = w2_packed[pk];
            uint8_t b3 = w3_packed[pk];
            
            // Decode each 2-bit value
            for (int j = 0; j < 4; ++j) {
                uint8_t enc0 = (b0 >> (j * 2)) & 3;
                uint8_t enc1 = (b1 >> (j * 2)) & 3;
                uint8_t enc2 = (b2 >> (j * 2)) & 3;
                uint8_t enc3 = (b3 >> (j * 2)) & 3;
                
                w0_buf[k + j] = (enc0 == 0) ? -1 : ((enc0 == 1) ? 0 : 1);
                w1_buf[k + j] = (enc1 == 0) ? -1 : ((enc1 == 1) ? 0 : 1);
                w2_buf[k + j] = (enc2 == 0) ? -1 : ((enc2 == 1) ? 0 : 1);
                w3_buf[k + j] = (enc3 == 0) ? -1 : ((enc3 == 1) ? 0 : 1);
            }
        }
        
        // Compute dot products
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0);
        int32x4_t acc3 = vdupq_n_s32(0);
        
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            int8x16_t xv = vld1q_s8(x_int8 + k);
            acc0 = vdotq_s32(acc0, xv, vld1q_s8(w0_buf + k));
            acc1 = vdotq_s32(acc1, xv, vld1q_s8(w1_buf + k));
            acc2 = vdotq_s32(acc2, xv, vld1q_s8(w2_buf + k));
            acc3 = vdotq_s32(acc3, xv, vld1q_s8(w3_buf + k));
        }
        
        int32_t sum0 = vaddvq_s32(acc0);
        int32_t sum1 = vaddvq_s32(acc1);
        int32_t sum2 = vaddvq_s32(acc2);
        int32_t sum3 = vaddvq_s32(acc3);
        
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
        
        for (size_t k = 0; k < K; k += 4) {
            uint8_t b = w_packed[k/4];
            for (int j = 0; j < 4; ++j) {
                uint8_t enc = (b >> (j * 2)) & 3;
                w_buf[k + j] = (enc == 0) ? -1 : ((enc == 1) ? 0 : 1);
            }
        }
        
        int32x4_t acc = vdupq_n_s32(0);
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            acc = vdotq_s32(acc, vld1q_s8(x_int8 + k), vld1q_s8(w_buf + k));
        }
        
        int32_t sum = vaddvq_s32(acc);
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

#else

static inline void compute_gemv_neon(
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
    
    // Process 2 outputs at a time
    size_t n = 0;
    
    // Allocate weight buffers
    int8_t* w0_buf = new int8_t[K];
    int8_t* w1_buf = new int8_t[K];
    int8_t* w_buf = new int8_t[K];
    
    for (; n + 2 <= N; n += 2) {
        
        const uint8_t* w0_packed = packed_weights + (n + 0) * packed_K;
        const uint8_t* w1_packed = packed_weights + (n + 1) * packed_K;
        
        for (size_t k = 0; k < K; k += 4) {
            uint8_t b0 = w0_packed[k/4];
            uint8_t b1 = w1_packed[k/4];
            for (int j = 0; j < 4; ++j) {
                uint8_t enc0 = (b0 >> (j * 2)) & 3;
                uint8_t enc1 = (b1 >> (j * 2)) & 3;
                w0_buf[k + j] = (enc0 == 0) ? -1 : ((enc0 == 1) ? 0 : 1);
                w1_buf[k + j] = (enc1 == 0) ? -1 : ((enc1 == 1) ? 0 : 1);
            }
        }
        
        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = vdupq_n_s32(0);
        
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            int8x16_t xv = vld1q_s8(x_int8 + k);
            int8x16_t wv0 = vld1q_s8(w0_buf + k);
            int8x16_t wv1 = vld1q_s8(w1_buf + k);
            
            int16x8_t lo0 = vmull_s8(vget_low_s8(xv), vget_low_s8(wv0));
            int16x8_t hi0 = vmull_high_s8(xv, wv0);
            int16x8_t lo1 = vmull_s8(vget_low_s8(xv), vget_low_s8(wv1));
            int16x8_t hi1 = vmull_high_s8(xv, wv1);
            
            acc0 = vpadalq_s16(acc0, lo0);
            acc0 = vpadalq_s16(acc0, hi0);
            acc1 = vpadalq_s16(acc1, lo1);
            acc1 = vpadalq_s16(acc1, hi1);
        }
        
        int32_t sum0 = vaddvq_s32(acc0);
        int32_t sum1 = vaddvq_s32(acc1);
        
        for (; k < K; ++k) {
            int32_t xi = x_int8[k];
            sum0 += xi * w0_buf[k];
            sum1 += xi * w1_buf[k];
        }
        
        Y_fp32[n + 0] = sum0 * deq_scale;
        Y_fp32[n + 1] = sum1 * deq_scale;
    }
    
    // Handle remaining output
    for (; n < N; ++n) {
        const uint8_t* w_packed = packed_weights + n * packed_K;
        
        for (size_t k = 0; k < K; k += 4) {
            uint8_t b = w_packed[k/4];
            for (int j = 0; j < 4; ++j) {
                uint8_t enc = (b >> (j * 2)) & 3;
                w_buf[k + j] = (enc == 0) ? -1 : ((enc == 1) ? 0 : 1);
            }
        }
        
        int32x4_t acc = vdupq_n_s32(0);
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            int8x16_t xv = vld1q_s8(x_int8 + k);
            int8x16_t wv = vld1q_s8(w_buf + k);
            
            int16x8_t lo = vmull_s8(vget_low_s8(xv), vget_low_s8(wv));
            int16x8_t hi = vmull_high_s8(xv, wv);
            
            acc = vpadalq_s16(acc, lo);
            acc = vpadalq_s16(acc, hi);
        }
        
        int32_t sum = vaddvq_s32(acc);
        for (; k < K; ++k) {
            sum += int32_t(x_int8[k]) * int32_t(w_buf[k]);
        }
        
        Y_fp32[n] = sum * deq_scale;
    }
    
    // Cleanup
    delete[] w0_buf;
    delete[] w1_buf;
    delete[] w_buf;
}

#endif

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
    #if defined(__ARM_FEATURE_DOTPROD)
    compute_gemv_sdot(x_int8, packed_data, Y_fp32, N, K, packed_K, scale_x, scale_w);
    #else
    compute_gemv_neon(x_int8, packed_data, Y_fp32, N, K, packed_K, scale_x, scale_w);
    #endif
    
    delete[] x_int8;
    
    // Add bias if provided
    if (bias != nullptr) {
        size_t n = 0;
        // Vectorized bias addition
        for (; n + 4 <= N; n += 4) {
            float32x4_t y_vec = vld1q_f32(Y_fp32 + n);
            float32x4_t b_vec = vld1q_f32(bias + n);
            y_vec = vaddq_f32(y_vec, b_vec);
            vst1q_f32(Y_fp32 + n, y_vec);
        }
        
        // Handle remainder
        for (; n < N; ++n) {
            Y_fp32[n] += bias[n];
        }
    }
}

}  // namespace arm_neon
}  // namespace bitkernels

