#include "../bitlinear.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <array>
#include <memory>
#include <arm_neon.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace bitlinear {

// ============================================================================
// Configuration
// ============================================================================

// Tiling parameters - tune these based on your cache sizes
constexpr size_t M_TILE = 8;   // Process 8 input rows together
constexpr size_t N_TILE = 32;  // Process 32 output channels together
constexpr size_t K_TILE = 256; // Process K dimension in chunks (for very large K)

// Cache line size (typical ARM is 64 bytes)
constexpr size_t CACHE_LINE = 64;

// ============================================================================
// Helpers
// ============================================================================

inline uint8_t pack_ternary(int8_t t0, int8_t t1, int8_t t2, int8_t t3) {
    auto enc = [](int8_t t) -> uint8_t {
        return t < 0 ? 0u : (t == 0 ? 1u : 2u);
    };
    return enc(t0) | (enc(t1) << 2) | (enc(t2) << 4) | (enc(t3) << 6);
}

// Fast LUT-based unpacking - each LUT entry is a NEON vector with 4 unpacked values
static auto generate_ternary_lut() {
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

static const auto TERNARY_LUT = generate_ternary_lut();

// Optimized unpacking that processes 32 values at once
static inline void unpack_weight_row_fast(
    const uint8_t* __restrict__ packed_row,
    int8_t* __restrict__ unpacked_row,
    size_t K
) {
    size_t k = 0;
    
    // Process 32 values at a time (8 packed bytes)
    for (; k + 32 <= K; k += 32) {
        size_t packed_idx = k / 4;
        
        // Prefetch next cache line
        __builtin_prefetch(packed_row + packed_idx + 8, 0, 1);
        
        // Load 8 packed bytes
        uint64_t packed8 = *reinterpret_cast<const uint64_t*>(packed_row + packed_idx);
        
        // Unpack using LUT - process 4 bytes in parallel
        for (int i = 0; i < 8; ++i) {
            uint8_t byte = (packed8 >> (i * 8)) & 0xFF;
            int8x8_t unpacked = TERNARY_LUT[byte];
            vst1_s8(unpacked_row + k + i * 4, unpacked);
        }
    }
    
    // Process 16 values at a time
    for (; k + 16 <= K; k += 16) {
        size_t packed_idx = k / 4;
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(packed_row + packed_idx);
        
        for (int i = 0; i < 4; ++i) {
            uint8_t byte = (packed4 >> (i * 8)) & 0xFF;
            int8x8_t unpacked = TERNARY_LUT[byte];
            vst1_s8(unpacked_row + k + i * 4, unpacked);
        }
    }
    
    // Handle remainder
    for (; k < K; k += 4) {
        int8x8_t w = TERNARY_LUT[packed_row[k/4]];
        vst1_s8(unpacked_row + k, w);
    }
}

// Batch unpacking for multiple rows with prefetching
static inline void unpack_weight_tile(
    const uint8_t* __restrict__ packed_data,
    int8_t* __restrict__ unpacked_tile,
    size_t start_n,
    size_t end_n,
    size_t K,
    size_t packed_K
) {
    #pragma omp parallel for if(end_n - start_n > 4)
    for (size_t n = start_n; n < end_n; ++n) {
        const uint8_t* packed_row = packed_data + n * packed_K;
        int8_t* unpacked_row = unpacked_tile + (n - start_n) * K;
        
        // Prefetch the entire packed row
        for (size_t pk = 0; pk < packed_K; pk += CACHE_LINE) {
            __builtin_prefetch(packed_row + pk, 0, 3);
        }
        
        unpack_weight_row_fast(packed_row, unpacked_row, K);
    }
}

// ============================================================================
// Weight Preparation (pure packing - no unpacked storage)
// ============================================================================

void prepare_weights(
    const float* weights_fp32,
    size_t N,
    size_t K,
    PackedTernaryWeights& packed_out,
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
            dst[k/4] = pack_ternary(row[k+0], row[k+1], row[k+2], row[k+3]);
        }
    }

    // 4. Set output (packed data only - no unpacked storage)
    packed_out.data = packed_data;
    packed_out.scale = scale;
    packed_out.N = N;
    packed_out.K = K;
    packed_out.packed_K = packed_K;
    packed_out._internal_unpacked = nullptr;  // No unpacked storage

    delete[] ternary;
}

// ============================================================================
// Activation Quantization
// ============================================================================

static inline void quantize_activations_tile(
    const float* x_fp32,
    size_t M_tile,
    size_t K,
    int8_t* x_int8,
    float* scales_out,
    float eps
) {
    #pragma omp parallel for if(M_tile > 2)
    for (size_t m = 0; m < M_tile; ++m) {
        const float* x_row = x_fp32 + m * K;
        int8_t* x_row_int8 = x_int8 + m * K;
        
        // Find abs max with NEON
        float32x4_t vmax = vdupq_n_f32(0.0f);
        size_t k = 0;
        
        for (; k + 16 <= K; k += 16) {
            float32x4_t v0 = vabsq_f32(vld1q_f32(x_row + k));
            float32x4_t v1 = vabsq_f32(vld1q_f32(x_row + k + 4));
            float32x4_t v2 = vabsq_f32(vld1q_f32(x_row + k + 8));
            float32x4_t v3 = vabsq_f32(vld1q_f32(x_row + k + 12));
            vmax = vmaxq_f32(vmax, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
        }
        
        float abs_max = vmaxvq_f32(vmax);
        for (; k < K; ++k) {
            abs_max = std::max(abs_max, std::abs(x_row[k]));
        }

        float scale = std::max(abs_max / 127.0f, eps);
        scales_out[m] = scale;
        const float inv_scale = 1.0f / scale;

        // Quantize with NEON
        float32x4_t vinv = vdupq_n_f32(inv_scale);
        k = 0;
        
        for (; k + 16 <= K; k += 16) {
            float32x4_t x0 = vmulq_f32(vld1q_f32(x_row + k), vinv);
            float32x4_t x1 = vmulq_f32(vld1q_f32(x_row + k + 4), vinv);
            float32x4_t x2 = vmulq_f32(vld1q_f32(x_row + k + 8), vinv);
            float32x4_t x3 = vmulq_f32(vld1q_f32(x_row + k + 12), vinv);

            int32x4_t i0 = vcvtnq_s32_f32(x0);
            int32x4_t i1 = vcvtnq_s32_f32(x1);
            int32x4_t i2 = vcvtnq_s32_f32(x2);
            int32x4_t i3 = vcvtnq_s32_f32(x3);

            int16x8_t i16_01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            int16x8_t i16_23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            int8x16_t i8 = vcombine_s8(vqmovn_s16(i16_01), vqmovn_s16(i16_23));
            
            vst1q_s8(x_row_int8 + k, i8);
        }
        
        for (; k < K; ++k) {
            float fq = std::round(x_row[k] * inv_scale);
            x_row_int8[k] = static_cast<int8_t>(std::clamp(fq, -127.0f, 127.0f));
        }
    }
}

// ============================================================================
// Optimized Tile GEMM Kernels
// ============================================================================

#if defined(__ARM_FEATURE_DOTPROD)

// Compute M_tile x N_tile outputs using SDOT
static inline void compute_tile_gemm_sdot(
    const int8_t* __restrict__ x_tile,      // M_tile x K
    const int8_t* __restrict__ w_tile,      // N_tile x K  
    float* __restrict__ Y_fp32,
    size_t m0,                              // Starting M index in global matrix
    size_t n0,                              // Starting N index in global matrix
    size_t M_tile_actual,
    size_t N_tile_actual,
    size_t K,
    size_t N,
    const float* scales_x,
    float scale_w
) {
    // Process 4 outputs at a time for each input
    for (size_t m = 0; m < M_tile_actual; ++m) {
        const int8_t* x_row = x_tile + m * K;
        const float deq_scale = scales_x[m] * scale_w;
        float* y_row = Y_fp32 + (m0 + m) * N + n0;  // Correct global indexing
        
        size_t n = 0;
        
        // Process 4 outputs at a time
        for (; n + 4 <= N_tile_actual; n += 4) {
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            
            const int8_t* w0 = w_tile + (n + 0) * K;
            const int8_t* w1 = w_tile + (n + 1) * K;
            const int8_t* w2 = w_tile + (n + 2) * K;
            const int8_t* w3 = w_tile + (n + 3) * K;
            
            // Compute dot products
            size_t k = 0;
            for (; k + 16 <= K; k += 16) {
                int8x16_t xv = vld1q_s8(x_row + k);
                
                // Prefetch next iteration
                if (k + 16 < K) {
                    __builtin_prefetch(x_row + k + 16, 0, 1);
                    __builtin_prefetch(w0 + k + 16, 0, 1);
                    __builtin_prefetch(w1 + k + 16, 0, 1);
                    __builtin_prefetch(w2 + k + 16, 0, 1);
                    __builtin_prefetch(w3 + k + 16, 0, 1);
                }
                
                acc0 = vdotq_s32(acc0, xv, vld1q_s8(w0 + k));
                acc1 = vdotq_s32(acc1, xv, vld1q_s8(w1 + k));
                acc2 = vdotq_s32(acc2, xv, vld1q_s8(w2 + k));
                acc3 = vdotq_s32(acc3, xv, vld1q_s8(w3 + k));
            }
            
            int32_t sum0 = vaddvq_s32(acc0);
            int32_t sum1 = vaddvq_s32(acc1);
            int32_t sum2 = vaddvq_s32(acc2);
            int32_t sum3 = vaddvq_s32(acc3);
            
            // Handle remainder
            for (; k < K; ++k) {
                int32_t xi = x_row[k];
                sum0 += xi * w0[k];
                sum1 += xi * w1[k];
                sum2 += xi * w2[k];
                sum3 += xi * w3[k];
            }
            
            // Dequantize and store
            y_row[n + 0] = sum0 * deq_scale;
            y_row[n + 1] = sum1 * deq_scale;
            y_row[n + 2] = sum2 * deq_scale;
            y_row[n + 3] = sum3 * deq_scale;
        }
        
        // Handle remaining outputs
        for (; n < N_tile_actual; ++n) {
            int32x4_t acc = vdupq_n_s32(0);
            const int8_t* w = w_tile + n * K;
            
            size_t k = 0;
            for (; k + 16 <= K; k += 16) {
                acc = vdotq_s32(acc, vld1q_s8(x_row + k), vld1q_s8(w + k));
            }
            
            int32_t sum = vaddvq_s32(acc);
            for (; k < K; ++k) {
                sum += int32_t(x_row[k]) * int32_t(w[k]);
            }
            
            y_row[n] = sum * deq_scale;
        }
    }
}

#else

// Compute M_tile x N_tile outputs using NEON vmull
static inline void compute_tile_gemm_neon(
    const int8_t* __restrict__ x_tile,      // M_tile x K
    const int8_t* __restrict__ w_tile,      // N_tile x K  
    float* __restrict__ Y_fp32,
    size_t m0,                              // Starting M index in global matrix
    size_t n0,                              // Starting N index in global matrix
    size_t M_tile_actual,
    size_t N_tile_actual,
    size_t K,
    size_t N,
    const float* scales_x,
    float scale_w
) {
    // Process each input row
    for (size_t m = 0; m < M_tile_actual; ++m) {
        const int8_t* x_row = x_tile + m * K;
        const float deq_scale = scales_x[m] * scale_w;
        float* y_row = Y_fp32 + (m0 + m) * N + n0;  // Correct global indexing
        
        size_t n = 0;
        
        // Process 2 outputs at a time for better ILP
        for (; n + 2 <= N_tile_actual; n += 2) {
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            
            const int8_t* w0 = w_tile + (n + 0) * K;
            const int8_t* w1 = w_tile + (n + 1) * K;
            
            size_t k = 0;
            for (; k + 16 <= K; k += 16) {
                int8x16_t xv = vld1q_s8(x_row + k);
                int8x16_t wv0 = vld1q_s8(w0 + k);
                int8x16_t wv1 = vld1q_s8(w1 + k);
                
                // Prefetch next iteration with bounds checking
                if (k + 16 < K) {
                    __builtin_prefetch(x_row + k + 16, 0, 1);
                    __builtin_prefetch(w0 + k + 16, 0, 1);
                    __builtin_prefetch(w1 + k + 16, 0, 1);
                }
                
                // Multiply and accumulate
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
            
            // Handle remainder
            for (; k < K; ++k) {
                int32_t xi = x_row[k];
                sum0 += xi * w0[k];
                sum1 += xi * w1[k];
            }
            
            // Dequantize and store
            y_row[n + 0] = sum0 * deq_scale;
            y_row[n + 1] = sum1 * deq_scale;
        }
        
        // Handle remaining output
        for (; n < N_tile_actual; ++n) {
            int32x4_t acc = vdupq_n_s32(0);
            const int8_t* w = w_tile + n * K;
            
            size_t k = 0;
            for (; k + 16 <= K; k += 16) {
                int8x16_t xv = vld1q_s8(x_row + k);
                int8x16_t wv = vld1q_s8(w + k);
                
                int16x8_t lo = vmull_s8(vget_low_s8(xv), vget_low_s8(wv));
                int16x8_t hi = vmull_high_s8(xv, wv);
                
                acc = vpadalq_s16(acc, lo);
                acc = vpadalq_s16(acc, hi);
            }
            
            int32_t sum = vaddvq_s32(acc);
            for (; k < K; ++k) {
                sum += int32_t(x_row[k]) * int32_t(w[k]);
            }
            
            y_row[n] = sum * deq_scale;
        }
    }
}

#endif

// ============================================================================
// Linear Forward Pass (Tiled with optimized unpacking)
// ============================================================================

void linear(
    const float* X_fp32,
    size_t M,
    size_t K,
    const PackedTernaryWeights& packed_weights,
    float* Y_fp32,
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
    
    // Zero output matrix
    std::memset(Y_fp32, 0, M * N * sizeof(float));

    // Process in tiles to maximize weight reuse and minimize unpacking overhead
    #pragma omp parallel
    {
        // Thread-local buffers for tiles
        // Use posix_memalign for better compatibility
        void* x_tile_ptr = nullptr;
        void* w_tile_ptr = nullptr;
        void* scales_x_ptr = nullptr;
        
        if (posix_memalign(&x_tile_ptr, 64, M_TILE * K + 64) != 0 ||
            posix_memalign(&w_tile_ptr, 64, N_TILE * K + 64) != 0 ||
            posix_memalign(&scales_x_ptr, 64, M_TILE * sizeof(float)) != 0) {
            std::cerr << "ERROR: Failed to allocate aligned memory in worker thread" << std::endl;
            std::abort();
        }
        
        int8_t* x_tile = static_cast<int8_t*>(x_tile_ptr);
        int8_t* w_tile = static_cast<int8_t*>(w_tile_ptr);
        float* scales_x = static_cast<float*>(scales_x_ptr);
        
        // Process M dimension in tiles
        #pragma omp for schedule(dynamic, 1)
        for (size_t m0 = 0; m0 < M; m0 += M_TILE) {
            const size_t m_end = std::min(m0 + M_TILE, M);
            const size_t M_tile_actual = m_end - m0;
            
            // Quantize activation tile (multiple rows at once)
            quantize_activations_tile(
                X_fp32 + m0 * K,
                M_tile_actual,
                K,
                x_tile,
                scales_x,
                eps
            );
            
            // Process N dimension in tiles
            for (size_t n0 = 0; n0 < N; n0 += N_TILE) {
                const size_t n_end = std::min(n0 + N_TILE, N);
                const size_t N_tile_actual = n_end - n0;
                
                // Unpack weight tile - this cost is now amortized over M_tile inputs
                unpack_weight_tile(
                    packed_data,
                    w_tile,
                    n0,
                    n_end,
                    K,
                    packed_K
                );
                
                // Compute M_tile x N_tile block of outputs
                // Each unpacked weight is reused M_tile times
                #if defined(__ARM_FEATURE_DOTPROD)
                compute_tile_gemm_sdot(
                    x_tile,
                    w_tile,
                    Y_fp32,
                    m0,
                    n0,
                    M_tile_actual,
                    N_tile_actual,
                    K,
                    N,
                    scales_x,
                    scale_w
                );
                #else
                compute_tile_gemm_neon(
                    x_tile,
                    w_tile,
                    Y_fp32,
                    m0,
                    n0,
                    M_tile_actual,
                    N_tile_actual,
                    K,
                    N,
                    scales_x,
                    scale_w
                );
                #endif
            }
        }
        
        // Clean up thread-local buffers
        free(x_tile);
        free(w_tile);
        free(scales_x);
    }
}

// ============================================================================
// Cleanup function
// ============================================================================

void cleanup_weights(PackedTernaryWeights& packed_weights) {
    if (packed_weights.data) {
        free(packed_weights.data);
        packed_weights.data = nullptr;
    }
    if (packed_weights._internal_unpacked) {
        delete[] packed_weights._internal_unpacked;
        packed_weights._internal_unpacked = nullptr;
    }
}

} // namespace bitlinear