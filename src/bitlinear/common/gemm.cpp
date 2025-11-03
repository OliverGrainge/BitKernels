#include "bitkernels/bitlinear.h"
#include "bitkernels/types.h"
#include "../../common/utils.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <array>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace bitkernels {
namespace common {

// ============================================================================
// Configuration
// ============================================================================

// Tiling parameters - tune these based on your cache sizes
constexpr size_t M_TILE = 8;   // Process 8 input rows together
constexpr size_t N_TILE = 32;  // Process 32 output channels together
constexpr size_t K_TILE = 256; // Process K dimension in chunks (for very large K)

// Cache line size (typical for most architectures)
constexpr size_t CACHE_LINE = 64;

// ============================================================================
// Fast LUT-based unpacking
// ============================================================================

// Pre-computed lookup table for unpacking ternary weights
static auto generate_ternary_lut() {
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

static const auto TERNARY_LUT = generate_ternary_lut();

// Optimized unpacking using lookup table
static inline void unpack_weight_row_fast(
    const uint8_t* __restrict__ packed_row,
    int8_t* __restrict__ unpacked_row,
    size_t K
) {
    size_t k = 0;
    
    // Process 32 values at a time (8 packed bytes) for better cache utilization
    for (; k + 32 <= K; k += 32) {
        size_t packed_idx = k / 4;
        
        // Prefetch next cache line
        __builtin_prefetch(packed_row + packed_idx + 8, 0, 1);
        
        // Unpack 8 bytes (32 ternary values)
        for (int i = 0; i < 8; ++i) {
            uint8_t byte = packed_row[packed_idx + i];
            const auto& vals = TERNARY_LUT[byte];
            unpacked_row[k + i * 4 + 0] = vals[0];
            unpacked_row[k + i * 4 + 1] = vals[1];
            unpacked_row[k + i * 4 + 2] = vals[2];
            unpacked_row[k + i * 4 + 3] = vals[3];
        }
    }
    
    // Process 16 values at a time
    for (; k + 16 <= K; k += 16) {
        size_t packed_idx = k / 4;
        
        for (int i = 0; i < 4; ++i) {
            uint8_t byte = packed_row[packed_idx + i];
            const auto& vals = TERNARY_LUT[byte];
            unpacked_row[k + i * 4 + 0] = vals[0];
            unpacked_row[k + i * 4 + 1] = vals[1];
            unpacked_row[k + i * 4 + 2] = vals[2];
            unpacked_row[k + i * 4 + 3] = vals[3];
        }
    }
    
    // Handle remainder
    for (; k < K; k += 4) {
        uint8_t byte = packed_row[k / 4];
        const auto& vals = TERNARY_LUT[byte];
        size_t remaining = std::min(size_t(4), K - k);
        for (size_t i = 0; i < remaining; ++i) {
            unpacked_row[k + i] = vals[i];
        }
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
        
        // Find absolute maximum
        float abs_max = 0.0f;
        
        for (size_t k = 0; k < K; ++k) {
            abs_max = std::max(abs_max, std::abs(x_row[k]));
        }

        float scale = std::max(abs_max / 127.0f, eps);
        scales_out[m] = scale;
        const float inv_scale = 1.0f / scale;

        // Quantize to int8
        for (size_t k = 0; k < K; ++k) {
            float fq = std::round(x_row[k] * inv_scale);
            // Clamp to [-127, 127]
            fq = std::max(-127.0f, std::min(127.0f, fq));
            x_row_int8[k] = static_cast<int8_t>(fq);
        }
    }
}

// ============================================================================
// Optimized Tile GEMM Kernel
// ============================================================================

// Compute M_tile x N_tile outputs using portable optimizations
static inline void compute_tile_gemm_portable(
    const int8_t* __restrict__ x_tile,
    const int8_t* __restrict__ w_tile,
    float* __restrict__ Y_fp32,
    size_t m0,
    size_t n0,
    size_t M_tile_actual,
    size_t N_tile_actual,
    size_t K,
    size_t N,
    const float* scales_x,
    float scale_w
) {
    for (size_t m = 0; m < M_tile_actual; ++m) {
        const int8_t* x_row = x_tile + m * K;
        const float deq_scale = scales_x[m] * scale_w;
        float* y_row = Y_fp32 + (m0 + m) * N + n0;
        
        // Process multiple outputs at once for better parallelism
        size_t n = 0;
        
        // Process 4 outputs at a time
        for (; n + 4 <= N_tile_actual; n += 4) {
            const int8_t* w0 = w_tile + (n + 0) * K;
            const int8_t* w1 = w_tile + (n + 1) * K;
            const int8_t* w2 = w_tile + (n + 2) * K;
            const int8_t* w3 = w_tile + (n + 3) * K;
            
            int32_t sum0 = 0;
            int32_t sum1 = 0;
            int32_t sum2 = 0;
            int32_t sum3 = 0;
            
            // Main dot product loop - compiler can auto-vectorize this
            size_t k = 0;
            
            // Process in blocks of 16 for better cache utilization
            for (; k + 16 <= K; k += 16) {
                // Prefetch next iteration
                if (k + 32 <= K) {
                    __builtin_prefetch(x_row + k + 16, 0, 1);
                    __builtin_prefetch(w0 + k + 16, 0, 1);
                    __builtin_prefetch(w1 + k + 16, 0, 1);
                    __builtin_prefetch(w2 + k + 16, 0, 1);
                    __builtin_prefetch(w3 + k + 16, 0, 1);
                }
                
                for (size_t kk = 0; kk < 16; ++kk) {
                    int32_t xi = x_row[k + kk];
                    sum0 += xi * w0[k + kk];
                    sum1 += xi * w1[k + kk];
                    sum2 += xi * w2[k + kk];
                    sum3 += xi * w3[k + kk];
                }
            }
            
            // Handle remainder
            for (; k < K; ++k) {
                int32_t xi = x_row[k];
                sum0 += xi * w0[k];
                sum1 += xi * w1[k];
                sum2 += xi * w2[k];
                sum3 += xi * w3[k];
            }
            
            y_row[n + 0] = sum0 * deq_scale;
            y_row[n + 1] = sum1 * deq_scale;
            y_row[n + 2] = sum2 * deq_scale;
            y_row[n + 3] = sum3 * deq_scale;
        }
        
        // Process remaining outputs one at a time
        for (; n < N_tile_actual; ++n) {
            const int8_t* w = w_tile + n * K;
            int32_t sum = 0;
            
            size_t k = 0;
            
            // Process in blocks of 16
            for (; k + 16 <= K; k += 16) {
                for (size_t kk = 0; kk < 16; ++kk) {
                    sum += int32_t(x_row[k + kk]) * int32_t(w[k + kk]);
                }
            }
            
            // Handle remainder
            for (; k < K; ++k) {
                sum += int32_t(x_row[k]) * int32_t(w[k]);
            }
            
            y_row[n] = sum * deq_scale;
        }
    }
}

// ============================================================================
// GEMM Implementation
// ============================================================================

void bitlinear_gemm_impl(
    const float* X_fp32,
    size_t M,
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
    
    // Zero output matrix
    std::memset(Y_fp32, 0, M * N * sizeof(float));

    // Process in tiles to maximize weight reuse and cache efficiency
    #pragma omp parallel
    {
        // Thread-local buffers for tile processing
        void* x_tile_ptr = nullptr;
        void* w_tile_ptr = nullptr;
        void* scales_x_ptr = nullptr;
        
        // Allocate aligned memory for better performance
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
            
            // Quantize activation tile once and reuse across all N tiles
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
                
                // Unpack weight tile from compressed format
                unpack_weight_tile(
                    packed_data,
                    w_tile,
                    n0,
                    n_end,
                    K,
                    packed_K
                );
                
                // Compute tile using optimized kernel
                compute_tile_gemm_portable(
                    x_tile, w_tile, Y_fp32,
                    m0, n0, M_tile_actual, N_tile_actual,
                    K, N, scales_x, scale_w
                );
            }
        }
        
        // Free thread-local buffers
        free(x_tile);
        free(w_tile);
        free(scales_x);
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        #pragma omp parallel for
        for (size_t m = 0; m < M; ++m) {
            float* y_row = Y_fp32 + m * N;
            
            // Vectorized bias addition - compiler can auto-vectorize
            for (size_t n = 0; n < N; ++n) {
                y_row[n] += bias[n];
            }
        }
    }
}

}  // namespace common
}  // namespace bitkernels

