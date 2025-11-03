#include "bitkernels/bitlinear.h"
#include "bitkernels/types.h"
#include <iostream>

namespace bitkernels {

// Forward declarations of architecture-specific implementations
namespace arm_neon {
    void bitlinear_gemm_impl(const float* X, size_t M, size_t K, 
                            const PackedWeights& W, float* Y, 
                            const float* bias, float eps);
    void bitlinear_gemv_impl(const float* X, size_t K,
                            const PackedWeights& W, float* Y, 
                            const float* bias, float eps);
}

// Threshold for choosing between GEMV and GEMM
constexpr size_t GEMV_THRESHOLD = 4;

// ============================================================================
// Public API - Automatic dispatch
// ============================================================================

void bitlinear_forward(
    const float* X_fp32,
    size_t M,
    size_t K,
    const PackedWeights& packed_weights,
    float* Y_fp32,
    const float* bias,
    float eps
) {
    // Dispatch based on batch size
    if (M < GEMV_THRESHOLD) {
        // For small M, use GEMV for each row
        for (size_t m = 0; m < M; ++m) {
            #ifdef __ARM_NEON
                arm_neon::bitlinear_gemv_impl(
                    X_fp32 + m * K, K, packed_weights,
                    Y_fp32 + m * packed_weights.N, bias, eps
                );
            #else
                std::cerr << "ERROR: No implementation available for this architecture" << std::endl;
                return;
            #endif
        }
    } else {
        // For larger M, use tiled GEMM
        #ifdef __ARM_NEON
            arm_neon::bitlinear_gemm_impl(X_fp32, M, K, packed_weights, Y_fp32, bias, eps);
        #else
            std::cerr << "ERROR: No implementation available for this architecture" << std::endl;
            return;
        #endif
    }
}

// ============================================================================
// Public API - Explicit GEMM
// ============================================================================

void bitlinear_gemm(
    const float* X_fp32,
    size_t M,
    size_t K,
    const PackedWeights& packed_weights,
    float* Y_fp32,
    const float* bias,
    float eps
) {
    #ifdef __ARM_NEON
        arm_neon::bitlinear_gemm_impl(X_fp32, M, K, packed_weights, Y_fp32, bias, eps);
    #else
        std::cerr << "ERROR: No ARM NEON implementation available" << std::endl;
    #endif
}

// ============================================================================
// Public API - Explicit GEMV
// ============================================================================

void bitlinear_gemv(
    const float* X_fp32,
    size_t K,
    const PackedWeights& packed_weights,
    float* Y_fp32,
    const float* bias,
    float eps
) {
    #ifdef __ARM_NEON
        arm_neon::bitlinear_gemv_impl(X_fp32, K, packed_weights, Y_fp32, bias, eps);
    #else
        std::cerr << "ERROR: No ARM NEON implementation available" << std::endl;
    #endif
}

}  // namespace bitkernels

