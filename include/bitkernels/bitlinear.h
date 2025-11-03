#ifndef BITKERNELS_BITLINEAR_H
#define BITKERNELS_BITLINEAR_H

#include "bitkernels/types.h"

/**
 * BitLinear: Ternary Weight {-1, 0, 1} × INT8 Activation Linear Layer
 * 
 * Weight format: Y[M×N] = X[M×K] @ W[N×K]^T
 * - W is stored transposed for cache-friendly access during matmul
 * - Ternary weights {-1, 0, 1} packed as 2 bits per weight (4 weights per byte)
 * - Per-tensor abs-mean quantization for weights
 * - Dynamic per-channel (per-M) quantization for activations
 */

namespace bitkernels {

/**
 * Prepare and quantize weights to ternary format
 * 
 * Quantization method: Abs-mean with per-tensor scale
 * 1. Compute scale = mean(abs(W))
 * 2. Quantize: W_ternary = sign(round(W / scale))
 *    Maps to: {-1, 0, 1}
 * 3. Pack 4 ternary values per byte using 2-bit encoding
 * 
 * @param weights_fp32  Input weight matrix W[N×K] in row-major format (FP32)
 * @param N             Number of output features (rows of W)
 * @param K             Number of input features (columns of W)
 * @param packed_out    Output packed ternary weight structure (allocated internally)
 * @param eps           Epsilon for numerical stability (default: 1e-6)
 * 
 * @note K must be divisible by 4 for packing
 * @note Caller is responsible for managing the lifecycle of packed_out
 */
void prepare_weights(
    const float* weights_fp32,
    size_t N,
    size_t K,
    PackedWeights& packed_out,
    float eps = 1e-6f
);

/**
 * Forward pass: Y = X @ W^T where W is ternary and X is dynamically quantized to INT8
 * 
 * This function automatically dispatches to the optimal kernel based on:
 * - Hardware architecture (ARM NEON, x86 AVX, etc.)
 * - Matrix dimensions (GEMM vs GEMV)
 * 
 * Computation flow:
 * 1. For each row of X (per-channel):
 *    - Compute abs_max across K dimensions
 *    - Compute scale_x[m] = abs_max / 127.0
 *    - Quantize to INT8: X_q[m,k] = round(X[m,k] / scale_x[m])
 * 2. Perform integer matmul: Y_int32[m,n] = sum_k(X_q[m,k] * W_ternary[n,k])
 * 3. Dequantize: Y[m,n] = Y_int32[m,n] * scale_x[m] * scale_w
 * 
 * @param X_fp32        Input activations X[M×K] in row-major format (FP32)
 * @param M             Batch size / number of input rows
 * @param K             Number of input features (must match weights)
 * @param packed_weights Prepared packed ternary weights W[N×K]
 * @param Y_fp32        Output activations Y[M×N] in row-major format (FP32)
 * @param eps           Epsilon for numerical stability (default: 1e-6)
 * 
 * @note K must match packed_weights.K
 * @note Y_fp32 must be pre-allocated with size M × N
 */
void bitlinear_forward(
    const float* X_fp32,
    size_t M,
    size_t K,
    const PackedWeights& packed_weights,
    float* Y_fp32,
    float eps = 1e-6f
);

// Explicit kernel variants (optional - users can call these directly if needed)

/**
 * Matrix-matrix multiplication (optimized for M > 1)
 * Uses tiled computation with efficient weight reuse
 */
void bitlinear_gemm(
    const float* X_fp32,
    size_t M,
    size_t K,
    const PackedWeights& packed_weights,
    float* Y_fp32,
    float eps = 1e-6f
);

/**
 * Matrix-vector multiplication (optimized for M == 1)
 * Simplified computation path for single input vector
 */
void bitlinear_gemv(
    const float* X_fp32,
    size_t K,
    const PackedWeights& packed_weights,
    float* Y_fp32,
    float eps = 1e-6f
);

}  // namespace bitkernels

#endif  // BITKERNELS_BITLINEAR_H

