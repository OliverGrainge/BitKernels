#ifndef BITLINEAR_H
#define BITLINEAR_H

#include <cstdint>
#include <cstddef>

/**
 * BitLinear: Ternary Weight {-1, 0, 1} × INT8 Activation Linear Layer
 * 
 * Weight format: Y[M×N] = X[M×K] @ W[N×K]^T
 * - W is stored transposed for cache-friendly access during matmul
 * - Ternary weights {-1, 0, 1} packed as 2 bits per weight (4 weights per byte)
 * - Per-tensor abs-mean quantization for weights
 * - Dynamic per-channel (per-M) quantization for activations
 */

namespace bitlinear {

/**
 * Packed ternary weight structure
 * 
 * Stores N×K weight matrix in packed format:
 * - Each weight is 2 bits: 00={-1}, 01={0}, 10={1}, 11=unused
 * - 4 weights packed per byte
 * - Layout: row-major for W[N×K]
 */
struct PackedTernaryWeights {
    uint8_t* data;           // Packed weight data: size = N * K / 4 bytes
    float scale;             // Per-tensor scale factor (from abs-mean quantization)
    size_t N;                // Number of output features
    size_t K;                // Number of input features
    size_t packed_K;         // K / 4 (number of bytes per row)
    int8_t* _internal_unpacked;  // Internal: unpacked weights for optimization (size = N * K)
    
    PackedTernaryWeights() 
        : data(nullptr), scale(1.0f), N(0), K(0), packed_K(0), _internal_unpacked(nullptr) {}
    
    ~PackedTernaryWeights() {
        if (data != nullptr) {
            delete[] data;
            data = nullptr;
        }
        if (_internal_unpacked != nullptr) {
            delete[] _internal_unpacked;
            _internal_unpacked = nullptr;
        }
    }
    
    // Disable copy to prevent double-free
    PackedTernaryWeights(const PackedTernaryWeights&) = delete;
    PackedTernaryWeights& operator=(const PackedTernaryWeights&) = delete;
    
    // Allow move
    PackedTernaryWeights(PackedTernaryWeights&& other) noexcept
        : data(other.data), scale(other.scale), N(other.N), K(other.K), packed_K(other.packed_K),
          _internal_unpacked(other._internal_unpacked) {
        other.data = nullptr;
        other._internal_unpacked = nullptr;
    }
};

/**
 * Prepare and quantize weights to ternary format
 * 
 * Quantization method: Abs-mean with per-tensor scale
 * 1. Compute scale = mean(abs(W))
 * 2. Quantize: W_ternary = sign(round(W / scale))
 *    Maps to: {-1, 0, 1}
 * 3. Pack 4 ternary values per byte using 2-bit encoding:
 *    - -1 → 00 (binary)
 *    -  0 → 01 (binary)
 *    - +1 → 10 (binary)
 * 
 * @param weights_fp32  Input weight matrix W[N×K] in row-major format (FP32)
 * @param N             Number of output features (rows of W)
 * @param K             Number of input features (columns of W)
 * @param packed_out    Output packed ternary weight structure (allocated internally)
 * @param eps           Epsilon for numerical stability
 * 
 * @note K must be divisible by 4 for packing
 * @note Caller is responsible for managing the lifecycle of packed_out
 */
void prepare_weights(
    const float* weights_fp32,
    size_t N,
    size_t K,
    PackedTernaryWeights& packed_out,
    float eps
);

/**
 * Forward pass: Y = X @ W^T where W is ternary and X is dynamically quantized to INT8
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
 * @param eps           Epsilon for numerical stability
 * 
 * @note K must match packed_weights.K
 * @note Y_fp32 must be pre-allocated with size M × N
 */
void linear(
    const float* X_fp32,
    size_t M,
    size_t K,
    const PackedTernaryWeights& packed_weights,
    float* Y_fp32,
    float eps
);

}  // namespace bitlinear

#endif  // BITLINEAR_H
