#ifndef BITKERNELS_TYPES_H
#define BITKERNELS_TYPES_H

#include <cstdint>
#include <cstddef>

/**
 * BitKernels: Optimized quantized neural network kernels
 * 
 * Common types and data structures used across all kernels
 */

namespace bitkernels {

/**
 * Packed ternary weight structure for BitLinear operations
 * 
 * Stores N×K weight matrix in packed format:
 * - Each weight is 2 bits: 00={-1}, 01={0}, 10={1}, 11=unused
 * - 4 weights packed per byte
 * - Layout: row-major for W[N×K]
 */
struct PackedWeights {
    uint8_t* data;           // Packed weight data: size = N * K / 4 bytes
    float scale;             // Per-tensor scale factor (from abs-mean quantization)
    size_t N;                // Number of output features
    size_t K;                // Number of input features
    size_t packed_K;         // K / 4 (number of bytes per row)
    
    // Internal optimization data (architecture-specific, may be nullptr)
    void* _internal_data;
    
    PackedWeights();
    ~PackedWeights();
    
    // Disable copy to prevent double-free
    PackedWeights(const PackedWeights&) = delete;
    PackedWeights& operator=(const PackedWeights&) = delete;
    
    // Allow move
    PackedWeights(PackedWeights&& other) noexcept;
    PackedWeights& operator=(PackedWeights&& other) noexcept;
};

}  // namespace bitkernels

#endif  // BITKERNELS_TYPES_H

