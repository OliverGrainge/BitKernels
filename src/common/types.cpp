#include "bitkernels/types.h"
#include <cstdlib>

namespace bitkernels {

PackedWeights::PackedWeights() 
    : data(nullptr), scale(1.0f), N(0), K(0), packed_K(0), _internal_data(nullptr) {}

PackedWeights::~PackedWeights() {
    if (data != nullptr) {
        free(data);  // Use free since we use posix_memalign
        data = nullptr;
    }
    if (_internal_data != nullptr) {
        delete[] static_cast<int8_t*>(_internal_data);
        _internal_data = nullptr;
    }
}

PackedWeights::PackedWeights(PackedWeights&& other) noexcept
    : data(other.data), scale(other.scale), N(other.N), K(other.K), 
      packed_K(other.packed_K), _internal_data(other._internal_data) {
    other.data = nullptr;
    other._internal_data = nullptr;
}

PackedWeights& PackedWeights::operator=(PackedWeights&& other) noexcept {
    if (this != &other) {
        // Clean up existing resources
        if (data != nullptr) {
            free(data);
        }
        if (_internal_data != nullptr) {
            delete[] static_cast<int8_t*>(_internal_data);
        }
        
        // Move from other
        data = other.data;
        scale = other.scale;
        N = other.N;
        K = other.K;
        packed_K = other.packed_K;
        _internal_data = other._internal_data;
        
        // Nullify other
        other.data = nullptr;
        other._internal_data = nullptr;
    }
    return *this;
}

}  // namespace bitkernels

