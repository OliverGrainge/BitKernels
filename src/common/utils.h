#ifndef BITKERNELS_COMMON_UTILS_H
#define BITKERNELS_COMMON_UTILS_H

#include <cstdint>

namespace bitkernels {
namespace common {

/**
 * Pack 4 ternary values {-1, 0, 1} into a single byte using 2-bit encoding:
 * - -1 → 00 (binary)
 * -  0 → 01 (binary)
 * - +1 → 10 (binary)
 */
inline uint8_t pack_ternary(int8_t t0, int8_t t1, int8_t t2, int8_t t3) {
    auto enc = [](int8_t t) -> uint8_t {
        return t < 0 ? 0u : (t == 0 ? 1u : 2u);
    };
    return enc(t0) | (enc(t1) << 2) | (enc(t2) << 4) | (enc(t3) << 6);
}

/**
 * Decode a single 2-bit ternary value
 */
inline int8_t decode_ternary_2bit(uint8_t encoded) {
    // 00 → -1, 01 → 0, 10 → 1, 11 → undefined (treat as 1)
    return (encoded == 0) ? -1 : ((encoded == 1) ? 0 : 1);
}

}  // namespace common
}  // namespace bitkernels

#endif  // BITKERNELS_COMMON_UTILS_H

