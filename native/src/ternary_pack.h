#pragma once

#include <cstdint>

namespace trix_native {

// Encode ternary {-1,0,+1} into 2-bit codes:
//   00 -> 0
//   01 -> +1
//   10 -> -1
// This matches the encoding used elsewhere in the repo.
static inline uint8_t ternary_code(int8_t v) {
  if (v > 0) return 0x01;
  if (v < 0) return 0x02;
  return 0x00;
}

// Pack 4 ternary codes into one byte (2 bits each).
static inline uint8_t pack4(int8_t a0, int8_t a1, int8_t a2, int8_t a3) {
  return static_cast<uint8_t>(
      (ternary_code(a0) << 0) | (ternary_code(a1) << 2) | (ternary_code(a2) << 4) |
      (ternary_code(a3) << 6));
}

static inline int popcount8(uint8_t x) {
  return __builtin_popcount(static_cast<unsigned int>(x));
}

}  // namespace trix_native
