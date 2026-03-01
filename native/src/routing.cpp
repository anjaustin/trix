#include "routing.h"

#include <algorithm>
#include <limits>
#include <random>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace trix_native {

static inline int32_t dot_ternary_scalar(const int8_t* a, const int8_t* b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++) {
    acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return acc;
}

#if defined(__ARM_NEON) || defined(__aarch64__)
static inline int32_t dot_ternary_neon(const int8_t* a, const int8_t* b, int n) {
  int i = 0;
  int32x4_t acc = vdupq_n_s32(0);

#if defined(__ARM_FEATURE_DOTPROD)
  // Dot-product extension: accumulate 4-way dot products.
  for (; i + 16 <= n; i += 16) {
    int8x16_t va = vld1q_s8(a + i);
    int8x16_t vb = vld1q_s8(b + i);
    acc = vdotq_s32(acc, va, vb);
  }
#else
  for (; i + 16 <= n; i += 16) {
    int8x16_t va = vld1q_s8(a + i);
    int8x16_t vb = vld1q_s8(b + i);

    int16x8_t prod0 = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
    int16x8_t prod1 = vmull_s8(vget_high_s8(va), vget_high_s8(vb));

    int32x4_t sum0 = vpaddlq_s16(prod0);
    int32x4_t sum1 = vpaddlq_s16(prod1);
    acc = vaddq_s32(acc, sum0);
    acc = vaddq_s32(acc, sum1);
  }
#endif

  // Horizontal sum
  int32_t out = vaddvq_s32(acc);
  for (; i < n; i++) {
    out += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return out;
}
#endif

static inline int32_t dot_ternary(const int8_t* a, const int8_t* b, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
  return dot_ternary_neon(a, b, n);
#else
  return dot_ternary_scalar(a, b, n);
#endif
}

std::vector<int8_t> make_random_ternary(int n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> dist(0, 2);
  std::vector<int8_t> out;
  out.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; i++) {
    int r = dist(rng);
    out[static_cast<size_t>(i)] = (r == 0) ? static_cast<int8_t>(-1) : (r == 1) ? static_cast<int8_t>(0)
                                                                                 : static_cast<int8_t>(+1);
  }
  return out;
}

void apply_ternary_resample_noise(int8_t* data, int n, double flip_prob, uint64_t seed) {
  if (flip_prob <= 0.0) return;
  if (flip_prob >= 1.0) flip_prob = 1.0;

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::uniform_int_distribution<int> dist(0, 2);

  for (int i = 0; i < n; i++) {
    if (u01(rng) < flip_prob) {
      int r = dist(rng);
      data[i] = (r == 0) ? static_cast<int8_t>(-1) : (r == 1) ? static_cast<int8_t>(0)
                                                              : static_cast<int8_t>(+1);
    }
  }
}

std::vector<int> route_argmax(
    const RoutingConfig& cfg,
    const int8_t* signatures,
    const int8_t* inputs,
    int inputs_n) {
  const int tiles = cfg.tiles;
  const int dim = cfg.dim;

  std::vector<int> routes;
  routes.resize(static_cast<size_t>(inputs_n));

  for (int i = 0; i < inputs_n; i++) {
    const int8_t* x = inputs + static_cast<size_t>(i) * static_cast<size_t>(dim);
    int best_t = 0;
    int32_t best = std::numeric_limits<int32_t>::min();

    for (int t = 0; t < tiles; t++) {
      const int8_t* sig = signatures + static_cast<size_t>(t) * static_cast<size_t>(dim);
      int32_t s = dot_ternary(x, sig, dim);
      if (s > best) {
        best = s;
        best_t = t;
      }
    }
    routes[static_cast<size_t>(i)] = best_t;
  }

  return routes;
}

}  // namespace trix_native
