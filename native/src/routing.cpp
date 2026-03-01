#include "routing.h"

#include <algorithm>
#include <cstdint>
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

std::vector<int8_t> make_zeros(int n) {
  std::vector<int8_t> out;
  out.assign(static_cast<size_t>(n), static_cast<int8_t>(0));
  return out;
}

std::vector<int8_t> make_identical_signatures(const RoutingConfig& cfg, uint64_t seed) {
  const int dim = cfg.dim;
  const int tiles = cfg.tiles;

  std::vector<int8_t> one = make_random_ternary(dim, seed);
  std::vector<int8_t> out;
  out.resize(static_cast<size_t>(tiles) * static_cast<size_t>(dim));

  for (int t = 0; t < tiles; t++) {
    std::copy(one.begin(), one.end(), out.begin() + static_cast<size_t>(t) * static_cast<size_t>(dim));
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
  RoutingStats ignored;
  return route_argmax_with_stats(cfg, signatures, inputs, inputs_n, TieBreak::First, 0, &ignored);
}

static inline uint64_t mix64(uint64_t x) {
  // SplitMix64
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

static inline uint64_t hash_input_row(const int8_t* x, int n, uint64_t seed) {
  uint64_t h = mix64(seed ^ static_cast<uint64_t>(n));
  const int step = (n >= 64) ? (n / 64) : 1;
  for (int i = 0; i < n; i += step) {
    h ^= static_cast<uint64_t>(static_cast<uint8_t>(x[i] + 2));
    h = mix64(h);
  }
  return h;
}

std::vector<int> route_argmax_with_stats(
    const RoutingConfig& cfg,
    const int8_t* signatures,
    const int8_t* inputs,
    int inputs_n,
    TieBreak tie_break,
    uint64_t seed,
    RoutingStats* out_stats) {
  const int tiles = cfg.tiles;
  const int dim = cfg.dim;

  std::vector<int> routes;
  routes.resize(static_cast<size_t>(inputs_n));

  int64_t ties = 0;
  int64_t near_ties = 0;
  double margin_sum = 0.0;

  for (int i = 0; i < inputs_n; i++) {
    const int8_t* x = inputs + static_cast<size_t>(i) * static_cast<size_t>(dim);

    int32_t best = std::numeric_limits<int32_t>::min();
    int32_t second = std::numeric_limits<int32_t>::min();
    int best_t = 0;
    int best_count = 0;

    for (int t = 0; t < tiles; t++) {
      const int8_t* sig = signatures + static_cast<size_t>(t) * static_cast<size_t>(dim);
      int32_t s = dot_ternary(x, sig, dim);
      if (s > best) {
        second = best;
        best = s;
        best_t = t;
        best_count = 1;
      } else if (s == best) {
        best_count++;
      } else if (s > second) {
        second = s;
      }
    }

    if (second == std::numeric_limits<int32_t>::min()) second = best;
    const int32_t margin = best - second;
    margin_sum += static_cast<double>(margin);
    if (margin == 0) ties++;
    if (margin <= 2) near_ties++;

    if (best_count > 1 && tie_break == TieBreak::Hash) {
      const uint64_t h = hash_input_row(x, dim, seed);
      best_t = static_cast<int>(h % static_cast<uint64_t>(tiles));
    }

    routes[static_cast<size_t>(i)] = best_t;
  }

  if (out_stats) {
    RoutingStats s;
    s.inputs = inputs_n;
    s.ties = ties;
    s.near_ties = near_ties;
    s.tie_rate = (inputs_n > 0) ? (static_cast<double>(ties) / static_cast<double>(inputs_n)) : 0.0;
    s.near_tie_rate = (inputs_n > 0) ? (static_cast<double>(near_ties) / static_cast<double>(inputs_n)) : 0.0;
    s.margin_mean = (inputs_n > 0) ? (margin_sum / static_cast<double>(inputs_n)) : 0.0;
    *out_stats = s;
  }

  return routes;
}

}  // namespace trix_native
