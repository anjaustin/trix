#include "ternary_pack.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

static std::vector<int8_t> make_random_pm1(int n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> dist(0, 1);
  std::vector<int8_t> out;
  out.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; i++) {
    out[static_cast<size_t>(i)] = dist(rng) ? static_cast<int8_t>(+1) : static_cast<int8_t>(-1);
  }
  return out;
}

static std::vector<int8_t> make_random_ternary(int n, uint64_t seed) {
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

static int32_t dot_i8(const int8_t* a, const int8_t* b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++) {
    acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return acc;
}

static std::vector<uint8_t> pack_ternary_2b(const int8_t* v, int n) {
  const int packed = (n + 3) / 4;
  std::vector<uint8_t> out;
  out.resize(static_cast<size_t>(packed), 0);
  for (int p = 0; p < packed; p++) {
    const int base = p * 4;
    const int8_t a0 = (base + 0 < n) ? v[base + 0] : 0;
    const int8_t a1 = (base + 1 < n) ? v[base + 1] : 0;
    const int8_t a2 = (base + 2 < n) ? v[base + 2] : 0;
    const int8_t a3 = (base + 3 < n) ? v[base + 3] : 0;
    out[static_cast<size_t>(p)] = trix_native::pack4(a0, a1, a2, a3);
  }
  return out;
}

static void test_equivalence_binary_inputs() {
  // Under x in {+1,-1}^d, argmax dot equals argmin popcount distance.
  const int tiles = 32;
  const int dim = 513;  // not divisible by 4
  const int inputs_n = 256;
  const int packed = (dim + 3) / 4;

  std::vector<int8_t> sig = make_random_ternary(tiles * dim, 1);

  for (int i = 0; i < inputs_n; i++) {
    std::vector<int8_t> x = make_random_pm1(dim, static_cast<uint64_t>(i + 10));
    std::vector<uint8_t> px = pack_ternary_2b(x.data(), dim);

    int best_dot_t = 0;
    int32_t best_dot = std::numeric_limits<int32_t>::min();

    int best_dist_t = 0;
    int best_dist = std::numeric_limits<int>::max();

    for (int t = 0; t < tiles; t++) {
      const int8_t* st = &sig[static_cast<size_t>(t) * static_cast<size_t>(dim)];

      const int32_t d = dot_i8(x.data(), st, dim);
      if (d > best_dot) {
        best_dot = d;
        best_dot_t = t;
      }

      std::vector<uint8_t> ps = pack_ternary_2b(st, dim);
      const int dist = trix_native::popcount_distance_packed(px.data(), ps.data(), packed, false);
      if (dist < best_dist) {
        best_dist = dist;
        best_dist_t = t;
      }
    }

    assert(best_dot_t == best_dist_t);
  }
}

static void test_equivalence_with_masked_zeros() {
  // When x has zeros, equivalence holds if we ignore x==0 coordinates in popcount distance.
  const int tiles = 16;
  const int dim = 257;
  const int inputs_n = 128;
  const int packed = (dim + 3) / 4;

  std::vector<int8_t> sig = make_random_ternary(tiles * dim, 777);

  for (int i = 0; i < inputs_n; i++) {
    std::vector<int8_t> x = make_random_ternary(dim, static_cast<uint64_t>(900 + i));
    std::vector<uint8_t> px = pack_ternary_2b(x.data(), dim);

    int best_dot_t = 0;
    int32_t best_dot = std::numeric_limits<int32_t>::min();

    int best_dist_t = 0;
    int best_dist = std::numeric_limits<int>::max();

    for (int t = 0; t < tiles; t++) {
      const int8_t* st = &sig[static_cast<size_t>(t) * static_cast<size_t>(dim)];

      const int32_t d = dot_i8(x.data(), st, dim);
      if (d > best_dot) {
        best_dot = d;
        best_dot_t = t;
      }

      std::vector<uint8_t> ps = pack_ternary_2b(st, dim);
      const int dist = trix_native::popcount_distance_packed(px.data(), ps.data(), packed, true);
      if (dist < best_dist) {
        best_dist = dist;
        best_dist_t = t;
      }
    }

    assert(best_dot_t == best_dist_t);
  }
}

int main() {
  test_equivalence_binary_inputs();
  test_equivalence_with_masked_zeros();
  return 0;
}
