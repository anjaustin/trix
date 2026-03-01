#include "routing.h"

#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

// Minimal property test: increasing p_zero should increase tie_rate for the synthetic near-tie setup.

static std::vector<int8_t> make_base_pm1(int dim, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> dist(0, 1);
  std::vector<int8_t> base;
  base.resize(static_cast<size_t>(dim));
  for (int i = 0; i < dim; i++) base[static_cast<size_t>(i)] = dist(rng) ? int8_t(+1) : int8_t(-1);
  return base;
}

static void run_once(double p_zero, double* out_tie_rate) {
  trix_native::RoutingConfig cfg;
  cfg.tiles = 2;
  cfg.dim = 256;
  const int inputs_n = 2048;
  const int diff_dims = 16;
  const double bias = 0.55;

  std::vector<int8_t> base = make_base_pm1(cfg.dim, 1);

  // Flip first diff_dims indices.
  std::vector<int> flip_idx;
  for (int j = 0; j < diff_dims; j++) flip_idx.push_back(j);

  std::vector<int8_t> sig;
  sig.resize(static_cast<size_t>(cfg.tiles) * static_cast<size_t>(cfg.dim));
  for (int i = 0; i < cfg.dim; i++) sig[static_cast<size_t>(i)] = base[static_cast<size_t>(i)];
  for (int i = 0; i < cfg.dim; i++) sig[static_cast<size_t>(cfg.dim + i)] = base[static_cast<size_t>(i)];
  for (int j : flip_idx) sig[static_cast<size_t>(cfg.dim + j)] = static_cast<int8_t>(-sig[static_cast<size_t>(cfg.dim + j)]);

  std::vector<int8_t> x;
  x.resize(static_cast<size_t>(inputs_n) * static_cast<size_t>(cfg.dim));
  std::mt19937_64 rng(2);
  std::uniform_real_distribution<double> u01(0.0, 1.0);

  for (int n = 0; n < inputs_n; n++) {
    int8_t* row = x.data() + static_cast<size_t>(n) * static_cast<size_t>(cfg.dim);
    for (int i = 0; i < cfg.dim; i++) row[i] = base[static_cast<size_t>(i)];
    for (int j : flip_idx) {
      if (u01(rng) < p_zero) {
        row[j] = 0;
      } else {
        const bool pick0 = (u01(rng) < bias);
        const int8_t s0 = sig[static_cast<size_t>(j)];
        const int8_t s1 = sig[static_cast<size_t>(cfg.dim + j)];
        row[j] = pick0 ? s0 : s1;
      }
    }
  }

  trix_native::RoutingStats stats;
  (void)trix_native::route_argmax_with_stats(
      cfg, sig.data(), x.data(), inputs_n, trix_native::TieBreak::First, 0, &stats);
  *out_tie_rate = stats.tie_rate;
}

int main() {
  double tie_lo = 0.0;
  double tie_hi = 0.0;
  run_once(0.0, &tie_lo);
  run_once(0.99, &tie_hi);
  assert(tie_hi > tie_lo);
  return 0;
}
