#pragma once

#include <cstdint>
#include <vector>

namespace trix_native {

struct RoutingConfig {
  int tiles = 64;
  int dim = 512;
};

enum class TieBreak {
  First = 0,
  Hash = 1,
};

struct RoutingStats {
  int64_t inputs = 0;
  int64_t ties = 0;      // count of inputs where best == second_best
  double tie_rate = 0.0;
  double margin_mean = 0.0;  // mean(best - second_best)
};

// Signatures and inputs are in {-1, 0, +1} stored as int8.
// Layout:
// - signatures: tiles * dim
// - inputs:     inputs_n * dim

std::vector<int8_t> make_random_ternary(int n, uint64_t seed);

// Fill vector with zeros.
std::vector<int8_t> make_zeros(int n);

// Make signatures where every tile has the same signature.
std::vector<int8_t> make_identical_signatures(const RoutingConfig& cfg, uint64_t seed);

// Route each input to argmax over tile dot(signature, input).
// Returns a vector of tile indices (size = inputs_n).
std::vector<int> route_argmax(
    const RoutingConfig& cfg,
    const int8_t* signatures,
    const int8_t* inputs,
    int inputs_n);

// Same as route_argmax, but also computes tie/margin stats.
// tie_break controls how to select a winner when multiple tiles share the max score.
std::vector<int> route_argmax_with_stats(
    const RoutingConfig& cfg,
    const int8_t* signatures,
    const int8_t* inputs,
    int inputs_n,
    TieBreak tie_break,
    uint64_t seed,
    RoutingStats* out_stats);

// Apply random independent flips to ternary vectors.
// flip_prob is probability per element of resampling it uniformly from {-1,0,+1}.
void apply_ternary_resample_noise(int8_t* data, int n, double flip_prob, uint64_t seed);

}  // namespace trix_native
