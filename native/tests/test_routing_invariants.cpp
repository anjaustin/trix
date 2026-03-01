#include "metrics.h"
#include "routing.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

using trix_native::RoutingConfig;
using trix_native::apply_ternary_resample_noise;
using trix_native::make_random_ternary;
using trix_native::route_argmax;
using trix_native::route_churn_rate;
using trix_native::usage_metrics;

static void test_valid_tile_ids() {
  RoutingConfig cfg;
  cfg.tiles = 32;
  cfg.dim = 256;

  const int inputs_n = 2048;
  std::vector<int8_t> sig = make_random_ternary(cfg.tiles * cfg.dim, 123);
  std::vector<int8_t> in = make_random_ternary(inputs_n * cfg.dim, 456);

  std::vector<int> routes = route_argmax(cfg, sig.data(), in.data(), inputs_n);
  assert(static_cast<int>(routes.size()) == inputs_n);
  for (int r : routes) {
    assert(r >= 0);
    assert(r < cfg.tiles);
  }
}

static void test_distribution_health_noncollapse() {
  // Deterministic synthetic setup; this is not a claim about all routing,
  // just a sanity check that argmax(dot) on random ternary signatures does not
  // trivially collapse.
  RoutingConfig cfg;
  cfg.tiles = 64;
  cfg.dim = 1024;

  const int inputs_n = 8192;
  std::vector<int8_t> sig = make_random_ternary(cfg.tiles * cfg.dim, 1);
  std::vector<int8_t> in = make_random_ternary(inputs_n * cfg.dim, 2);

  std::vector<int> routes = route_argmax(cfg, sig.data(), in.data(), inputs_n);
  auto m = usage_metrics(routes, cfg.tiles);

  // Entropy lower bound (nats). Uniform entropy is ln(tiles) ~ 4.159.
  // We just want "not collapsed".
  assert(m.entropy_nats > 2.5);

  // Max tile shouldn't own an absurd fraction.
  const double frac = static_cast<double>(m.max_count) / static_cast<double>(m.total);
  assert(frac < 0.10);

  // Skew shouldn't be extreme.
  assert(m.gini < 0.35);
}

static void test_stability_under_small_perturbation() {
  RoutingConfig cfg;
  cfg.tiles = 64;
  cfg.dim = 1024;

  const int inputs_n = 8192;
  std::vector<int8_t> sig0 = make_random_ternary(cfg.tiles * cfg.dim, 7);
  std::vector<int8_t> in = make_random_ternary(inputs_n * cfg.dim, 8);

  std::vector<int> r0 = route_argmax(cfg, sig0.data(), in.data(), inputs_n);

  std::vector<int8_t> sig1 = sig0;
  apply_ternary_resample_noise(sig1.data(), static_cast<int>(sig1.size()), 0.001, 9);
  std::vector<int> r1 = route_argmax(cfg, sig1.data(), in.data(), inputs_n);

  const double churn = route_churn_rate(r0, r1);

  // Controlled bound: tiny signature noise should not induce near-total rerouting.
  assert(churn < 0.40);
}

int main() {
  test_valid_tile_ids();
  test_distribution_health_noncollapse();
  test_stability_under_small_perturbation();
  return 0;
}
