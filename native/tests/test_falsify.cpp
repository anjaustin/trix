#include "metrics.h"
#include "routing.h"
#include "ternary_pack.h"

#include <cassert>
#include <cstdint>
#include <vector>

using trix_native::RoutingConfig;
using trix_native::UsageMetrics;
using trix_native::apply_ternary_resample_noise;
using trix_native::make_identical_signatures;
using trix_native::make_random_ternary;
using trix_native::make_zeros;
using trix_native::route_argmax;
using trix_native::route_churn_rate;
using trix_native::usage_metrics;

static int bit_hamming_packed(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  assert(a.size() == b.size());
  int d = 0;
  for (size_t i = 0; i < a.size(); i++) {
    d += trix_native::popcount8(static_cast<uint8_t>(a[i] ^ b[i]));
  }
  return d;
}

static std::vector<uint8_t> pack_ternary_2b(const std::vector<int8_t>& v) {
  // 4 entries per byte
  const size_t n = v.size();
  const size_t packed = (n + 3) / 4;
  std::vector<uint8_t> out;
  out.resize(packed, 0);
  for (size_t p = 0; p < packed; p++) {
    const size_t base = p * 4;
    int8_t a0 = (base + 0 < n) ? v[base + 0] : 0;
    int8_t a1 = (base + 1 < n) ? v[base + 1] : 0;
    int8_t a2 = (base + 2 < n) ? v[base + 2] : 0;
    int8_t a3 = (base + 3 < n) ? v[base + 3] : 0;
    out[p] = trix_native::pack4(a0, a1, a2, a3);
  }
  return out;
}

// 1) Falsify "non-collapse" as a universal property.
static void falsify_noncollapse_universal() {
  RoutingConfig cfg;
  cfg.tiles = 32;
  cfg.dim = 256;

  const int inputs_n = 4096;
  std::vector<int8_t> inputs = make_random_ternary(inputs_n * cfg.dim, 100);

  // Case A: Identical signatures -> dot scores tie -> argmax tie-break collapses to first tile.
  std::vector<int8_t> sig_ident = make_identical_signatures(cfg, 200);
  std::vector<int> routes_ident = route_argmax(cfg, sig_ident.data(), inputs.data(), inputs_n);
  UsageMetrics m_ident = usage_metrics(routes_ident, cfg.tiles);
  assert(m_ident.entropy_nats == 0.0);
  assert(m_ident.max_tile == 0);
  assert(m_ident.max_count == m_ident.total);

  // Case B: Zero signatures -> all scores 0 -> also collapses.
  std::vector<int8_t> sig_zero = make_zeros(cfg.tiles * cfg.dim);
  std::vector<int> routes_zero = route_argmax(cfg, sig_zero.data(), inputs.data(), inputs_n);
  UsageMetrics m_zero = usage_metrics(routes_zero, cfg.tiles);
  assert(m_zero.entropy_nats == 0.0);
  assert(m_zero.max_tile == 0);
  assert(m_zero.max_count == m_zero.total);
}

// 2) Falsify "stability under small perturbations" as a universal property.
static void falsify_stability_universal() {
  RoutingConfig cfg;
  cfg.tiles = 64;
  cfg.dim = 1024;
  const int inputs_n = 8192;

  std::vector<int8_t> inputs = make_random_ternary(inputs_n * cfg.dim, 300);

  // Start from a degenerate tie geometry: all-zero signatures -> always routes to tile 0.
  std::vector<int8_t> sig0 = make_zeros(cfg.tiles * cfg.dim);
  std::vector<int> r0 = route_argmax(cfg, sig0.data(), inputs.data(), inputs_n);

  // Apply a *per-element* small perturbation probability.
  // Even tiny per-element perturbations can produce large churn when the baseline geometry is tie-degenerate.
  std::vector<int8_t> sig1 = sig0;
  apply_ternary_resample_noise(sig1.data(), static_cast<int>(sig1.size()), 0.001, 301);
  std::vector<int> r1 = route_argmax(cfg, sig1.data(), inputs.data(), inputs_n);

  const double churn = route_churn_rate(r0, r1);
  assert(churn > 0.50);
}

// 3) Falsify "argmax(dot) == argmin(hamming)" as a universal identity for ternary vectors.
static void falsify_dot_equals_hamming_universal() {
  // Construct a counterexample in {-1,0,+1}.
  // x has zeros; dot can ignore mismatches where x=0, while (bit) Hamming still counts them.
  std::vector<int8_t> x = {+1, +1, 0, 0};
  std::vector<int8_t> sigA = {+1, 0, 0, 0};
  std::vector<int8_t> sigB = {+1, +1, -1, -1};

  // Dot scores
  int dotA = x[0] * sigA[0] + x[1] * sigA[1] + x[2] * sigA[2] + x[3] * sigA[3];
  int dotB = x[0] * sigB[0] + x[1] * sigB[1] + x[2] * sigB[2] + x[3] * sigB[3];
  assert(dotB > dotA);

  // Bit-Hamming on 2-bit packed codes
  auto px = pack_ternary_2b(x);
  auto pA = pack_ternary_2b(sigA);
  auto pB = pack_ternary_2b(sigB);
  int hA = bit_hamming_packed(px, pA);
  int hB = bit_hamming_packed(px, pB);

  // Hamming prefers A (smaller distance) while dot prefers B.
  assert(hA < hB);
}

int main() {
  falsify_noncollapse_universal();
  falsify_stability_universal();
  falsify_dot_equals_hamming_universal();
  return 0;
}
