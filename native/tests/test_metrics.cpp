#include "metrics.h"

#include <cassert>
#include <cmath>
#include <vector>

using trix_native::route_churn_rate;
using trix_native::usage_metrics;

static void test_churn() {
  std::vector<int> a = {0, 1, 2, 3, 4};
  std::vector<int> b = {0, 1, 2, 3, 4};
  assert(route_churn_rate(a, b) == 0.0);

  std::vector<int> c = {4, 3, 2, 1, 0};
  double churn = route_churn_rate(a, c);
  assert(std::abs(churn - 0.8) < 1e-12);
}

static void test_usage_entropy_bounds() {
  // All to one tile -> entropy 0
  {
    std::vector<int> r(100, 0);
    auto m = usage_metrics(r, 4);
    assert(m.total == 100);
    assert(std::abs(m.entropy_nats - 0.0) < 1e-12);
    assert(m.max_tile == 0);
    assert(m.max_count == 100);
  }

  // Uniform -> entropy ln(tiles)
  {
    std::vector<int> r;
    for (int i = 0; i < 400; i++) r.push_back(i % 4);
    auto m = usage_metrics(r, 4);
    assert(m.total == 400);
    double expected = std::log(4.0);
    assert(std::abs(m.entropy_nats - expected) < 1e-9);
    assert(m.gini < 1e-12);
  }
}

int main() {
  test_churn();
  test_usage_entropy_bounds();
  return 0;
}
