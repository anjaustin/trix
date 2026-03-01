#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace trix_native {

static double safe_log(double x) {
  return std::log(x);
}

UsageMetrics usage_metrics(const std::vector<int>& routes, int tiles) {
  if (tiles <= 0) throw std::invalid_argument("tiles must be > 0");

  UsageMetrics m;
  m.total = static_cast<int64_t>(routes.size());
  m.counts.assign(static_cast<size_t>(tiles), 0);

  for (int r : routes) {
    if (r < 0 || r >= tiles) {
      throw std::invalid_argument("route index out of range");
    }
    m.counts[static_cast<size_t>(r)]++;
  }

  // Entropy (nats)
  if (m.total > 0) {
    double H = 0.0;
    for (int t = 0; t < tiles; t++) {
      const int64_t c = m.counts[static_cast<size_t>(t)];
      if (c == 0) continue;
      const double p = static_cast<double>(c) / static_cast<double>(m.total);
      H -= p * safe_log(p);
    }
    m.entropy_nats = H;
  }

  // Max tile
  m.max_tile = 0;
  m.max_count = m.counts[0];
  for (int t = 1; t < tiles; t++) {
    int64_t c = m.counts[static_cast<size_t>(t)];
    if (c > m.max_count) {
      m.max_count = c;
      m.max_tile = t;
    }
  }

  // Gini coefficient on counts
  // G = sum_i sum_j |xi-xj| / (2 n sum_i xi)
  // Compute via sorting for O(n log n):
  // G = (2 * sum_i i*xi) / (n*sum_x) - (n+1)/n
  // where i is 1-indexed after sorting ascending.
  std::vector<int64_t> sorted = m.counts;
  std::sort(sorted.begin(), sorted.end());
  const double n = static_cast<double>(tiles);
  const double sum_x = static_cast<double>(std::accumulate(sorted.begin(), sorted.end(), int64_t{0}));
  if (sum_x == 0.0) {
    m.gini = 0.0;
  } else {
    double num = 0.0;
    for (int i = 0; i < tiles; i++) {
      const double xi = static_cast<double>(sorted[static_cast<size_t>(i)]);
      const double ii = static_cast<double>(i + 1);
      num += ii * xi;
    }
    m.gini = (2.0 * num) / (n * sum_x) - (n + 1.0) / n;
    if (m.gini < 0.0) m.gini = 0.0;
    if (m.gini > 1.0) m.gini = 1.0;
  }

  return m;
}

double route_churn_rate(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.size() != b.size()) throw std::invalid_argument("route vectors differ in size");
  if (a.empty()) return 0.0;

  int64_t diff = 0;
  for (size_t i = 0; i < a.size(); i++) {
    diff += (a[i] == b[i]) ? 0 : 1;
  }
  return static_cast<double>(diff) / static_cast<double>(a.size());
}

double route_agreement_rate(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.size() != b.size()) throw std::invalid_argument("route vectors differ in size");
  if (a.empty()) return 1.0;

  int64_t same = 0;
  for (size_t i = 0; i < a.size(); i++) {
    same += (a[i] == b[i]) ? 1 : 0;
  }
  return static_cast<double>(same) / static_cast<double>(a.size());
}

}  // namespace trix_native
