#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trix_native {

struct UsageMetrics {
  int64_t total = 0;
  double entropy_nats = 0.0;
  double gini = 0.0;
  int max_tile = -1;
  int64_t max_count = 0;
  std::vector<int64_t> counts;
};

UsageMetrics usage_metrics(const std::vector<int>& routes, int tiles);

double route_churn_rate(const std::vector<int>& a, const std::vector<int>& b);

}  // namespace trix_native
