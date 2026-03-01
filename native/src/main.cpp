#include "jsonl.h"
#include "metrics.h"
#include "routing.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string benchmark = "routing";  // routing|stability|margin_sweep
  int tiles = 64;
  int dim = 512;
  int inputs = 4096;
  uint64_t seed = 1;
  double flip_prob = 0.01;
  std::optional<std::string> jsonl;
  std::string tie_break = "first";  // first|hash
  bool guard_ties = false;

  // margin_sweep params (requires tiles==2)
  int diff_dims = 16;
  double bias = 0.55;
};

static std::string make_run_id(const Args& args) {
  std::ostringstream oss;
  oss << "native-" << args.benchmark << "-t" << args.tiles << "-d" << args.dim << "-n" << args.inputs
      << "-s" << args.seed;
  return oss.str();
}

static void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--benchmark routing|stability|margin_sweep] [--tiles N] [--dim N] [--inputs N] [--seed N]"
         " [--flip-prob P] [--jsonl PATH] [--tie-break first|hash] [--guard-ties]"
         " [--diff-dims K] [--bias P]\n";
}

static bool parse_int(const char* s, int* out) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!end || *end != '\0') return false;
  *out = static_cast<int>(v);
  return true;
}

static bool parse_u64(const char* s, uint64_t* out) {
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (!end || *end != '\0') return false;
  *out = static_cast<uint64_t>(v);
  return true;
}

static bool parse_double(const char* s, double* out) {
  char* end = nullptr;
  double v = std::strtod(s, &end);
  if (!end || *end != '\0') return false;
  *out = v;
  return true;
}

static double clamp01(double x) {
  if (x < 0.0) return 0.0;
  if (x > 1.0) return 1.0;
  return x;
}

static std::string json_number(double x) {
  std::ostringstream oss;
  oss.precision(17);
  oss << x;
  return oss.str();
}

static std::string json_int64(int64_t x) {
  return std::to_string(x);
}

static std::string json_string(const std::string& s) {
  return std::string("\"") + trix_native::json_escape(s) + "\"";
}

}  // namespace

int main(int argc, char** argv) {
  Args args;

  for (int i = 1; i < argc; i++) {
    const char* a = argv[i];
    auto need = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        usage(argv[0]);
        std::exit(2);
      }
      return argv[++i];
    };

    if (std::strcmp(a, "--benchmark") == 0) {
      args.benchmark = need("--benchmark");
    } else if (std::strcmp(a, "--tiles") == 0) {
      if (!parse_int(need("--tiles"), &args.tiles)) return 2;
    } else if (std::strcmp(a, "--dim") == 0) {
      if (!parse_int(need("--dim"), &args.dim)) return 2;
    } else if (std::strcmp(a, "--inputs") == 0) {
      if (!parse_int(need("--inputs"), &args.inputs)) return 2;
    } else if (std::strcmp(a, "--seed") == 0) {
      if (!parse_u64(need("--seed"), &args.seed)) return 2;
    } else if (std::strcmp(a, "--flip-prob") == 0) {
      if (!parse_double(need("--flip-prob"), &args.flip_prob)) return 2;
    } else if (std::strcmp(a, "--jsonl") == 0) {
      args.jsonl = std::string(need("--jsonl"));
    } else if (std::strcmp(a, "--tie-break") == 0) {
      args.tie_break = need("--tie-break");
    } else if (std::strcmp(a, "--guard-ties") == 0) {
      args.guard_ties = true;
    } else if (std::strcmp(a, "--diff-dims") == 0) {
      if (!parse_int(need("--diff-dims"), &args.diff_dims)) return 2;
    } else if (std::strcmp(a, "--bias") == 0) {
      if (!parse_double(need("--bias"), &args.bias)) return 2;
    } else if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  if (args.tiles <= 0 || args.dim <= 0 || args.inputs <= 0) {
    std::cerr << "tiles/dim/inputs must be > 0\n";
    return 2;
  }

  trix_native::RoutingConfig cfg;
  cfg.tiles = args.tiles;
  cfg.dim = args.dim;

  trix_native::TieBreak tie_break = trix_native::TieBreak::First;
  if (args.tie_break == "hash") {
    tie_break = trix_native::TieBreak::Hash;
  } else if (args.tie_break == "first") {
    tie_break = trix_native::TieBreak::First;
  } else {
    std::cerr << "Unknown --tie-break: " << args.tie_break << "\n";
    return 2;
  }

  const int sig_n = cfg.tiles * cfg.dim;
  const int in_n = args.inputs * cfg.dim;

  std::vector<int8_t> signatures = trix_native::make_random_ternary(sig_n, args.seed ^ 0xA5A5A5A5ULL);
  std::vector<int8_t> inputs = trix_native::make_random_ternary(in_n, args.seed ^ 0x5A5A5A5AULL);

  std::optional<trix_native::JsonlWriter> writer;
  if (args.jsonl.has_value()) {
    writer.emplace(*args.jsonl);
  }

  const std::string run_id = make_run_id(args);

  trix_native::RoutingStats stats0;

  auto t0 = std::chrono::steady_clock::now();
  std::vector<int> routes0 = trix_native::route_argmax_with_stats(
      cfg, signatures.data(), inputs.data(), args.inputs, tie_break, args.seed, &stats0);
  auto t1 = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double routes_per_s = (static_cast<double>(args.inputs) / ms) * 1000.0;

  trix_native::UsageMetrics um = trix_native::usage_metrics(routes0, cfg.tiles);

  bool fallback_applied = false;
  std::string fallback_reason;
  trix_native::TieBreak effective_tie_break = tie_break;
  if (args.guard_ties && stats0.tie_rate > 0.0 && tie_break == trix_native::TieBreak::First) {
    // Guard against tie-degenerate collapse: switch to hash tie-break.
    effective_tie_break = trix_native::TieBreak::Hash;
    fallback_applied = true;
    fallback_reason = "tie_guard";

    auto tg0 = std::chrono::steady_clock::now();
    routes0 = trix_native::route_argmax_with_stats(
        cfg, signatures.data(), inputs.data(), args.inputs, effective_tie_break, args.seed, &stats0);
    auto tg1 = std::chrono::steady_clock::now();
    ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
    routes_per_s = (static_cast<double>(args.inputs) / ms) * 1000.0;
    um = trix_native::usage_metrics(routes0, cfg.tiles);
  }

  if (args.benchmark == "routing") {
    std::cout << "routing_ms=" << ms << " routes_per_s=" << routes_per_s << " entropy_nats=" << um.entropy_nats
              << " gini=" << um.gini << " max_tile=" << um.max_tile << " max_count=" << um.max_count << "\n";

    if (writer.has_value()) {
      const char* tb = (effective_tie_break == trix_native::TieBreak::Hash) ? "hash" : "first";
      writer->write_kv({
          {"schema_version", "1"},
          {"event", json_string("routing")},
          {"run_id", json_string(run_id)},
          {"tiles", std::to_string(cfg.tiles)},
          {"dim", std::to_string(cfg.dim)},
          {"inputs", std::to_string(args.inputs)},
          {"seed", std::to_string(args.seed)},
          {"address_type", json_string("tile_id")},
          {"tie_rate", json_number(stats0.tie_rate)},
          {"near_tie_rate", json_number(stats0.near_tie_rate)},
          {"margin_mean", json_number(stats0.margin_mean)},
          {"tie_break", json_string(tb)},
          {"fallback_applied", fallback_applied ? "true" : "false"},
          {"fallback_reason", fallback_applied ? json_string(fallback_reason) : "null"},
          {"routing_ms", json_number(ms)},
          {"routes_per_s", json_number(routes_per_s)},
          {"entropy_nats", json_number(um.entropy_nats)},
          {"gini", json_number(um.gini)},
          {"max_tile", std::to_string(um.max_tile)},
          {"max_count", json_int64(um.max_count)},
      });
    }
    return 0;
  }

  if (args.benchmark == "stability") {
    std::vector<int8_t> signatures2 = signatures;
    trix_native::apply_ternary_resample_noise(signatures2.data(), sig_n, args.flip_prob, args.seed ^ 0xDEADBEEFULL);

    trix_native::RoutingStats stats1;
    std::vector<int> routes1 = trix_native::route_argmax_with_stats(
        cfg, signatures2.data(), inputs.data(), args.inputs, tie_break, args.seed, &stats1);
    const double churn = trix_native::route_churn_rate(routes0, routes1);

    std::cout << "routing_ms=" << ms << " routes_per_s=" << routes_per_s << " churn=" << churn
              << " flip_prob=" << args.flip_prob << "\n";

    if (writer.has_value()) {
      const char* tb = (tie_break == trix_native::TieBreak::Hash) ? "hash" : "first";
      writer->write_kv({
          {"schema_version", "1"},
          {"event", json_string("stability")},
          {"run_id", json_string(run_id)},
          {"tiles", std::to_string(cfg.tiles)},
          {"dim", std::to_string(cfg.dim)},
          {"inputs", std::to_string(args.inputs)},
          {"seed", std::to_string(args.seed)},
          {"address_type", json_string("tile_id")},
          {"tie_rate", json_number(stats0.tie_rate)},
          {"near_tie_rate", json_number(stats0.near_tie_rate)},
          {"margin_mean", json_number(stats0.margin_mean)},
          {"tie_break", json_string(tb)},
          {"flip_prob", json_number(args.flip_prob)},
          {"routing_ms", json_number(ms)},
          {"routes_per_s", json_number(routes_per_s)},
          {"churn", json_number(churn)},
      });
    }

    return 0;
  }

  if (args.benchmark == "margin_sweep") {
    if (cfg.tiles != 2) {
      std::cerr << "margin_sweep requires --tiles 2\n";
      return 2;
    }
    if (args.diff_dims <= 0 || args.diff_dims > cfg.dim) {
      std::cerr << "--diff-dims must be in [1, dim]\n";
      return 2;
    }
    args.bias = clamp01(args.bias);
    args.flip_prob = clamp01(args.flip_prob);

    // Construct signatures with controlled near-tie geometry.
    // tile0 = base; tile1 = base with `diff_dims` sign flips.
    std::vector<int8_t> base = trix_native::make_random_ternary(cfg.dim, args.seed ^ 0x11111111ULL);
    for (int i = 0; i < cfg.dim; i++) {
      if (base[static_cast<size_t>(i)] == 0) base[static_cast<size_t>(i)] = 1;
    }

    std::vector<int> flip_idx;
    flip_idx.reserve(static_cast<size_t>(args.diff_dims));
    {
      std::mt19937_64 rng(args.seed ^ 0x22222222ULL);
      std::uniform_int_distribution<int> dist(0, cfg.dim - 1);
      std::vector<uint8_t> used(static_cast<size_t>(cfg.dim), 0);
      while (static_cast<int>(flip_idx.size()) < args.diff_dims) {
        int j = dist(rng);
        if (used[static_cast<size_t>(j)]) continue;
        used[static_cast<size_t>(j)] = 1;
        flip_idx.push_back(j);
      }
    }

    std::vector<int8_t> sig;
    sig.resize(static_cast<size_t>(cfg.tiles) * static_cast<size_t>(cfg.dim));
    for (int i = 0; i < cfg.dim; i++) sig[static_cast<size_t>(i)] = base[static_cast<size_t>(i)];
    for (int i = 0; i < cfg.dim; i++) sig[static_cast<size_t>(cfg.dim + i)] = base[static_cast<size_t>(i)];
    for (int j : flip_idx) {
      sig[static_cast<size_t>(cfg.dim + j)] = static_cast<int8_t>(-sig[static_cast<size_t>(cfg.dim + j)]);
    }

    // Sweep bias (keep p_zero=0.0) to produce near-ties without requiring exact ties.
    const double bias_list[] = {0.50, 0.51, 0.52, 0.55, 0.60, 0.70, 0.80, 0.90};
    for (double bias : bias_list) {
      const double p_zero = 0.0;
      bias = clamp01(bias);

      std::vector<int8_t> x;
      x.resize(static_cast<size_t>(args.inputs) * static_cast<size_t>(cfg.dim));

      std::mt19937_64 rng(args.seed ^ 0x33333333ULL);
      std::uniform_real_distribution<double> u01(0.0, 1.0);

      for (int n = 0; n < args.inputs; n++) {
        int8_t* row = x.data() + static_cast<size_t>(n) * static_cast<size_t>(cfg.dim);
        for (int i = 0; i < cfg.dim; i++) row[i] = base[static_cast<size_t>(i)];
        for (int j : flip_idx) {
          const double r = u01(rng);
          if (r < p_zero) {
            row[j] = 0;
          } else {
            const bool pick0 = (u01(rng) < bias);
            const int8_t s0 = sig[static_cast<size_t>(j)];
            const int8_t s1 = sig[static_cast<size_t>(cfg.dim + j)];
            row[j] = pick0 ? s0 : s1;
          }
        }
      }

      trix_native::RoutingStats s0;
      auto a0 = std::chrono::steady_clock::now();
      std::vector<int> r0 = trix_native::route_argmax_with_stats(
          cfg, sig.data(), x.data(), args.inputs, tie_break, args.seed, &s0);
      auto a1 = std::chrono::steady_clock::now();
      const double routing_ms = std::chrono::duration<double, std::milli>(a1 - a0).count();
      const double rps = (static_cast<double>(args.inputs) / routing_ms) * 1000.0;

      std::vector<int8_t> sig2 = sig;
      trix_native::apply_ternary_resample_noise(
          sig2.data(), static_cast<int>(sig2.size()), args.flip_prob, args.seed ^ 0x44444444ULL);
      trix_native::RoutingStats s1;
      std::vector<int> r1 = trix_native::route_argmax_with_stats(
          cfg, sig2.data(), x.data(), args.inputs, tie_break, args.seed, &s1);
      const double churn = trix_native::route_churn_rate(r0, r1);

      std::cout << "sweep=bias bias=" << bias << " diff_dims=" << args.diff_dims << " margin_mean=" << s0.margin_mean
                << " tie_rate=" << s0.tie_rate << " near_tie_rate=" << s0.near_tie_rate << " churn=" << churn
                << "\n";

      if (writer.has_value()) {
        const char* tb = (tie_break == trix_native::TieBreak::Hash) ? "hash" : "first";
        writer->write_kv({
            {"schema_version", "1"},
            {"event", json_string("margin_sweep")},
            {"run_id", json_string(run_id)},
            {"tiles", std::to_string(cfg.tiles)},
            {"dim", std::to_string(cfg.dim)},
            {"inputs", std::to_string(args.inputs)},
            {"seed", std::to_string(args.seed)},
            {"address_type", json_string("tile_id")},
            {"tie_break", json_string(tb)},
            {"sweep", json_string("bias")},
            {"diff_dims", std::to_string(args.diff_dims)},
            {"bias", json_number(bias)},
            {"p_zero", json_number(p_zero)},
            {"flip_prob", json_number(args.flip_prob)},
            {"routing_ms", json_number(routing_ms)},
            {"routes_per_s", json_number(rps)},
            {"tie_rate", json_number(s0.tie_rate)},
            {"near_tie_rate", json_number(s0.near_tie_rate)},
            {"margin_mean", json_number(s0.margin_mean)},
            {"churn", json_number(churn)},
        });
      }
    }

    // Sweep p_zero (keep bias fixed) to show tie-degenerate regimes.
    const double pzero_list[] = {0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99};
    for (double p_zero : pzero_list) {
      std::vector<int8_t> x;
      x.resize(static_cast<size_t>(args.inputs) * static_cast<size_t>(cfg.dim));

      std::mt19937_64 rng(args.seed ^ 0x33333333ULL);
      std::uniform_real_distribution<double> u01(0.0, 1.0);

      for (int n = 0; n < args.inputs; n++) {
        int8_t* row = x.data() + static_cast<size_t>(n) * static_cast<size_t>(cfg.dim);
        for (int i = 0; i < cfg.dim; i++) row[i] = base[static_cast<size_t>(i)];
        for (int j : flip_idx) {
          const double r = u01(rng);
          if (r < p_zero) {
            row[j] = 0;
          } else {
            const bool pick0 = (u01(rng) < args.bias);
            const int8_t s0 = sig[static_cast<size_t>(j)];
            const int8_t s1 = sig[static_cast<size_t>(cfg.dim + j)];
            row[j] = pick0 ? s0 : s1;
          }
        }
      }

      trix_native::RoutingStats s0;
      auto a0 = std::chrono::steady_clock::now();
      std::vector<int> r0 = trix_native::route_argmax_with_stats(
          cfg, sig.data(), x.data(), args.inputs, tie_break, args.seed, &s0);
      auto a1 = std::chrono::steady_clock::now();
      const double routing_ms = std::chrono::duration<double, std::milli>(a1 - a0).count();
      const double rps = (static_cast<double>(args.inputs) / routing_ms) * 1000.0;

      std::vector<int8_t> sig2 = sig;
      trix_native::apply_ternary_resample_noise(
          sig2.data(), static_cast<int>(sig2.size()), args.flip_prob, args.seed ^ 0x44444444ULL);
      trix_native::RoutingStats s1;
      std::vector<int> r1 = trix_native::route_argmax_with_stats(
          cfg, sig2.data(), x.data(), args.inputs, tie_break, args.seed, &s1);
      const double churn = trix_native::route_churn_rate(r0, r1);

      std::cout << "p_zero=" << p_zero << " diff_dims=" << args.diff_dims << " bias=" << args.bias
                << " margin_mean=" << s0.margin_mean << " tie_rate=" << s0.tie_rate
                << " near_tie_rate=" << s0.near_tie_rate << " churn=" << churn << "\n";

      if (writer.has_value()) {
        const char* tb = (tie_break == trix_native::TieBreak::Hash) ? "hash" : "first";
        writer->write_kv({
            {"schema_version", "1"},
            {"event", json_string("margin_sweep")},
            {"run_id", json_string(run_id)},
            {"tiles", std::to_string(cfg.tiles)},
            {"dim", std::to_string(cfg.dim)},
            {"inputs", std::to_string(args.inputs)},
            {"seed", std::to_string(args.seed)},
            {"address_type", json_string("tile_id")},
            {"tie_break", json_string(tb)},
            {"diff_dims", std::to_string(args.diff_dims)},
            {"bias", json_number(args.bias)},
            {"sweep", json_string("p_zero")},
            {"p_zero", json_number(p_zero)},
            {"flip_prob", json_number(args.flip_prob)},
            {"routing_ms", json_number(routing_ms)},
            {"routes_per_s", json_number(rps)},
            {"tie_rate", json_number(s0.tie_rate)},
            {"near_tie_rate", json_number(s0.near_tie_rate)},
            {"margin_mean", json_number(s0.margin_mean)},
            {"churn", json_number(churn)},
        });
      }
    }

    return 0;
  }

  std::cerr << "Unknown benchmark: " << args.benchmark << "\n";
  usage(argv[0]);
  return 2;
}
