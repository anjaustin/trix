#include "jsonl.h"
#include "metrics.h"
#include "routing.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string benchmark = "routing";  // routing|stability
  int tiles = 64;
  int dim = 512;
  int inputs = 4096;
  uint64_t seed = 1;
  double flip_prob = 0.01;
  std::optional<std::string> jsonl;
};

static void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--benchmark routing|stability] [--tiles N] [--dim N] [--inputs N] [--seed N]"
         " [--flip-prob P] [--jsonl PATH]\n";
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

  const int sig_n = cfg.tiles * cfg.dim;
  const int in_n = args.inputs * cfg.dim;

  std::vector<int8_t> signatures = trix_native::make_random_ternary(sig_n, args.seed ^ 0xA5A5A5A5ULL);
  std::vector<int8_t> inputs = trix_native::make_random_ternary(in_n, args.seed ^ 0x5A5A5A5AULL);

  std::optional<trix_native::JsonlWriter> writer;
  if (args.jsonl.has_value()) {
    writer.emplace(*args.jsonl);
  }

  auto t0 = std::chrono::steady_clock::now();
  std::vector<int> routes0 = trix_native::route_argmax(cfg, signatures.data(), inputs.data(), args.inputs);
  auto t1 = std::chrono::steady_clock::now();

  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double routes_per_s = (static_cast<double>(args.inputs) / ms) * 1000.0;

  trix_native::UsageMetrics um = trix_native::usage_metrics(routes0, cfg.tiles);

  if (args.benchmark == "routing") {
    std::cout << "routing_ms=" << ms << " routes_per_s=" << routes_per_s << " entropy_nats=" << um.entropy_nats
              << " gini=" << um.gini << " max_tile=" << um.max_tile << " max_count=" << um.max_count << "\n";

    if (writer.has_value()) {
      writer->write_kv({
          {"event", json_string("routing")},
          {"tiles", std::to_string(cfg.tiles)},
          {"dim", std::to_string(cfg.dim)},
          {"inputs", std::to_string(args.inputs)},
          {"seed", std::to_string(args.seed)},
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

    std::vector<int> routes1 = trix_native::route_argmax(cfg, signatures2.data(), inputs.data(), args.inputs);
    const double churn = trix_native::route_churn_rate(routes0, routes1);

    std::cout << "routing_ms=" << ms << " routes_per_s=" << routes_per_s << " churn=" << churn
              << " flip_prob=" << args.flip_prob << "\n";

    if (writer.has_value()) {
      writer->write_kv({
          {"event", json_string("stability")},
          {"tiles", std::to_string(cfg.tiles)},
          {"dim", std::to_string(cfg.dim)},
          {"inputs", std::to_string(args.inputs)},
          {"seed", std::to_string(args.seed)},
          {"flip_prob", json_number(args.flip_prob)},
          {"routing_ms", json_number(ms)},
          {"routes_per_s", json_number(routes_per_s)},
          {"churn", json_number(churn)},
      });
    }

    return 0;
  }

  std::cerr << "Unknown benchmark: " << args.benchmark << "\n";
  usage(argv[0]);
  return 2;
}
