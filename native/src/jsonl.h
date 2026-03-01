#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace trix_native {

struct JsonlWriter {
  explicit JsonlWriter(const std::string& path);
  ~JsonlWriter();
  JsonlWriter(const JsonlWriter&) = delete;
  JsonlWriter& operator=(const JsonlWriter&) = delete;

  void write_kv(const std::vector<std::pair<std::string, std::string>>& fields);

private:
  void* fp_ = nullptr;
};

std::string json_escape(const std::string& s);

}  // namespace trix_native
