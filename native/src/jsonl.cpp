#include "jsonl.h"

#include <cstdio>
#include <stdexcept>

namespace trix_native {

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char ch : s) {
    switch (ch) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          // Control chars: replace with a space to keep writer tiny.
          out += ' ';
        } else {
          out += ch;
        }
        break;
    }
  }
  return out;
}

JsonlWriter::JsonlWriter(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "ab");
  if (!f) {
    throw std::runtime_error("failed to open jsonl file: " + path);
  }
  fp_ = reinterpret_cast<void*>(f);
}

JsonlWriter::~JsonlWriter() {
  if (!fp_) return;
  FILE* f = reinterpret_cast<FILE*>(fp_);
  std::fclose(f);
  fp_ = nullptr;
}

void JsonlWriter::write_kv(const std::vector<std::pair<std::string, std::string>>& fields) {
  if (!fp_) throw std::runtime_error("jsonl writer is closed");
  FILE* f = reinterpret_cast<FILE*>(fp_);

  std::fputc('{', f);
  for (size_t i = 0; i < fields.size(); i++) {
    const auto& kv = fields[i];
    if (i) std::fputc(',', f);
    std::string k = json_escape(kv.first);
    std::string v = kv.second;
    std::fputc('"', f);
    std::fwrite(k.data(), 1, k.size(), f);
    std::fputc('"', f);
    std::fputc(':', f);

    // Value is expected to already be a JSON literal (quoted string or number).
    std::fwrite(v.data(), 1, v.size(), f);
  }
  std::fputc('}', f);
  std::fputc('\n', f);
  std::fflush(f);
}

}  // namespace trix_native
