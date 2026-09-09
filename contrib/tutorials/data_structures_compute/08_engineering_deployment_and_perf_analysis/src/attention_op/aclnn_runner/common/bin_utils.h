#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

inline std::vector<uint8_t> ReadBinary(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("Cannot open " + path);
    auto pos = f.tellg();
    if (pos == std::streampos(-1)) throw std::runtime_error("Cannot get size of " + path);
    size_t sz = static_cast<size_t>(pos);
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char *>(buf.data()), sz);
    if (!f) throw std::runtime_error("Failed to read " + path);
    return buf;
}

inline void WriteBinary(const std::string &path, const void *data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot write " + path);
    f.write(reinterpret_cast<const char *>(data), bytes);
}
