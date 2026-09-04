#pragma once

#include "csr.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mpi_spmv {

struct BenchmarkConfig {
    std::int32_t warmup_iterations = 10;
    std::int32_t repeat_iterations = 100;
    std::string matrix_directory = "matrices";
    std::string csv_path;
    std::string matrix_name = "all";
};

std::vector<MatrixSpec> selected_specs(const std::string& matrix_name);
CSRMatrix load_or_generate_matrix(const BenchmarkConfig& config, const MatrixSpec& spec);
BenchmarkConfig parse_arguments(int argc, char** argv, const std::string& default_csv);
void print_usage(const char* program, bool mpi_program);
void ensure_parent_directory(const std::string& path);

}  // namespace mpi_spmv
