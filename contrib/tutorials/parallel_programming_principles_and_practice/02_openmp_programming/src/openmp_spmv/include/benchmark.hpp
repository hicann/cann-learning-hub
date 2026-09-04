#pragma once

#include "spmv.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace spmv {

struct BenchmarkConfig {
    std::int32_t warmup_iterations = 10;
    std::int32_t repeat_iterations = 100;
    std::string matrix_directory = "matrices";
    std::string results_directory = "results";
    std::string csv_path = "results/openmp_spmv.csv";
};

}  // namespace spmv
