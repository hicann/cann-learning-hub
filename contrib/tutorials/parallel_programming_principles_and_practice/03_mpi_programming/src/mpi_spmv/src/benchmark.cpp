#include "benchmark.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace mpi_spmv {
namespace fs = std::filesystem;

std::vector<MatrixSpec> selected_specs(const std::string& matrix_name) {
    const auto specs = default_benchmark_specs();
    if (matrix_name == "all") return specs;
    for (const auto& spec : specs) {
        if (spec.name == matrix_name) return {spec};
    }
    throw std::runtime_error("unknown matrix '" + matrix_name +
                             "' (expected U1, U2, L1, L2, B1, B2, or all)");
}

CSRMatrix load_or_generate_matrix(const BenchmarkConfig& config, const MatrixSpec& spec) {
    const fs::path cache_path = fs::path(config.matrix_directory) / (spec.name + ".csrbin");
    CSRMatrix matrix;
    std::string error;
    if (fs::exists(cache_path) && CSRMatrix::load_binary(cache_path.string(), &matrix, &error)) {
        if (matrix.rows == spec.rows && matrix.cols == spec.cols && matrix.nnz() == spec.nnz) {
            return matrix;
        }
        std::cerr << "Ignoring cache with unexpected dimensions: " << cache_path << '\n';
    }

    if (spec.name.front() == 'U') {
        matrix = generate_uniform_matrix(spec, 42);
    } else if (spec.name.front() == 'L') {
        matrix = generate_long_tail_matrix(spec, 42);
    } else {
        matrix = generate_block_matrix(spec, 32, 42);
    }
    if (!matrix.validate(&error)) {
        throw std::runtime_error("generated matrix " + spec.name + " is invalid: " + error);
    }
    fs::create_directories(cache_path.parent_path());
    if (!matrix.save_binary(cache_path.string(), &error)) {
        throw std::runtime_error("failed to cache " + spec.name + ": " + error);
    }
    return matrix;
}

BenchmarkConfig parse_arguments(int argc, char** argv, const std::string& default_csv) {
    BenchmarkConfig config;
    config.csv_path = default_csv;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& option) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(option + " requires a value");
            return argv[++i];
        };
        if (arg == "--warmup") {
            config.warmup_iterations = std::stoi(require_value(arg));
        } else if (arg == "--repeat") {
            config.repeat_iterations = std::stoi(require_value(arg));
        } else if (arg == "--matrix-dir") {
            config.matrix_directory = require_value(arg);
        } else if (arg == "--csv") {
            config.csv_path = require_value(arg);
        } else if (arg == "--matrix") {
            config.matrix_name = require_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            config.matrix_name = "__help__";
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (config.warmup_iterations < 0) throw std::runtime_error("--warmup must be >= 0");
    if (config.repeat_iterations <= 0) throw std::runtime_error("--repeat must be > 0");
    return config;
}

void print_usage(const char* program, bool mpi_program) {
    std::cout << "Usage: ";
    if (mpi_program) std::cout << "mpirun -np 16 ";
    std::cout << program << " [options]\n\n"
              << "Options:\n"
              << "  --matrix NAME      U1, U2, L1, L2, B1, B2, or all (default: all)\n"
              << "  --warmup N         warmup iterations (default: 10)\n"
              << "  --repeat N         measured iterations (default: 100)\n"
              << "  --matrix-dir PATH  CSR cache directory (default: matrices)\n"
              << "  --csv PATH         result CSV path\n"
              << "  -h, --help         show this help\n";
}

void ensure_parent_directory(const std::string& path) {
    const fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) fs::create_directories(parent);
}

}  // namespace mpi_spmv
