#include "benchmark.hpp"
#include "spmv.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        const auto config = mpi_spmv::parse_arguments(argc, argv, "results/cpu_spmv.csv");
        if (config.matrix_name == "__help__") {
            mpi_spmv::print_usage(argv[0], false);
            return 0;
        }
        mpi_spmv::ensure_parent_directory(config.csv_path);
        std::ofstream csv(config.csv_path);
        if (!csv) throw std::runtime_error("failed to open CSV: " + config.csv_path);
        csv << "matrix,rows,cols,nnz,warmup,repeat,cpu_time_ms\n";

        std::cout << std::fixed << std::setprecision(6);
        for (const auto& spec : mpi_spmv::selected_specs(config.matrix_name)) {
            const auto matrix = mpi_spmv::load_or_generate_matrix(config, spec);
            const auto x = mpi_spmv::generate_rhs_vector(matrix.cols, 42);
            std::vector<float> y;
            for (std::int32_t i = 0; i < config.warmup_iterations; ++i) {
                mpi_spmv::spmv_cpu(matrix, x, &y);
            }
            double elapsed_ms = 0.0;
            for (std::int32_t i = 0; i < config.repeat_iterations; ++i) {
                const auto start = std::chrono::steady_clock::now();
                mpi_spmv::spmv_cpu(matrix, x, &y);
                const auto end = std::chrono::steady_clock::now();
                elapsed_ms += std::chrono::duration<double, std::milli>(end - start).count();
            }
            const double average_ms = elapsed_ms / config.repeat_iterations;
            std::cout << "CPU SpMV:\n"
                      << "  matrix = " << spec.name << '\n'
                      << "  matrix_size = " << matrix.rows << " x " << matrix.cols << '\n'
                      << "  nnz = " << matrix.nnz() << '\n'
                      << "  time = " << average_ms << " ms\n";
            csv << spec.name << ',' << matrix.rows << ',' << matrix.cols << ',' << matrix.nnz()
                << ',' << config.warmup_iterations << ',' << config.repeat_iterations << ','
                << std::setprecision(9) << average_ms << '\n';
            std::cout << std::setprecision(6);
        }
        std::cout << "CSV saved to " << config.csv_path << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "CPU benchmark failed: " << exception.what() << '\n';
        return 1;
    }
}
