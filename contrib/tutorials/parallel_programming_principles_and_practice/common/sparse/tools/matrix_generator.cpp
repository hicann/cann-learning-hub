#include <course_sparse/csr_matrix.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef COURSE_SPARSE_GENERATOR_KIND
#error "COURSE_SPARSE_GENERATOR_KIND must be 0 (uniform), 1 (long-tail), or 2 (block)"
#endif

namespace {

constexpr char matrix_prefix() {
#if COURSE_SPARSE_GENERATOR_KIND == 0
    return 'U';
#elif COURSE_SPARSE_GENERATOR_KIND == 1
    return 'L';
#elif COURSE_SPARSE_GENERATOR_KIND == 2
    return 'B';
#else
#error "Unsupported COURSE_SPARSE_GENERATOR_KIND"
#endif
}

course_sparse::CSRMatrix generate(const course_sparse::MatrixSpec& spec) {
    if constexpr (COURSE_SPARSE_GENERATOR_KIND == 0) {
        return course_sparse::generate_uniform_matrix(spec);
    } else if constexpr (COURSE_SPARSE_GENERATOR_KIND == 1) {
        return course_sparse::generate_long_tail_matrix(spec);
    } else {
        return course_sparse::generate_block_matrix(spec);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) {
            std::cerr << "usage: " << argv[0] << " [OUTPUT_DIRECTORY]\n";
            return 2;
        }
        const std::filesystem::path output_directory = argc == 2 ? argv[1] : "matrices";
        std::filesystem::create_directories(output_directory);
        for (const auto& spec : course_sparse::default_benchmark_specs()) {
            if (spec.name.empty() || spec.name.front() != matrix_prefix()) continue;
            const auto output = output_directory / (spec.name + ".csrbin");
            const auto matrix = generate(spec);
            std::string error;
            if (!matrix.save_binary(output.string(), &error)) throw std::runtime_error(error);
            std::cout << "generated " << output.string() << '\n';
        }
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "matrix generation failed: " << exception.what() << '\n';
        return 1;
    }
}
