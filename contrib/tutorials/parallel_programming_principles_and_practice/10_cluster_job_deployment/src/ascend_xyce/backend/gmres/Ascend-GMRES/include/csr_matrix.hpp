#pragma once

#include <course_sparse/csr_matrix.hpp>

namespace ascend_gmres {

using course_sparse::CSRMatrix;
using course_sparse::MatrixSpec;

std::vector<MatrixSpec> default_matrix_specs();
CSRMatrix generate_uniform_matrix(const MatrixSpec& spec, std::uint32_t seed = 42);
CSRMatrix generate_long_tail_matrix(const MatrixSpec& spec, std::uint32_t seed = 42);
CSRMatrix generate_block_matrix(const MatrixSpec& spec, std::int32_t block_size = 32, std::uint32_t seed = 42);
CSRMatrix make_gmres_ready_matrix(const CSRMatrix& input);

std::vector<float> generate_solution_vector(std::int32_t size, std::uint32_t seed = 42);
double relative_error(const std::vector<float>& reference, const std::vector<float>& candidate);

}  // namespace ascend_gmres
