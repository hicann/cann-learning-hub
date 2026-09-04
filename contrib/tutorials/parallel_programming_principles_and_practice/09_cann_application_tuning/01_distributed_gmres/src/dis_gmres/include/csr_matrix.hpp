#pragma once

#include <course_sparse/csr_matrix.hpp>

namespace dis_gmres {

using course_sparse::CSRMatrix;
using course_sparse::MatrixSpec;

struct RowPartition {
  std::int32_t first = 0;
  std::int32_t last = 0;
  std::int64_t nnz = 0;
};

std::vector<MatrixSpec> default_matrix_specs();
const MatrixSpec* find_matrix_spec(const std::string& name);
CSRMatrix generate_matrix(const MatrixSpec& spec, std::uint32_t seed = 42);
CSRMatrix make_gmres_ready_matrix(const CSRMatrix& input);
std::vector<float> generate_solution_vector(std::int32_t size, std::uint32_t seed = 42);

std::vector<RowPartition> partition_rows(const CSRMatrix& matrix, int world_size, bool nnz_balanced);
CSRMatrix extract_rows(const CSRMatrix& matrix, std::int32_t first, std::int32_t last);

}  // namespace dis_gmres
