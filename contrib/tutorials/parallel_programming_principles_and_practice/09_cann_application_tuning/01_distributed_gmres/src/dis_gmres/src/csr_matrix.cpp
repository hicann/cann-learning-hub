#include "csr_matrix.hpp"

#include <algorithm>

namespace dis_gmres {

std::vector<MatrixSpec> default_matrix_specs() {
  return course_sparse::default_benchmark_specs();
}

const MatrixSpec* find_matrix_spec(const std::string& name) {
  static const auto specs = default_matrix_specs();
  const auto it = std::find_if(specs.begin(), specs.end(), [&](const MatrixSpec& spec) {
    return spec.name == name;
  });
  return it == specs.end() ? nullptr : &*it;
}

CSRMatrix generate_matrix(const MatrixSpec& spec, std::uint32_t seed) {
  CSRMatrix matrix;
  if (!spec.name.empty() && spec.name.front() == 'L') {
    matrix = course_sparse::generate_long_tail_matrix(spec, seed);
  } else if (!spec.name.empty() && spec.name.front() == 'B') {
    matrix = course_sparse::generate_block_matrix(spec, 32, seed);
  } else {
    matrix = course_sparse::generate_uniform_matrix(spec, seed);
  }
  return make_gmres_ready_matrix(matrix);
}

CSRMatrix make_gmres_ready_matrix(const CSRMatrix& input) {
  return course_sparse::make_diagonally_dominant(input);
}

std::vector<float> generate_solution_vector(std::int32_t size, std::uint32_t seed) {
  return course_sparse::generate_signed_vector(size, seed);
}

std::vector<RowPartition> partition_rows(const CSRMatrix& matrix, int world_size,
                                         bool nnz_balanced) {
  int partition_count = world_size;
  if (partition_count <= 0) partition_count = 1;
  std::vector<RowPartition> result(static_cast<std::size_t>(partition_count));
  std::int32_t first = 0;
  for (int rank = 0; rank < partition_count; ++rank) {
    std::int32_t last = matrix.rows;
    if (rank + 1 < partition_count) {
      if (nnz_balanced) {
        const auto target = matrix.nnz() * (rank + 1) / partition_count;
        last = static_cast<std::int32_t>(
            std::lower_bound(matrix.row_ptr.begin(), matrix.row_ptr.end(), target) -
            matrix.row_ptr.begin());
        last = std::max(first, std::min(last, matrix.rows));
      } else {
        last = static_cast<std::int32_t>(
            static_cast<std::int64_t>(matrix.rows) * (rank + 1) / partition_count);
      }
    }
    result[static_cast<std::size_t>(rank)] =
        {first, last, matrix.row_ptr[static_cast<std::size_t>(last)] -
                          matrix.row_ptr[static_cast<std::size_t>(first)]};
    first = last;
  }
  return result;
}

CSRMatrix extract_rows(const CSRMatrix& matrix, std::int32_t first, std::int32_t last) {
  first = std::max<std::int32_t>(0, first);
  last = std::max(first, std::min(last, matrix.rows));
  CSRMatrix local;
  local.rows = last - first;
  local.cols = matrix.cols;
  const auto nnz_first = matrix.row_ptr[static_cast<std::size_t>(first)];
  const auto nnz_last = matrix.row_ptr[static_cast<std::size_t>(last)];
  local.row_ptr.resize(static_cast<std::size_t>(local.rows + 1));
  for (std::int32_t row = 0; row <= local.rows; ++row) {
    local.row_ptr[static_cast<std::size_t>(row)] =
        matrix.row_ptr[static_cast<std::size_t>(first + row)] - nnz_first;
  }
  local.col_idx.assign(matrix.col_idx.begin() + nnz_first,
                       matrix.col_idx.begin() + nnz_last);
  local.values.assign(matrix.values.begin() + nnz_first,
                      matrix.values.begin() + nnz_last);
  return local;
}

}  // namespace dis_gmres
