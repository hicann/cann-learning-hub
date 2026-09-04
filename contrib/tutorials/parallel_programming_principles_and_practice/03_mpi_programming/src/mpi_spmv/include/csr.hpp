#pragma once

#include <course_sparse/csr_matrix.hpp>

namespace mpi_spmv {

using course_sparse::CSRMatrix;
using course_sparse::MatrixSpec;
using course_sparse::default_benchmark_specs;
using course_sparse::generate_block_matrix;
using course_sparse::generate_long_tail_matrix;
using course_sparse::generate_rhs_vector;
using course_sparse::generate_uniform_matrix;

}  // namespace mpi_spmv
