#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace course_sparse {

struct MatrixSpec {
    std::string name;
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::int64_t nnz = 0;
};

struct CSRMatrix {
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::vector<std::int32_t> row_ptr;
    std::vector<std::int32_t> col_idx;
    std::vector<float> values;

    std::int64_t nnz() const;
    bool empty() const;
    bool validate(std::string* error = nullptr) const;
    bool save_binary(const std::string& path, std::string* error = nullptr) const;
    static bool load_binary(const std::string& path, CSRMatrix* matrix,
                            std::string* error = nullptr);
};

std::vector<MatrixSpec> default_benchmark_specs();
CSRMatrix generate_uniform_matrix(const MatrixSpec& spec, std::uint32_t seed = 42);
CSRMatrix generate_long_tail_matrix(const MatrixSpec& spec, std::uint32_t seed = 42);
CSRMatrix generate_block_matrix(const MatrixSpec& spec, std::int32_t block_size = 32,
                                std::uint32_t seed = 42);
CSRMatrix make_diagonally_dominant(const CSRMatrix& matrix);

std::vector<float> generate_rhs_vector(std::int32_t cols, std::uint32_t seed = 42);
std::vector<float> generate_signed_vector(std::int32_t size, std::uint32_t seed = 42);
std::uint16_t float32_to_bf16_bits(float value);
float bf16_bits_to_float32(std::uint16_t value);
double relative_error(const std::vector<float>& reference,
                      const std::vector<float>& candidate);

template <typename Product>
float reduce_csr_row(const std::vector<std::int32_t>& row_ptr,
                     std::int32_t row, Product product) {
    float sum = 0.0f;
    for (std::int32_t index = row_ptr[static_cast<std::size_t>(row)];
         index < row_ptr[static_cast<std::size_t>(row + 1)]; ++index) {
        sum += product(static_cast<std::size_t>(index));
    }
    return sum;
}

float csr_row_dot(const CSRMatrix& matrix, const std::vector<float>& x,
                  std::int32_t row);
void spmv_csr_reference(const CSRMatrix& matrix, const std::vector<float>& x,
                        std::vector<float>* y);

}  // namespace course_sparse
