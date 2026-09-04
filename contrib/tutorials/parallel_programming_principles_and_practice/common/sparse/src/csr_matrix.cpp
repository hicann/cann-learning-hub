#include <course_sparse/csr_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace course_sparse {

namespace {

template <typename T>
bool write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return static_cast<bool>(out);
}

template <typename T>
bool read_pod(std::ifstream& in, T* value) {
    in.read(reinterpret_cast<char*>(value), sizeof(T));
    return static_cast<bool>(in);
}

std::vector<std::int32_t> sample_unique_columns(std::int32_t cols, std::int32_t count, std::mt19937& rng) {
    std::vector<std::int32_t> sampled;
    sampled.reserve(count);
    std::uniform_int_distribution<std::int32_t> distribution(0, cols - 1);

    while (static_cast<std::int32_t>(sampled.size()) < count) {
        const std::int32_t candidate = distribution(rng);
        if (std::find(sampled.begin(), sampled.end(), candidate) == sampled.end()) {
            sampled.push_back(candidate);
        }
    }

    std::sort(sampled.begin(), sampled.end());
    return sampled;
}

void adjust_counts_to_target(std::vector<std::int32_t>& counts, std::int64_t target, std::int32_t min_count, std::int32_t max_count) {
    std::mt19937 rng(42);
    std::vector<std::int32_t> order(counts.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    auto current_sum = [&counts]() -> std::int64_t {
        return std::accumulate(counts.begin(), counts.end(), static_cast<std::int64_t>(0));
    };

    std::int64_t sum = current_sum();
    if (sum < target) {
        std::size_t cursor = 0;
        while (sum < target) {
            const std::int32_t row = order[cursor % order.size()];
            if (counts[row] < max_count) {
                ++counts[row];
                ++sum;
            }
            ++cursor;
        }
    } else if (sum > target) {
        std::size_t cursor = 0;
        while (sum > target) {
            const std::int32_t row = order[cursor % order.size()];
            if (counts[row] > min_count) {
                --counts[row];
                --sum;
            }
            ++cursor;
        }
    }
}

CSRMatrix generate_from_row_counts(const MatrixSpec& spec,
                                   const std::vector<std::int32_t>& counts,
                                   std::mt19937& rng, float minimum_value,
                                   float maximum_value) {
    CSRMatrix matrix;
    matrix.rows = spec.rows;
    matrix.cols = spec.cols;
    matrix.row_ptr.resize(static_cast<std::size_t>(spec.rows + 1));
    std::partial_sum(counts.begin(), counts.end(), matrix.row_ptr.begin() + 1);
    matrix.col_idx.resize(static_cast<std::size_t>(spec.nnz));
    matrix.values.resize(static_cast<std::size_t>(spec.nnz));

    std::uniform_real_distribution<float> value_distribution(minimum_value,
                                                              maximum_value);
    for (std::int32_t row = 0; row < spec.rows; ++row) {
        const auto count = counts[static_cast<std::size_t>(row)];
        const auto columns = sample_unique_columns(spec.cols, count, rng);
        const auto begin = matrix.row_ptr[static_cast<std::size_t>(row)];
        for (std::int32_t offset = 0; offset < count; ++offset) {
            const auto index = static_cast<std::size_t>(begin + offset);
            matrix.col_idx[index] = columns[static_cast<std::size_t>(offset)];
            matrix.values[index] = value_distribution(rng);
        }
    }
    return matrix;
}

}  // namespace

std::vector<MatrixSpec> default_benchmark_specs() {
    return {{"U1", 100000, 100000, 1000000}, {"U2", 1000000, 1000000, 10000000},
            {"L1", 100000, 100000, 1000000}, {"L2", 1000000, 1000000, 10000000},
            {"B1", 100000, 100000, 1000000}, {"B2", 1000000, 1000000, 10000000}};
}

std::int64_t CSRMatrix::nnz() const {
    return static_cast<std::int64_t>(col_idx.size());
}

bool CSRMatrix::empty() const {
    return rows == 0 || cols == 0 || col_idx.empty();
}

bool CSRMatrix::validate(std::string* error) const {
    if (rows < 0 || cols < 0) {
        if (error) {
            *error = "matrix dimensions must be non-negative";
        }
        return false;
    }
    if (row_ptr.size() != static_cast<std::size_t>(rows + 1)) {
        if (error) {
            *error = "row_ptr size mismatch";
        }
        return false;
    }
    if (row_ptr.empty() || row_ptr.front() != 0) {
        if (error) {
            *error = "row_ptr must start at 0";
        }
        return false;
    }
    for (std::size_t i = 1; i < row_ptr.size(); ++i) {
        if (row_ptr[i] < row_ptr[i - 1]) {
            if (error) {
                *error = "row_ptr must be non-decreasing";
            }
            return false;
        }
    }
    if (static_cast<std::size_t>(row_ptr.back()) != col_idx.size() || col_idx.size() != values.size()) {
        if (error) {
            *error = "nnz size mismatch";
        }
        return false;
    }
    for (std::int64_t i = 0; i < nnz(); ++i) {
        if (col_idx[static_cast<std::size_t>(i)] < 0 || col_idx[static_cast<std::size_t>(i)] >= cols) {
            if (error) {
                *error = "column index out of range";
            }
            return false;
        }
    }
    return true;
}

bool CSRMatrix::save_binary(const std::string& path, std::string* error) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (error) {
            *error = "failed to open file for writing: " + path;
        }
        return false;
    }

    const char magic[4] = {'C', 'S', 'R', '1'};
    out.write(magic, sizeof(magic));
    if (!write_pod(out, rows) || !write_pod(out, cols)) {
        if (error) {
            *error = "failed to write header";
        }
        return false;
    }

    const std::int64_t nnz_value = nnz();
    if (!write_pod(out, nnz_value)) {
        if (error) {
            *error = "failed to write nnz";
        }
        return false;
    }

    out.write(reinterpret_cast<const char*>(row_ptr.data()), static_cast<std::streamsize>(row_ptr.size() * sizeof(std::int32_t)));
    out.write(reinterpret_cast<const char*>(col_idx.data()), static_cast<std::streamsize>(col_idx.size() * sizeof(std::int32_t)));
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));

    if (!out) {
        if (error) {
            *error = "failed to write matrix payload";
        }
        return false;
    }
    return true;
}

bool CSRMatrix::load_binary(const std::string& path, CSRMatrix* matrix, std::string* error) {
    if (matrix == nullptr) {
        if (error) {
            *error = "output matrix pointer is null";
        }
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error) {
            *error = "failed to open file for reading: " + path;
        }
        return false;
    }

    char magic[4] = {};
    in.read(magic, sizeof(magic));
    if (!in || magic[0] != 'C' || magic[1] != 'S' || magic[2] != 'R' || magic[3] != '1') {
        if (error) {
            *error = "invalid CSR file magic";
        }
        return false;
    }

    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::int64_t nnz_value = 0;
    if (!read_pod(in, &rows) || !read_pod(in, &cols) || !read_pod(in, &nnz_value)) {
        if (error) {
            *error = "failed to read matrix header";
        }
        return false;
    }

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->row_ptr.resize(static_cast<std::size_t>(rows + 1));
    matrix->col_idx.resize(static_cast<std::size_t>(nnz_value));
    matrix->values.resize(static_cast<std::size_t>(nnz_value));

    in.read(reinterpret_cast<char*>(matrix->row_ptr.data()), static_cast<std::streamsize>(matrix->row_ptr.size() * sizeof(std::int32_t)));
    in.read(reinterpret_cast<char*>(matrix->col_idx.data()), static_cast<std::streamsize>(matrix->col_idx.size() * sizeof(std::int32_t)));
    in.read(reinterpret_cast<char*>(matrix->values.data()), static_cast<std::streamsize>(matrix->values.size() * sizeof(float)));

    if (!in) {
        if (error) {
            *error = "failed to read matrix payload";
        }
        return false;
    }

    return matrix->validate(error);
}

CSRMatrix generate_uniform_matrix(const MatrixSpec& spec, std::uint32_t seed) {
    std::vector<std::int32_t> counts(static_cast<std::size_t>(spec.rows), static_cast<std::int32_t>(spec.nnz / spec.rows));
    const std::int32_t remainder = static_cast<std::int32_t>(spec.nnz % spec.rows);

    std::vector<std::int32_t> order(static_cast<std::size_t>(spec.rows));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    for (std::int32_t i = 0; i < remainder; ++i) {
        ++counts[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
    }
    return generate_from_row_counts(spec, counts, rng, 0.5f, 1.5f);
}

CSRMatrix generate_long_tail_matrix(const MatrixSpec& spec, std::uint32_t seed) {
    constexpr std::int32_t min_count = 1;
    constexpr std::int32_t max_count = 512;

    std::vector<double> weights(static_cast<std::size_t>(max_count));
    for (std::int32_t i = 1; i <= max_count; ++i) {
        weights[static_cast<std::size_t>(i - 1)] = 1.0 / std::pow(static_cast<double>(i), 1.35);
    }

    std::mt19937 rng(seed);
    std::discrete_distribution<std::int32_t> distribution(weights.begin(), weights.end());
    std::vector<std::int32_t> counts(static_cast<std::size_t>(spec.rows));
    for (std::int32_t row = 0; row < spec.rows; ++row) {
        counts[static_cast<std::size_t>(row)] = min_count + distribution(rng);
    }

    adjust_counts_to_target(counts, spec.nnz, min_count, max_count);
    // Keep long rows contiguous so scheduling experiments expose load imbalance.
    std::sort(counts.rbegin(), counts.rend());
    return generate_from_row_counts(spec, counts, rng, 0.25f, 2.0f);
}

namespace {

struct Block {
    std::int32_t row_block = 0;
    std::int32_t col_block = 0;
    std::int32_t fill_count = 0;
};

struct BlockLayout {
    std::int32_t size = 0;
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::int64_t area = 0;
    std::int64_t full_blocks = 0;
    std::int32_t remainder = 0;
};

BlockLayout make_block_layout(const MatrixSpec& spec, std::int32_t block_size) {
    if (block_size == 0) throw std::invalid_argument("block size must not be zero");
    if (block_size < 0 || spec.rows <= 0 || spec.cols <= 0) {
        throw std::invalid_argument("block size and matrix dimensions must be positive");
    }
    if (block_size > spec.rows || block_size > spec.cols) {
        throw std::invalid_argument("block size and matrix dimensions must be positive");
    }
    const std::int64_t matrix_area = static_cast<std::int64_t>(spec.rows) * spec.cols;
    if (spec.nnz < 0 || spec.nnz > matrix_area) {
        throw std::invalid_argument("nnz must fit within the matrix dimensions");
    }
    BlockLayout layout;
    layout.size = block_size;
    layout.rows = spec.rows / block_size;
    layout.cols = spec.cols / block_size;
    layout.area = static_cast<std::int64_t>(block_size) * block_size;
    layout.full_blocks = spec.nnz / layout.area;
    layout.remainder = static_cast<std::int32_t>(spec.nnz % layout.area);
    const std::int64_t required = layout.full_blocks + (layout.remainder > 0 ? 1 : 0);
    if (required > static_cast<std::int64_t>(layout.rows) * layout.cols) {
        throw std::invalid_argument("nnz exceeds the capacity of the requested block layout");
    }
    return layout;
}

std::pair<std::int32_t, std::int32_t> sample_unique_block(
        const BlockLayout& layout, std::mt19937& rng,
        std::uniform_int_distribution<std::int32_t>& row_distribution,
        std::uniform_int_distribution<std::int32_t>& col_distribution,
        std::unordered_set<std::int64_t>* chosen, std::int32_t* fallback_row,
        std::int32_t* fallback_col) {
    for (std::int32_t attempt = 0; attempt < 64; ++attempt) {
        const std::int32_t row = row_distribution(rng);
        const std::int32_t col = col_distribution(rng);
        const std::int64_t key = static_cast<std::int64_t>(row) * layout.cols + col;
        if (chosen->insert(key).second) return {row, col};
    }
    for (; *fallback_row < layout.rows; ++*fallback_row) {
        for (; *fallback_col < layout.cols;) {
            const std::int32_t col = (*fallback_col)++;
            const std::int64_t key =
                static_cast<std::int64_t>(*fallback_row) * layout.cols + col;
            if (chosen->insert(key).second) return {*fallback_row, col};
        }
        *fallback_col = 0;
    }
    throw std::runtime_error("failed to select a unique matrix block");
}

std::vector<Block> select_blocks(const BlockLayout& layout, std::mt19937& rng) {
    const std::int64_t count = layout.full_blocks + (layout.remainder > 0 ? 1 : 0);
    std::uniform_int_distribution<std::int32_t> row_distribution(0, layout.rows - 1);
    std::uniform_int_distribution<std::int32_t> col_distribution(0, layout.cols - 1);
    std::unordered_set<std::int64_t> chosen;
    chosen.reserve(static_cast<std::size_t>(count) * 2);
    std::vector<Block> blocks;
    blocks.reserve(static_cast<std::size_t>(count));
    std::int32_t fallback_row = 0;
    std::int32_t fallback_col = 0;
    auto sample = [&]() {
        return sample_unique_block(layout, rng, row_distribution, col_distribution,
                                   &chosen, &fallback_row, &fallback_col);
    };
    for (std::int64_t i = 0; i < layout.full_blocks; ++i) {
        const auto [row, col] = sample();
        blocks.push_back({row, col, static_cast<std::int32_t>(layout.area)});
    }
    if (layout.remainder > 0) {
        const auto [row, col] = sample();
        blocks.push_back({row, col, layout.remainder});
    }
    std::sort(blocks.begin(), blocks.end(), [](const Block& a, const Block& b) {
        return a.row_block == b.row_block ? a.col_block < b.col_block
                                          : a.row_block < b.row_block;
    });
    return blocks;
}

std::vector<std::int32_t> count_block_rows(const MatrixSpec& spec,
                                           const BlockLayout& layout,
                                           const std::vector<Block>& blocks) {
    std::vector<std::int32_t> counts(static_cast<std::size_t>(spec.rows), 0);
    for (const Block& block : blocks) {
        const std::int32_t start_row = block.row_block * layout.size;
        const std::int32_t start_col = block.col_block * layout.size;
        const std::int32_t row_extent = std::min(layout.size, spec.rows - start_row);
        const std::int32_t col_extent = std::min(layout.size, spec.cols - start_col);
        if (col_extent == 0) continue;
        const std::int32_t fill = std::min(block.fill_count, row_extent * col_extent);
        for (std::int32_t index = 0; index < fill; ++index) {
            ++counts[static_cast<std::size_t>(start_row + index / col_extent)];
        }
    }
    return counts;
}

CSRMatrix allocate_block_matrix(const MatrixSpec& spec,
                                const std::vector<std::int32_t>& row_counts) {
    CSRMatrix matrix;
    matrix.rows = spec.rows;
    matrix.cols = spec.cols;
    matrix.row_ptr.resize(static_cast<std::size_t>(spec.rows + 1));
    std::partial_sum(row_counts.begin(), row_counts.end(), matrix.row_ptr.begin() + 1);
    matrix.col_idx.resize(static_cast<std::size_t>(spec.nnz));
    matrix.values.resize(static_cast<std::size_t>(spec.nnz));
    return matrix;
}

void fill_blocks(const MatrixSpec& spec, const BlockLayout& layout,
                 const std::vector<Block>& blocks, std::mt19937& rng,
                 CSRMatrix* matrix) {
    std::vector<std::int32_t> offsets = matrix->row_ptr;
    std::uniform_real_distribution<float> value_distribution(0.1f, 1.0f);
    for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        const Block& block = blocks[block_index];
        const std::int32_t start_row = block.row_block * layout.size;
        const std::int32_t start_col = block.col_block * layout.size;
        const std::int32_t row_extent = std::min(layout.size, spec.rows - start_row);
        const std::int32_t col_extent = std::min(layout.size, spec.cols - start_col);
        if (col_extent == 0) continue;
        const std::int32_t fill = std::min(block.fill_count, row_extent * col_extent);
        for (std::int32_t index = 0; index < fill; ++index) {
            const std::int32_t row = start_row + index / col_extent;
            const std::int32_t col = start_col + index % col_extent;
            const std::size_t out = static_cast<std::size_t>(offsets[row]++);
            matrix->col_idx[out] = col;
            matrix->values[out] = value_distribution(rng) +
                                  static_cast<float>((block_index % 11) * 0.01);
        }
    }
}

}  // namespace

CSRMatrix generate_block_matrix(const MatrixSpec& spec, std::int32_t block_size,
                                std::uint32_t seed) {
    const BlockLayout layout = make_block_layout(spec, block_size);
    std::mt19937 rng(seed);
    const std::vector<Block> blocks = select_blocks(layout, rng);
    const std::vector<std::int32_t> row_counts = count_block_rows(spec, layout, blocks);
    CSRMatrix matrix = allocate_block_matrix(spec, row_counts);
    fill_blocks(spec, layout, blocks, rng, &matrix);
    return matrix;
}

CSRMatrix make_diagonally_dominant(const CSRMatrix& matrix) {
    std::vector<std::vector<std::pair<std::int32_t, float>>> rows(
        static_cast<std::size_t>(matrix.rows));
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        float off_diagonal_sum = 0.0f;
        bool has_diagonal = false;
        for (std::int32_t index = matrix.row_ptr[static_cast<std::size_t>(row)];
             index < matrix.row_ptr[static_cast<std::size_t>(row + 1)]; ++index) {
            const auto column = matrix.col_idx[static_cast<std::size_t>(index)];
            const auto value = matrix.values[static_cast<std::size_t>(index)];
            has_diagonal = has_diagonal || column == row;
            if (column != row) off_diagonal_sum += std::fabs(value);
            rows[static_cast<std::size_t>(row)].push_back({column, value});
        }
        const float diagonal = 1.0f + off_diagonal_sum;
        if (has_diagonal) {
            for (auto& entry : rows[static_cast<std::size_t>(row)]) {
                if (entry.first == row) {
                    entry.second = diagonal;
                    break;
                }
            }
        } else {
            rows[static_cast<std::size_t>(row)].push_back({row, diagonal});
        }
        std::sort(rows[static_cast<std::size_t>(row)].begin(),
                  rows[static_cast<std::size_t>(row)].end());
    }

    CSRMatrix result;
    result.rows = matrix.rows;
    result.cols = matrix.cols;
    result.row_ptr.resize(static_cast<std::size_t>(result.rows + 1));
    for (std::int32_t row = 0; row < result.rows; ++row) {
        result.row_ptr[static_cast<std::size_t>(row + 1)] =
            result.row_ptr[static_cast<std::size_t>(row)] +
            static_cast<std::int32_t>(rows[static_cast<std::size_t>(row)].size());
    }
    result.col_idx.resize(static_cast<std::size_t>(result.row_ptr.back()));
    result.values.resize(result.col_idx.size());
    for (std::int32_t row = 0; row < result.rows; ++row) {
        auto index = result.row_ptr[static_cast<std::size_t>(row)];
        for (const auto& entry : rows[static_cast<std::size_t>(row)]) {
            result.col_idx[static_cast<std::size_t>(index)] = entry.first;
            result.values[static_cast<std::size_t>(index++)] = entry.second;
        }
    }
    return result;
}

std::vector<float> generate_rhs_vector(std::int32_t cols, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> rhs(static_cast<std::size_t>(cols));
    for (float& value : rhs) {
        value = distribution(rng);
    }
    return rhs;
}

std::vector<float> generate_signed_vector(std::int32_t size, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> values(static_cast<std::size_t>(size));
    for (float& value : values) value = distribution(rng);
    return values;
}

std::uint16_t float32_to_bf16_bits(float value) {
    std::uint32_t bits = 0;
    std::copy_n(reinterpret_cast<const unsigned char*>(&value), sizeof(value),
                reinterpret_cast<unsigned char*>(&bits));
    bits += ((bits >> 16) & 1u) + 0x7fffu;
    return static_cast<std::uint16_t>(bits >> 16);
}

float bf16_bits_to_float32(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result = 0.0f;
    std::copy_n(reinterpret_cast<const unsigned char*>(&bits), sizeof(bits),
                reinterpret_cast<unsigned char*>(&result));
    return result;
}

double relative_error(const std::vector<float>& reference, const std::vector<float>& candidate) {
    if (reference.size() != candidate.size()) {
        return std::numeric_limits<double>::infinity();
    }

    double max_error = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double ref = static_cast<double>(reference[i]);
        const double cand = static_cast<double>(candidate[i]);
        const double denominator = std::abs(ref);
        const double difference = std::abs(ref - cand);
        double error = difference / 1e-12;
        if (denominator > 1e-12) {
            error = difference / denominator;
        }
        max_error = std::max(max_error, error);
    }
    return max_error;
}

float csr_row_dot(const CSRMatrix& matrix, const std::vector<float>& x,
                  std::int32_t row) {
    return reduce_csr_row(matrix.row_ptr, row, [&](std::size_t index) {
        return matrix.values[index] *
               x[static_cast<std::size_t>(matrix.col_idx[index])];
    });
}

void spmv_csr_reference(const CSRMatrix& matrix, const std::vector<float>& x,
                        std::vector<float>* y) {
    if (y == nullptr) return;
    y->assign(static_cast<std::size_t>(matrix.rows), 0.0f);
    for (std::int32_t row = 0; row < matrix.rows; ++row) {
        (*y)[static_cast<std::size_t>(row)] = csr_row_dot(matrix, x, row);
    }
}

}  // namespace course_sparse
