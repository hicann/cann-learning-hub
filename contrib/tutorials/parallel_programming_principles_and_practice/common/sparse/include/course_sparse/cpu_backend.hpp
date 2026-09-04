#pragma once

#include <course_sparse/csr_matrix.hpp>

#include <chrono>
#include <string>
#include <vector>

namespace course_sparse {

struct BackendTimings {
    double initialization_ms = 0.0;
    double transfer_in_ms = 0.0;
    double kernel_ms = 0.0;
    double transfer_out_ms = 0.0;
    double total_ms = 0.0;
    std::int32_t actual_threads = 1;
    std::int32_t schedule_kind = 0;
    std::int32_t schedule_chunk = 0;
};

class ISpmvBackend {
public:
    virtual ~ISpmvBackend() = default;
    virtual std::string name() const = 0;
    virtual bool prepare(const CSRMatrix& matrix, std::string* error = nullptr) = 0;
    virtual bool run(const std::vector<float>& x, std::vector<float>* y,
                     BackendTimings* timings, std::string* error = nullptr) = 0;
    virtual double initialization_ms() const { return 0.0; }
};

class CpuSingleBackend final : public ISpmvBackend {
public:
    std::string name() const override;
    bool prepare(const CSRMatrix& matrix, std::string* error = nullptr) override;
    bool run(const std::vector<float>& x, std::vector<float>* y,
             BackendTimings* timings, std::string* error = nullptr) override;

private:
    CSRMatrix matrix_;
};

bool validate_spmv_run(const CSRMatrix& matrix, const std::vector<float>& x,
                       const std::vector<float>* y, std::string* error);
bool validate_spmv_buffers(std::int32_t columns, std::size_t input_size,
                           const void* output, std::string* error);
void set_backend_timings(double initialization_ms, double transfer_in_ms,
                         double kernel_ms, double transfer_out_ms,
                         BackendTimings* timings);
void finish_backend_run(
    const std::vector<float>& host_output, std::vector<float>* output,
    double initialization_ms,
    std::chrono::steady_clock::time_point transfer_in_start,
    std::chrono::steady_clock::time_point transfer_in_end,
    std::chrono::steady_clock::time_point kernel_start,
    std::chrono::steady_clock::time_point kernel_end,
    BackendTimings* timings);
void set_host_compute_timing(double milliseconds, BackendTimings* timings);

template <typename Converter>
void convert_fp32_vector(const std::vector<float>& source,
                         std::vector<std::uint16_t>* destination,
                         Converter convert) {
    destination->resize(source.size());
    for (std::size_t index = 0; index < source.size(); ++index) {
        (*destination)[index] = convert(source[index]);
    }
}

}  // namespace course_sparse
