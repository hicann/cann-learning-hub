#include <course_sparse/cpu_backend.hpp>

#include <chrono>

namespace course_sparse {

bool validate_spmv_run(const CSRMatrix& matrix, const std::vector<float>& x,
                       const std::vector<float>* y, std::string* error) {
    return validate_spmv_buffers(matrix.cols, x.size(), y, error);
}

bool validate_spmv_buffers(std::int32_t columns, std::size_t input_size,
                           const void* output, std::string* error) {
    if (output == nullptr) {
        if (error) *error = "output vector is null";
        return false;
    }
    if (input_size != static_cast<std::size_t>(columns)) {
        if (error) *error = "input vector size mismatch";
        return false;
    }
    return true;
}

void set_backend_timings(double initialization_ms, double transfer_in_ms,
                         double kernel_ms, double transfer_out_ms,
                         BackendTimings* timings) {
    if (timings == nullptr) return;
    timings->initialization_ms = initialization_ms;
    timings->transfer_in_ms = transfer_in_ms;
    timings->kernel_ms = kernel_ms;
    timings->transfer_out_ms = transfer_out_ms;
    timings->total_ms = transfer_in_ms + kernel_ms + transfer_out_ms;
}

void finish_backend_run(
    const std::vector<float>& host_output, std::vector<float>* output,
    double initialization_ms,
    std::chrono::steady_clock::time_point transfer_in_start,
    std::chrono::steady_clock::time_point transfer_in_end,
    std::chrono::steady_clock::time_point kernel_start,
    std::chrono::steady_clock::time_point kernel_end,
    BackendTimings* timings) {
    *output = host_output;
    const auto transfer_out_end = std::chrono::steady_clock::now();
    set_backend_timings(
        initialization_ms,
        std::chrono::duration<double, std::milli>(transfer_in_end -
                                                  transfer_in_start).count(),
        std::chrono::duration<double, std::milli>(kernel_end -
                                                  kernel_start).count(),
        std::chrono::duration<double, std::milli>(transfer_out_end -
                                                  kernel_end).count(),
        timings);
}

void set_host_compute_timing(double milliseconds, BackendTimings* timings) {
    set_backend_timings(0.0, 0.0, milliseconds, 0.0, timings);
}

std::string CpuSingleBackend::name() const { return "CPU single"; }

bool CpuSingleBackend::prepare(const CSRMatrix& matrix, std::string* error) {
    matrix_ = matrix;
    return matrix_.validate(error);
}

bool CpuSingleBackend::run(const std::vector<float>& x, std::vector<float>* y,
                           BackendTimings* timings, std::string* error) {
    if (!validate_spmv_run(matrix_, x, y, error)) return false;
    const auto start = std::chrono::steady_clock::now();
    spmv_csr_reference(matrix_, x, y);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    set_host_compute_timing(elapsed, timings);
    return true;
}

}  // namespace course_sparse
