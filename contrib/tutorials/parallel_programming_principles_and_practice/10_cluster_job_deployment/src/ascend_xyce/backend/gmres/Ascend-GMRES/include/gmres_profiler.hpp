#pragma once

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <string>

namespace ascend_gmres {

struct GMRESProfiler {
    double spmv_ms = 0.0;
    double dot_ms = 0.0;
    double axpy_ms = 0.0;
    double norm_ms = 0.0;
    double givens_ms = 0.0;
    double residual_ms = 0.0;
    double other_ms = 0.0;
    double device_transfer_ms = 0.0;
    double communication_ms = 0.0;
    double kernel_launch_ms = 0.0;
    double synchronization_ms = 0.0;

    void reset() {
        spmv_ms = 0.0;
        dot_ms = 0.0;
        axpy_ms = 0.0;
        norm_ms = 0.0;
        givens_ms = 0.0;
        residual_ms = 0.0;
        other_ms = 0.0;
        device_transfer_ms = communication_ms = kernel_launch_ms = synchronization_ms = 0.0;
    }

    double accounted_ms() const {
        return spmv_ms + dot_ms + axpy_ms + norm_ms + givens_ms + residual_ms + other_ms;
    }

    void accumulate(const GMRESProfiler& other) {
        spmv_ms += other.spmv_ms;
        dot_ms += other.dot_ms;
        axpy_ms += other.axpy_ms;
        norm_ms += other.norm_ms;
        givens_ms += other.givens_ms;
        residual_ms += other.residual_ms;
        other_ms += other.other_ms;
        device_transfer_ms += other.device_transfer_ms;
        communication_ms += other.communication_ms;
        kernel_launch_ms += other.kernel_launch_ms;
        synchronization_ms += other.synchronization_ms;
    }

    void scale(double factor) {
        spmv_ms *= factor;
        dot_ms *= factor;
        axpy_ms *= factor;
        norm_ms *= factor;
        givens_ms *= factor;
        residual_ms *= factor;
        other_ms *= factor;
        device_transfer_ms *= factor;
        communication_ms *= factor;
        kernel_launch_ms *= factor;
        synchronization_ms *= factor;
    }
};

void print_profiler_breakdown(std::ostream& out, const std::string& solver_name, const GMRESProfiler& profiler, double total_ms, int iterations);

}  // namespace ascend_gmres
