#include "blas_backend.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ascend_gmres {

CpuBlasBackend::CpuBlasBackend(bool parallel) : parallel_(parallel) {}

std::string CpuBlasBackend::name() const {
    return parallel_ ? "CPU OpenMP16 BLAS1" : "CPU single BLAS1";
}

float CpuBlasBackend::dot(const std::vector<float>& x, const std::vector<float>& y) {
    const std::size_t size = std::min(x.size(), y.size());
    double sum = 0.0;
    if (parallel_) {
#if defined(_OPENMP)
        omp_set_num_threads(16);
#pragma omp parallel for reduction(+ : sum) schedule(static)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(size); ++i) {
            sum += static_cast<double>(x[static_cast<std::size_t>(i)]) * static_cast<double>(y[static_cast<std::size_t>(i)]);
        }
    } else {
        for (std::size_t i = 0; i < size; ++i) {
            sum += static_cast<double>(x[i]) * static_cast<double>(y[i]);
        }
    }
    return static_cast<float>(sum);
}

float CpuBlasBackend::norm2(const std::vector<float>& x) {
    return std::sqrt(std::max(0.0f, dot(x, x)));
}

float HostPrototypeBlasBackend::dot(const std::vector<float>& x, const std::vector<float>& y) {
    return emulation_.dot(x, y);
}

float HostPrototypeBlasBackend::norm2(const std::vector<float>& x) {
    return emulation_.norm2(x);
}

std::string HostPrototypeBlasBackend::name() const {
    return "HostPrototype BLAS1";
}

}  // namespace ascend_gmres
