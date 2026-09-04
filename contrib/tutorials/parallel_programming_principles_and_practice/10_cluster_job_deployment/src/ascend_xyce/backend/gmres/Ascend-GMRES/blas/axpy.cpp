#include "blas_backend.hpp"

#include <algorithm>
#include <cstdint>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ascend_gmres {

void CpuBlasBackend::axpy(float alpha, const std::vector<float>& x, std::vector<float>* y) {
    const std::size_t size = std::min(x.size(), y == nullptr ? std::size_t{0} : y->size());
    if (parallel_) {
#if defined(_OPENMP)
        omp_set_num_threads(16);
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(size); ++i) {
            (*y)[static_cast<std::size_t>(i)] += alpha * x[static_cast<std::size_t>(i)];
        }
    } else {
        for (std::size_t i = 0; i < size; ++i) {
            (*y)[i] += alpha * x[i];
        }
    }
}

void CpuBlasBackend::scal(float alpha, std::vector<float>* x) {
    if (x == nullptr) {
        return;
    }
    if (parallel_) {
#if defined(_OPENMP)
        omp_set_num_threads(16);
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(x->size()); ++i) {
            (*x)[static_cast<std::size_t>(i)] *= alpha;
        }
    } else {
        for (auto& value : *x) {
            value *= alpha;
        }
    }
}

void CpuBlasBackend::copy(const std::vector<float>& x, std::vector<float>* y) {
    if (y != nullptr) {
        *y = x;
    }
}

void HostPrototypeBlasBackend::axpy(float alpha, const std::vector<float>& x, std::vector<float>* y) {
    emulation_.axpy(alpha, x, y);
}

void HostPrototypeBlasBackend::scal(float alpha, std::vector<float>* x) {
    emulation_.scal(alpha, x);
}

void HostPrototypeBlasBackend::copy(const std::vector<float>& x, std::vector<float>* y) {
    emulation_.copy(x, y);
}

}  // namespace ascend_gmres
