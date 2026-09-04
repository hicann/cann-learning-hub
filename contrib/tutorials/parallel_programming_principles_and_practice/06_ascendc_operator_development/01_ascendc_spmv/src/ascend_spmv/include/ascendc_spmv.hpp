#pragma once

#include "spmv.hpp"

#include <string>

namespace spmv {

class AscendCSpmvBackend final : public ISpmvBackend {
public:
    explicit AscendCSpmvBackend(int device = 0);
    ~AscendCSpmvBackend() override;
    std::string name() const override;
    bool prepare(const CSRMatrix& matrix, std::string* error = nullptr) override;
    bool run(const std::vector<float>& x, std::vector<float>* y,
             BackendTimings* timings, std::string* error = nullptr) override;
    double initialization_ms() const override { return initialization_ms_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;
    int device_ = 0;
    double initialization_ms_ = 0.0;
};

}  // namespace spmv
