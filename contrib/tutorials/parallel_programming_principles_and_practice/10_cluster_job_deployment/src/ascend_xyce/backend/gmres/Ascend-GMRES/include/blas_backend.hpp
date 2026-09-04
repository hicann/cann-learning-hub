#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ascend_gmres {

class IBlasBackend {
public:
    virtual ~IBlasBackend() = default;
    virtual std::string name() const = 0;
    virtual float dot(const std::vector<float>& x, const std::vector<float>& y) = 0;
    virtual void axpy(float alpha, const std::vector<float>& x, std::vector<float>* y) = 0;
    virtual void scal(float alpha, std::vector<float>* x) = 0;
    virtual float norm2(const std::vector<float>& x) = 0;
    virtual void copy(const std::vector<float>& x, std::vector<float>* y) = 0;
};

class CpuBlasBackend final : public IBlasBackend {
public:
    explicit CpuBlasBackend(bool parallel = false);
    std::string name() const override;
    float dot(const std::vector<float>& x, const std::vector<float>& y) override;
    void axpy(float alpha, const std::vector<float>& x, std::vector<float>* y) override;
    void scal(float alpha, std::vector<float>* x) override;
    float norm2(const std::vector<float>& x) override;
    void copy(const std::vector<float>& x, std::vector<float>* y) override;

private:
    bool parallel_ = false;
};

class HostPrototypeBlasBackend final : public IBlasBackend {
public:
    std::string name() const override;
    float dot(const std::vector<float>& x, const std::vector<float>& y) override;
    void axpy(float alpha, const std::vector<float>& x, std::vector<float>* y) override;
    void scal(float alpha, std::vector<float>* x) override;
    float norm2(const std::vector<float>& x) override;
    void copy(const std::vector<float>& x, std::vector<float>* y) override;

private:
    CpuBlasBackend emulation_{true};
};

}  // namespace ascend_gmres
