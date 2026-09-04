#pragma once

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace acl_demo {

inline void check_acl(aclError code, const char* expression) {
    if (code != ACL_SUCCESS) {
        throw std::runtime_error(std::string(expression) + " failed, aclError=" + std::to_string(code));
    }
}

inline void check_aclnn(int code, const char* expression) {
    if (code != 0) {
        throw std::runtime_error(std::string(expression) + " failed, aclnnStatus=" + std::to_string(code));
    }
}

#define ACL_CHECK(expr) ::acl_demo::check_acl((expr), #expr)
#define ACLNN_CHECK(expr) ::acl_demo::check_aclnn((expr), #expr)

inline std::size_t numel(const std::vector<int64_t>& shape) {
    std::size_t result = 1;
    for (int64_t dimension : shape) result *= static_cast<std::size_t>(dimension);
    return result;
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes) { allocate(bytes); }
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void allocate(std::size_t bytes) {
        reset();
        bytes_ = bytes;
        if (bytes_ != 0) ACL_CHECK(aclrtMalloc(&data_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    void reset() noexcept { if (data_ != nullptr) { aclrtFree(data_); data_ = nullptr; } bytes_ = 0; }
    void* data() const { return data_; }
    std::size_t bytes() const { return bytes_; }
private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

class Tensor {
public:
    Tensor() = default;
    Tensor(const std::vector<int64_t>& shape, aclDataType dtype, void* data) { create(shape, dtype, data); }
    ~Tensor() { reset(); }
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    void create(const std::vector<int64_t>& shape, aclDataType dtype, void* data) {
        reset();
        std::vector<int64_t> strides(shape.size(), 1);
        for (std::size_t i = shape.size(); i-- > 1;) strides[i - 1] = strides[i] * shape[i];
        tensor_ = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                  shape.data(), shape.size(), data);
        if (tensor_ == nullptr) throw std::runtime_error("aclCreateTensor failed");
    }
    void reset() noexcept { if (tensor_ != nullptr) { aclDestroyTensor(tensor_); tensor_ = nullptr; } }
    aclTensor* get() const { return tensor_; }
private:
    aclTensor* tensor_ = nullptr;
};

class Runtime {
public:
    explicit Runtime(int device) {
        ACL_CHECK(aclInit(nullptr));
        initialized_ = true;
        ACL_CHECK(aclrtSetDevice(device));
        ACL_CHECK(aclrtCreateContext(&context_, device));
        ACL_CHECK(aclrtSetCurrentContext(context_));
        ACL_CHECK(aclrtCreateStream(&stream_));
    }
    ~Runtime() {
        if (stream_) aclrtDestroyStream(stream_);
        if (context_) aclrtDestroyContext(context_);
        if (initialized_) aclFinalize();
    }
    aclrtStream stream() const { return stream_; }
private:
    bool initialized_ = false;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
};

inline void copy_to_device(const DeviceBuffer& destination, const void* source, std::size_t bytes) {
    ACL_CHECK(aclrtMemcpy(destination.data(), destination.bytes(), source, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}
inline void copy_to_host(void* destination, std::size_t bytes, const DeviceBuffer& source) {
    ACL_CHECK(aclrtMemcpy(destination, bytes, source.data(), source.bytes(), ACL_MEMCPY_DEVICE_TO_HOST));
}
inline void* allocate_workspace(std::size_t bytes) {
    void* p = nullptr;
    if (bytes != 0) ACL_CHECK(aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return p;
}
inline void free_workspace(void* p) { if (p) aclrtFree(p); }

}  // namespace acl_demo
