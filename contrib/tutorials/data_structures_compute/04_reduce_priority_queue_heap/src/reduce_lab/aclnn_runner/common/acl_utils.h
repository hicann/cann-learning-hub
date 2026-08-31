#pragma once
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <aclnn/aclnn_base.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_ACL(call)                                                                      do {                                                                                         aclError _e = (call);                                                                    if (_e != ACL_SUCCESS) {                                                                     const char *_msg = aclGetRecentErrMsg();                                                 throw std::runtime_error(std::string("ACL error ") + std::to_string(_e) +                   " at " + __FILE__ + ":" + std::to_string(__LINE__) +                                   (_msg ? std::string(" recent_msg=") + _msg : std::string("")));                }                                                                                    } while (0)

inline void *MallocDevice(size_t bytes) {
    void *ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

inline aclTensor *CreateTensor(const std::vector<int64_t> &shape, aclDataType dtype,
                               aclFormat format, void *devicePtr) {
    aclTensor *tensor = aclCreateTensor(shape.data(), shape.size(), dtype,
                                        nullptr, 0, format,
                                        shape.data(), shape.size(), devicePtr);
    if (!tensor) {
        throw std::runtime_error("aclCreateTensor failed");
    }
    return tensor;
}

class AclRuntimeGuard {
public:
    explicit AclRuntimeGuard(int deviceId) : deviceId_(deviceId) {
        CHECK_ACL(aclInit(nullptr));
        CHECK_ACL(aclrtSetDevice(deviceId_));
        CHECK_ACL(aclrtCreateStream(&stream_));
    }
    ~AclRuntimeGuard() {
        if (stream_) { aclrtDestroyStream(stream_); }
        aclrtResetDevice(deviceId_);
        aclFinalize();
    }
    aclrtStream stream() const { return stream_; }
private:
    int deviceId_ = 0;
    aclrtStream stream_ = nullptr;
};
