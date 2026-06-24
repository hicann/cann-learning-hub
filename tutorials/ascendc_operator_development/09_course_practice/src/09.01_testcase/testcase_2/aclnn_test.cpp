#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_log_sigmoid_custom.h"
#include "case_config.h"

#define SUCCESS 0
#define FAILED 1

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    return std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
}

template <typename T>
bool ReadBinFile(const std::string &fileName, std::vector<T> &data)
{
    std::ifstream input(fileName, std::ios::binary);
    if (!input) {
        LOG_PRINT("open %s failed\n", fileName.c_str());
        return false;
    }
    input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (input.gcount() != static_cast<std::streamsize>(data.size() * sizeof(T))) {
        LOG_PRINT("read %s failed, expected %zu bytes, got %lld bytes\n", fileName.c_str(), data.size() * sizeof(T),
                  static_cast<long long>(input.gcount()));
        return false;
    }
    return true;
}

template <typename T>
bool WriteBinFile(const std::string &fileName, const std::vector<T> &data)
{
    std::ofstream output(fileName, std::ios::binary);
    if (!output) {
        LOG_PRINT("open %s failed\n", fileName.c_str());
        return false;
    }
    output.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    return output.good();
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);
    return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return FAILED);
    return SUCCESS;
}

void DestroyResources(const std::vector<void *> &tensors, const std::vector<void *> &deviceAddrs, aclrtStream stream,
                      int32_t deviceId, void *workspaceAddr = nullptr)
{
    for (size_t i = 0; i < std::min(tensors.size(), deviceAddrs.size()); i++) {
        if (tensors[i] != nullptr) {
            aclDestroyTensor(reinterpret_cast<aclTensor *>(tensors[i]));
        }
        if (deviceAddrs[i] != nullptr) {
            aclrtFree(deviceAddrs[i]);
        }
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main(int argc, char **argv)
{
    const std::string dataDir = argc > 1 ? argv[1] : ".";
    const std::string inputFile = dataDir + "/input.bin";
    const std::string outputFile = dataDir + "/output.bin";

    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    std::vector<int64_t> inputXShape = CASE_SHAPE;
    std::vector<int64_t> outputYShape = CASE_SHAPE;
    const int64_t elementCount = GetShapeSize(inputXShape);

    std::vector<HostDataType> inputXHostData(elementCount);
    std::vector<HostDataType> outputYHostData(elementCount, static_cast<HostDataType>(0));
    CHECK_RET(ReadBinFile(inputFile, inputXHostData), DestroyResources({}, {}, stream, deviceId); return FAILED);

    void *inputXDeviceAddr = nullptr;
    void *outputYDeviceAddr = nullptr;
    aclTensor *inputX = nullptr;
    aclTensor *outputY = nullptr;

    ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, CASE_ACL_DTYPE, &inputX);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources({inputX, outputY}, {inputXDeviceAddr, outputYDeviceAddr}, stream,
                                                   deviceId);
              return FAILED);
    ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, CASE_ACL_DTYPE, &outputY);
    CHECK_RET(ret == ACL_SUCCESS, DestroyResources({inputX, outputY}, {inputXDeviceAddr, outputYDeviceAddr}, stream,
                                                   deviceId);
              return FAILED);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnLogSigmoidCustomGetWorkspaceSize(inputX, outputY, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSigmoidCustomGetWorkspaceSize failed. ERROR: %d\n", ret);
              DestroyResources({inputX, outputY}, {inputXDeviceAddr, outputYDeviceAddr}, stream, deviceId);
              return FAILED);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  DestroyResources({inputX, outputY}, {inputXDeviceAddr, outputYDeviceAddr}, stream, deviceId,
                                   workspaceAddr);
                  return FAILED);
    }

    ret = aclnnLogSigmoidCustom(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSigmoidCustom failed. ERROR: %d\n", ret);
              DestroyResources({inputX, outputY}, {inputXDeviceAddr, outputYDeviceAddr}, stream, deviceId,
                               workspaceAddr);
              return FAILED);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              DestroyResources({inputX, outputY}, {inputXDeviceAddr, outputYDeviceAddr}, stream, deviceId,
                               workspaceAddr);
              return FAILED);

    std::vector<HostDataType> resultData(elementCount);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputYDeviceAddr,
                      resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              DestroyResources({inputX, outputY}, {inputXDeviceAddr, outputYDeviceAddr}, stream, deviceId,
                               workspaceAddr);
              return FAILED);

    CHECK_RET(WriteBinFile(outputFile, resultData), DestroyResources({inputX, outputY},
                                                                     {inputXDeviceAddr, outputYDeviceAddr}, stream,
                                                                     deviceId, workspaceAddr);
              return FAILED);

    DestroyResources({inputX, outputY}, {inputXDeviceAddr, outputYDeviceAddr}, stream, deviceId, workspaceAddr);
    LOG_PRINT("write output to %s success\n", outputFile.c_str());
    return SUCCESS;
}
