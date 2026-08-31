#include "common/acl_utils.h"
#include "common/bin_utils.h"

#include <aclnn_tree_queue_pipeline_lite.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static constexpr size_t FP16_BYTES = 2;
static constexpr int WARMUP_ITERS = 3;
static constexpr int TIMED_ITERS = 10;

static inline float fp16_to_float(uint16_t h) {
    int sign = (h >> 15) & 1;
    int exp = (h >> 10) & 0x1F;
    int mant = h & 0x3FF;
    if (exp == 0) {
        return (sign ? -1.0f : 1.0f) * std::ldexp(static_cast<float>(mant), -24);
    }
    if (exp == 31) {
        return mant == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
    }
    return (sign ? -1.0f : 1.0f) * std::ldexp(1.0f + mant / 1024.0f, exp - 15);
}

template <typename Function>
static double RunOpTimed(aclrtStream stream, Function &&function) {
    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    CHECK_ACL(aclrtCreateEvent(&startEvent));
    CHECK_ACL(aclrtCreateEvent(&endEvent));
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        function();
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtRecordEvent(startEvent, stream));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        function();
    }
    CHECK_ACL(aclrtRecordEvent(endEvent, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    float elapsedMs = 0.0f;
    CHECK_ACL(aclrtEventElapsedTime(&elapsedMs, startEvent, endEvent));
    aclrtDestroyEvent(startEvent);
    aclrtDestroyEvent(endEvent);
    return static_cast<double>(elapsedMs) / TIMED_ITERS;
}

struct DeviceResult {
    double elapsedMs = 0.0;
    std::vector<uint16_t> stageEnd;
    int32_t dependencyOk = 0;
};

static DeviceResult RunTreeQueue(void *parentDev, void *costDev, void *orderDev,
                                 void *stageEndDev, void *dependencyDev,
                                 int64_t taskCount, aclrtStream stream) {
    aclTensor *parent = CreateTensor({taskCount}, ACL_INT32, ACL_FORMAT_ND, parentDev);
    aclTensor *cost = CreateTensor({taskCount}, ACL_FLOAT16, ACL_FORMAT_ND, costDev);
    aclTensor *order = CreateTensor({taskCount}, ACL_INT32, ACL_FORMAT_ND, orderDev);
    aclTensor *stageEnd = CreateTensor({taskCount}, ACL_FLOAT16, ACL_FORMAT_ND, stageEndDev);
    aclTensor *dependencyOk = CreateTensor({1}, ACL_INT32, ACL_FORMAT_ND, dependencyDev);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACL(aclnnTreeQueuePipelineLiteGetWorkspaceSize(
        parent, cost, order, stageEnd, dependencyOk, &workspaceSize, &executor));
    void *workspace = workspaceSize > 0 ? MallocDevice(workspaceSize) : nullptr;

    DeviceResult result;
    result.elapsedMs = RunOpTimed(stream, [&]() {
        CHECK_ACL(aclnnTreeQueuePipelineLite(workspace, workspaceSize, executor, stream));
    });
    CHECK_ACL(aclrtSynchronizeStream(stream));

    result.stageEnd.resize(static_cast<size_t>(taskCount));
    CHECK_ACL(aclrtMemcpy(result.stageEnd.data(), taskCount * FP16_BYTES,
                          stageEndDev, taskCount * FP16_BYTES,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(&result.dependencyOk, sizeof(result.dependencyOk),
                          dependencyDev, sizeof(result.dependencyOk),
                          ACL_MEMCPY_DEVICE_TO_HOST));

    aclDestroyTensor(parent);
    aclDestroyTensor(cost);
    aclDestroyTensor(order);
    aclDestroyTensor(stageEnd);
    aclDestroyTensor(dependencyOk);
    if (workspace) {
        aclrtFree(workspace);
    }
    return result;
}

int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: main_tree_queue_benchmark <data_dir> [fifo|priority]\n";
            return 1;
        }
        const std::string dataDir = argv[1];
        const std::string mode = argc >= 3 ? argv[2] : "priority";
        if (mode != "fifo" && mode != "priority") {
            std::cerr << "mode must be fifo or priority\n";
            return 1;
        }

        const std::string inputDir = dataDir + "/input";
        auto parentHost = ReadBinary(inputDir + "/parent.bin");
        auto costHost = ReadBinary(inputDir + "/cost.bin");
        auto orderHost = ReadBinary(inputDir + "/order_" + mode + ".bin");
        auto refStageHost = ReadBinary(inputDir + "/ref_stage_end_" + mode + ".bin");
        const int64_t taskCount = static_cast<int64_t>(parentHost.size() / sizeof(int32_t));
        if (taskCount <= 0 || costHost.size() != static_cast<size_t>(taskCount) * FP16_BYTES
            || orderHost.size() != parentHost.size()) {
            throw std::runtime_error("input binary sizes do not match");
        }

        AclRuntimeGuard guard(0);
        aclrtStream stream = guard.stream();
        void *parentDev = MallocDevice(parentHost.size());
        void *costDev = MallocDevice(costHost.size());
        void *orderDev = MallocDevice(orderHost.size());
        void *stageEndDev = MallocDevice(costHost.size());
        void *dependencyDev = MallocDevice(sizeof(int32_t));
        CHECK_ACL(aclrtMemcpy(parentDev, parentHost.size(), parentHost.data(), parentHost.size(), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(costDev, costHost.size(), costHost.data(), costHost.size(), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(orderDev, orderHost.size(), orderHost.data(), orderHost.size(), ACL_MEMCPY_HOST_TO_DEVICE));

        std::cout << "=== Tree Queue Pipeline 910B Benchmark ===\n";
        std::cout << "tasks=" << taskCount << " order=" << mode
                  << " queue_depth=2 compute_lanes=2 block_dim=1\n";
        DeviceResult result = RunTreeQueue(parentDev, costDev, orderDev, stageEndDev,
                                           dependencyDev, taskCount, stream);

        bool stagePass = refStageHost.size() == costHost.size();
        float maxError = 0.0f;
        float deviceEnd = 0.0f;
        float referenceEnd = 0.0f;
        for (int64_t i = 0; i < taskCount && stagePass; ++i) {
            uint16_t refValue = 0;
            std::memcpy(&refValue, refStageHost.data() + i * FP16_BYTES, FP16_BYTES);
            float got = fp16_to_float(result.stageEnd[static_cast<size_t>(i)]);
            float expected = fp16_to_float(refValue);
            maxError = std::max(maxError, std::fabs(got - expected));
            deviceEnd = std::max(deviceEnd, got);
            referenceEnd = std::max(referenceEnd, expected);
        }
        stagePass = stagePass && maxError <= 0.02f;
        bool dependencyPass = result.dependencyOk == 1;
        std::cout << std::fixed << std::setprecision(4)
                  << "[pipeline] time=" << result.elapsedMs << " ms"
                  << " end=" << deviceEnd << " ref=" << referenceEnd
                  << " max_error=" << maxError
                  << (stagePass ? " PASS" : " FAIL") << "\n";
        std::cout << "[dependency] value=" << result.dependencyOk
                  << (dependencyPass ? " PASS" : " FAIL") << "\n";

        aclrtFree(parentDev);
        aclrtFree(costDev);
        aclrtFree(orderDev);
        aclrtFree(stageEndDev);
        aclrtFree(dependencyDev);
        return stagePass && dependencyPass ? 0 : 2;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        return 99;
    }
}
