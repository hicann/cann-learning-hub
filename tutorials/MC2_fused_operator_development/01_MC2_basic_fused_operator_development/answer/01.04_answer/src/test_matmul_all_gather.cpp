#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <getopt.h>
#include <cmath>
#include <cstdlib>
#include "hccl/hccl.h"
#include "acl/acl.h"
#include "hccl/hcom.h"
#include "hccl/hccl_comm.h"
#include "securec.h"
#include "tiling/hccl/hccl_tiling.h"
#include "tiling_data.h"

using namespace std;
using namespace MatmulAllGatherImpl;

static constexpr size_t BF16_SIZE = 2;

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

constexpr int64_t BLOCK_NUM = 64;
int streamWithTimeout = 10000;

int gRankId = 0;
int gRankSize = 2;
int64_t gM = 1024;
int64_t gK = 1024;
int64_t gN = 1024;
int64_t gWorkSpaceSize = 256 * 1024 * 1024;
int64_t gHcclBufferSize = 0;  // 动态计算
std::string gOutputDir = ".";

extern "C" void matmul_all_gather_demo(uint32_t blockDim, void* stream,
    uint8_t* a, uint8_t* b, uint8_t* y, uint8_t* workspaceGM, uint8_t* mc2Context, uint8_t* tilingGM);

extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);

void GenerateRandomBF16Data(uint16_t* data, int64_t count, int seed, bool allOne = false)
{
    srand(seed);
    for (int64_t i = 0; i < count; ++i) {
        float val;
        if (allOne) {
            val = 1.0f;
        } else {
            val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        uint32_t x;
        memcpy(&x, &val, sizeof(float));
        uint16_t bf16 = (x >> 16) & 0xFFFF;
        data[i] = bf16;
    }
}

int WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        printf("Write file failed. Buffer is nullptr.\n");
        return -1;
    }
    FILE* fp = fopen(filePath.c_str(), "wb");
    if (fp == nullptr) {
        printf("Open file failed. path = %s\n", filePath.c_str());
        return -1;
    }
    size_t written = fwrite(buffer, 1, size, fp);
    if (written != size) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

int CreateTilingDataAndContext(const char* hcomName, aclrtStream stream,
    void **deviceTilingAddr, void **deviceContextAddr)
{
    MatmulAllGatherTilingData *tilingData = new MatmulAllGatherTilingData();
    if (tilingData == nullptr) {
        LOG_PRINT("[ERROR] tilingData is nullptr\n");
        return -1;
    }
    *tilingData = {};

    if (gHcclBufferSize <= 0) {
        int64_t winDataSizeBytes = gM * gN * BF16_SIZE;
        int64_t winDataSizeKB = (winDataSizeBytes + 1023) / 1024;
        gHcclBufferSize = winDataSizeKB + 1024;  // Win 数据区 + 状态区(STATE_WIN_SIZE=1MB)
        LOG_PRINT("[INFO] Calculated hcclBufferSize = %ld KB (winData=%ld KB)\n", 
                  gHcclBufferSize, winDataSizeKB);
    }
    
    tilingData->tilingInfo.mDim = gM;
    tilingData->tilingInfo.kDim = gK;
    tilingData->tilingInfo.nDim = gN;
    tilingData->tilingInfo.aivNum = BLOCK_NUM;
    tilingData->tilingInfo.totalWinSize = gHcclBufferSize;
    tilingData->tilingInfo.baseM = 128;
    tilingData->tilingInfo.baseK = 128;
    tilingData->tilingInfo.baseN = 128;

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(hcomName, gRankSize, "AlltoAll=level0:fullmesh;level1:pairwise");
    mc2CcTilingConfig.SetCommEngine(3);

    int ret = mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GetTiling mc2InitTiling failed. ret = %d\n", ret); return ret);
    ret = mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GetTiling mc2CcTiling failed. ret = %d\n", ret); return ret);

    int tilingSize = sizeof(MatmulAllGatherTilingData);
    ret = aclrtMalloc(deviceTilingAddr, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc TilingData failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceTilingAddr, tilingSize, tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy TilingData failed. ret: %d\n", ret); return ret);

    HcclComm commHandle;
    ret = HcomGetCommHandleByGroup(hcomName, &commHandle);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcomGetCommHandleByGroup failed. ret = %d\n", ret); return ret);

    void *mc2Context = nullptr;
    ret = HcclAllocComResourceByTiling(commHandle, stream, tilingData, &mc2Context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclAllocComResourceByTiling failed. ret = %d\n", ret); return ret);

    if (mc2Context == nullptr) {
        LOG_PRINT("[ERROR] mc2Context is nullptr\n");
        return -1;
    }

    constexpr size_t mc2ContextSize = sizeof(MatmulAllGatherImpl::HcclA5OpResParam);
    ret = aclrtMalloc(deviceContextAddr, mc2ContextSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc Mc2Context failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceContextAddr, mc2ContextSize, mc2Context, mc2ContextSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy Mc2Context failed. ret: %d\n", ret); return ret);

    delete tilingData;
    return ACL_SUCCESS;
}

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

int LaunchOneThread(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return ret);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s\n", args.rankId, hcomName);

    void *aDeviceAddr = nullptr;
    void *bDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    void *tilingAddr = nullptr;
    void *mc2ContextAddr = nullptr;

    ret = CreateTilingDataAndContext(hcomName, args.stream, &tilingAddr, &mc2ContextAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateTilingDataAndContext failed. ret = %d\n", ret); return ret);

    int64_t aSize = gM * gK * BF16_SIZE;
    int64_t bSize = gK * gN * BF16_SIZE;
    int64_t ySize = gM * gN * gRankSize * BF16_SIZE;  // AllGather 输出大小：M × N × rankSize

    ret = aclrtMalloc(&aDeviceAddr, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc a failed. ret: %d\n", ret); return ret);

uint16_t *aHost;
    aclrtMallocHost(reinterpret_cast<void**>(&aHost), aSize);
    GenerateRandomBF16Data(aHost, gM * gK, 42, true);
    ret = aclrtMemcpy(aDeviceAddr, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy a failed. ret = %d\n", ret); return ret);

    ret = aclrtMalloc(&bDeviceAddr, bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc b failed. ret: %d\n", ret); return ret);

    uint16_t *bHost;
    aclrtMallocHost(reinterpret_cast<void**>(&bHost), bSize);
    GenerateRandomBF16Data(bHost, gK * gN, 43, true);
    ret = aclrtMemcpy(bDeviceAddr, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy b failed. ret: %d\n", ret); return ret);

    ret = aclrtMalloc(&yDeviceAddr, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc y failed. ret: %d\n", ret); return ret);

    if (gWorkSpaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, gWorkSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret: %d\n", ret); return ret);
    }

    LOG_PRINT("[INFO] rank=%d Launching kernel: M=%ld, K=%ld, N=%ld, rankDim=%d, blockDim=%ld\n",
        args.rankId, gM, gK, gN, gRankSize, BLOCK_NUM);

    matmul_all_gather_demo(BLOCK_NUM, args.stream,
        (uint8_t*)aDeviceAddr, (uint8_t*)bDeviceAddr, (uint8_t*)yDeviceAddr,
        (uint8_t*)workspaceAddr, (uint8_t*)mc2ContextAddr, (uint8_t*)tilingAddr);

    ret = aclrtSynchronizeStreamWithTimeout(args.stream, streamWithTimeout);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret);
    LOG_PRINT("[INFO] rank=%d kernel executed successfully\n", args.rankId);

    uint16_t *yHost;
    aclrtMallocHost(reinterpret_cast<void**>(&yHost), ySize);
    ret = aclrtMemcpy(yHost, ySize, yDeviceAddr, ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy y failed. ret: %d\n", ret); return ret);

    std::string outFile = gOutputDir + "/output_rank" + std::to_string(args.rankId) + ".bin";
    WriteFile(outFile, yHost, ySize);
    LOG_PRINT("[INFO] rank=%d output written to %s\n", args.rankId, outFile.c_str());

    std::string aFile = gOutputDir + "/input_a_rank" + std::to_string(args.rankId) + ".bin";
    WriteFile(aFile, aHost, aSize);
    std::string bFile = gOutputDir + "/input_b_rank" + std::to_string(args.rankId) + ".bin";
    WriteFile(bFile, bHost, bSize);
    LOG_PRINT("[INFO] rank=%d input written\n", args.rankId);

    aclrtFreeHost(aHost);
    aclrtFreeHost(bHost);
    aclrtFreeHost(yHost);
    aclrtFree(aDeviceAddr);
    aclrtFree(bDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclrtFree(mc2ContextAddr);
    aclrtFree(tilingAddr);

    return ACL_SUCCESS;
}

int main(int argc, char *argv[])
{
    static const struct option longOptions[] = {
        {"rank_id",     1, 0, 'a'},
        {"rank_size",   1, 0, 's'},
        {"m",           1, 0, 'm'},
        {"k",           1, 0, 'k'},
        {"n",           1, 0, 'n'},
        {"output_dir",  1, 0, 'o'},
        {0, 0, 0, 0}
    };

    if (gHcclBufferSize <= 0) {
        int64_t winDataSizeBytes = gM * gN * BF16_SIZE;
        int64_t winDataSizeKB = (winDataSizeBytes + 1023) / 1024;
        gHcclBufferSize = winDataSizeKB + 1024;  // Win 数据区 + 状态区(STATE_WIN_SIZE=1MB)
        LOG_PRINT("[INFO] Calculated hcclBufferSize = %ld KB (winData=%ld KB)\n", 
                  gHcclBufferSize, winDataSizeKB);
    }

    int opt;
    while ((opt = getopt_long(argc, argv, "a:s:m:k:n:o:", longOptions, nullptr)) != -1) {
        switch (opt) {
            case 'a': gRankId = atoi(optarg); break;
            case 's': gRankSize = atoi(optarg); break;
            case 'm': gM = atoll(optarg); break;
            case 'k': gK = atoll(optarg); break;
            case 'n': gN = atoll(optarg); break;
            case 'o': gOutputDir = optarg; break;
        }
    }

    LOG_PRINT("[INFO] Config: rankId=%d, rankSize=%d, M=%ld, K=%ld, N=%ld\n", gRankId, gRankSize, gM, gK, gN);

    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

    aclrtStream stream;
    aclrtContext context;
    HcclComm comms;

    ret = aclrtSetDevice(gRankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);

    ret = aclrtCreateContext(&context, gRankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);

    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);

    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;
    config.hcclBufferSize = gHcclBufferSize;
    ret = strcpy_s(config.hcclCommName, COMM_NAME_MAX_LENGTH - 1, "hccl_comm_matmul_all_gather");
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] hcclCommName strcpy failed.\n"); return ret);

    const char* rankTableFile = getenv("RANK_TABLE_FILE");
    CHECK_RET(rankTableFile != nullptr, LOG_PRINT("[ERROR] RANK_TABLE_FILE not set.\n"); return -1);

    ret = HcclCommInitClusterInfoConfig(rankTableFile, gRankId, &config, &comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitClusterInfoConfig failed. ret = %d\n", ret); return ret);

    Args args;
    args.rankId = gRankId;
    args.hcclComm = comms;
    args.stream = stream;
    args.context = context;

    ret = LaunchOneThread(args);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] LaunchOneThread failed. ret = %d\n", ret); return ret);

    HcclCommDestroy(comms);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(gRankId);
    aclFinalize();

    LOG_PRINT("[INFO] rank=%d completed successfully\n", gRankId);
    return 0;
}