/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_and_combine_final.cpp
 * \brief
 */
#include <iostream>
#include <cstdlib>
#include <memory>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_operator.h"

#include "shmem.h"

#include "utils.h"
#include "../include/moe_distribute_dispatch.h"
#include "../include/moe_distribute_combine.h"

static constexpr uint64_t SHMEM_SPACE_SIZE = 1024UL * 1024UL * 1024UL;
static constexpr uint64_t DISPATCH_PACKAGE_SIZE = 512UL;
static constexpr uint64_t AIV_CORE_NUM = 64UL;

__global__ __aicore__ __vector__ void DispatchKernel(
    __gm__ void* shmemSpace, GM_ADDR x, GM_ADDR expertIds,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
    GM_ADDR workspaceGM, DispatchTilingData tilingData)
{
    AscendC::TPipe pipe;
    DispatchImpl::MoeDistributeDispatch dispatchOp;
    dispatchOp.Init(shmemSpace, x, expertIds,
        expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
        workspaceGM, &pipe, &tilingData);
    dispatchOp.Process();
    return;
}

__global__ __aicore__ __vector__ void CombineKernel(
    __gm__ void* shmemSpace, GM_ADDR expandX, GM_ADDR expertIds, 
    GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR expertScales, GM_ADDR XOut, 
    MoeDistributeCombineShmemTilingData combineTilingData)
{
    AscendC::TPipe pipe;
    MoeDistributeCombineShmemImpl::MoeDistributeCombineShmem<float16_t, float16_t, int32_t, true> op;
    op.Init((GM_ADDR)shmemSpace, expandX, expertIds, expandIdx, epSendCount, expertScales, XOut, &pipe, combineTilingData);
    op.Process();
    return;
}

void SetDispatchTilingData(DispatchTilingData& dispatchTilingData, int epWorldSize, int epRankId, int bs)
{
    dispatchTilingData.epWorldSize = epWorldSize;
    dispatchTilingData.epRankId = epRankId;
    dispatchTilingData.moeExpertNum = epWorldSize * 4;
    dispatchTilingData.bs = bs;
    dispatchTilingData.k = 8;
    dispatchTilingData.h = 7168;
    dispatchTilingData.expertTokenNumsType = 1;
    dispatchTilingData.aivNum = AIV_CORE_NUM;
    dispatchTilingData.symMemSize = SHMEM_SPACE_SIZE;
}

void SetCombineTilingData(MoeDistributeCombineShmemTilingData& combineTilingData, int epWorldSize, int epRankId, int bs)
{
    combineTilingData.epWorldSize = epWorldSize;
    combineTilingData.epRankId = epRankId;
    combineTilingData.moeExpertPerRankNum = 4;
    combineTilingData.moeExpertNum = epWorldSize * combineTilingData.moeExpertPerRankNum;
    combineTilingData.globalBs = bs * epWorldSize;
    combineTilingData.bs = bs;
    combineTilingData.k = 8;
    combineTilingData.h = 7168;
    combineTilingData.aivNum = AIV_CORE_NUM;
    combineTilingData.totalWinSize = SHMEM_SPACE_SIZE;
}

int runDispatchAndCombine(int rankNum, int rankId, int bs) {
    const char *ipport = "tcp://127.0.0.1:8998";
    INFO_LOG("rankNum=%d, rankId=%d, ipport=%s", rankNum, rankId, ipport);

    // Acl init
    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = rankId;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    // shmem init
    uint64_t localMemSize = SHMEM_SPACE_SIZE;
    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t defaultFlagUid;
    test_set_attr(rankId, rankNum, localMemSize, ipport, defaultFlagUid, &attributes);
    ACL_CHECK_WITH_RET(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes),
        ERROR_LOG("aclshmemx_init_attr failed"), return -1);

    int32_t aclshmemSize = SHMEM_SPACE_SIZE;
    void *shmemSpace = aclshmem_align(DISPATCH_PACKAGE_SIZE, aclshmemSize);

    // init dispatch tiling and io
    DispatchTilingData dispatchTilingData;
    SetDispatchTilingData(dispatchTilingData, rankNum, rankId, bs);
    size_t localExpertNum = dispatchTilingData.moeExpertNum
        / dispatchTilingData.epWorldSize;
    size_t maxReceivedTokenNum = dispatchTilingData.bs * dispatchTilingData.epWorldSize
        * std::min<size_t>(dispatchTilingData.k, localExpertNum);

    uint8_t *xInHost;
    uint8_t *xInDevice;
    size_t xInSize = dispatchTilingData.bs * dispatchTilingData.h * sizeof(float16_t);
    InitData(&xInHost, &xInDevice, xInSize, GetInputFilePath("x", rankId));

    uint8_t *expertIdsHost;
    uint8_t *expertIdsDevice;
    size_t expertIdsSize = dispatchTilingData.bs * dispatchTilingData.k * sizeof(int32_t);
    InitData(&expertIdsHost, &expertIdsDevice, expertIdsSize, GetInputFilePath("expert_ids", rankId));

    uint8_t *expandXOutHost;
    uint8_t *expandXOutDevice;
    size_t expandXOutSize = maxReceivedTokenNum * dispatchTilingData.h * sizeof(fp8_e5m2_t);
    InitData(&expandXOutHost, &expandXOutDevice, expandXOutSize);

    uint8_t *dynamicScalesHost;
    uint8_t *dynamicScalesDevice;
    size_t dynamicScalesSize = maxReceivedTokenNum
        * ((dispatchTilingData.h + 32 - 1) / 32) * sizeof(fp8_e8m0_t);
    InitData(&dynamicScalesHost, &dynamicScalesDevice, dynamicScalesSize);

    uint8_t *tokenSrcInfoHost;
    uint8_t *tokenSrcInfoDevice;
    size_t tokenSrcInfoSize = maxReceivedTokenNum * 128 * sizeof(int32_t);
    InitData(&tokenSrcInfoHost, &tokenSrcInfoDevice, tokenSrcInfoSize);

    uint8_t *expertTokenNumsHost;
    uint8_t *expertTokenNumsDevice;
    size_t expertTokenNumsSize = localExpertNum * sizeof(int64_t);
    InitData(&expertTokenNumsHost, &expertTokenNumsDevice, expertTokenNumsSize);

    uint8_t *sendCountsHost;
    uint8_t *sendCountsDevice;
    size_t sendCountSize = localExpertNum * dispatchTilingData.epWorldSize * sizeof(int32_t);
    InitData(&sendCountsHost, &sendCountsDevice, sendCountSize);

    uint8_t *disaptchWorkspaceGM;
    size_t disaptchWorkspaceSize = 16 * 1024 * 1024;
    ACL_CHECK(
        aclrtMalloc(reinterpret_cast<void**>(&disaptchWorkspaceGM), disaptchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // init combine tiling and io
    MoeDistributeCombineShmemTilingData combineTilingData;
    SetCombineTilingData(combineTilingData, rankNum, rankId, bs);

    uint8_t *expandXInHost;
    uint8_t *expandXInDevice;
    size_t expandXInSize = maxReceivedTokenNum * combineTilingData.h * sizeof(float16_t);
    InitData(&expandXInHost, &expandXInDevice, expandXInSize, GetInputFilePath("expand_x", rankId));

    uint8_t *expertScalestHost;
    uint8_t *expertScalesDevice;
    size_t expertScalesSize = combineTilingData.bs * combineTilingData.k * sizeof(float);
    InitData(&expertScalestHost, &expertScalesDevice, expertScalesSize, GetInputFilePath("expert_scales", rankId));

    uint8_t *xOutHost;
    uint8_t *xOutDevice;
    size_t xOutSize = combineTilingData.bs * combineTilingData.h * sizeof(float16_t);
    InitData(&xOutHost, &xOutDevice, xOutSize);

    int loopTimes = 20;
    for (int i = 0; i < loopTimes; ++i) {
        DispatchKernel<<<AIV_CORE_NUM, nullptr, stream>>>(
            shmemSpace, xInDevice, expertIdsDevice,
            expandXOutDevice, dynamicScalesDevice, tokenSrcInfoDevice, expertTokenNumsDevice, sendCountsDevice,  
            disaptchWorkspaceGM, dispatchTilingData);
        CombineKernel<<<AIV_CORE_NUM, nullptr, stream>>>(
            shmemSpace, expandXInDevice, expertIdsDevice, tokenSrcInfoDevice, sendCountsDevice, expertScalesDevice,
            xOutDevice, combineTilingData);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // free shmem space
    aclshmem_free(shmemSpace);

    // free dispatch io
    FinalizeData(xInHost, xInDevice);
    FinalizeData(expertIdsHost, expertIdsDevice);

    FinalizeData(expandXOutHost, expandXOutDevice, expandXOutSize, GetOutputFilePath("expand_x", rankId));
    FinalizeData(
        dynamicScalesHost, dynamicScalesDevice, dynamicScalesSize, GetOutputFilePath("dynamic_scales", rankId));
    FinalizeData(
        tokenSrcInfoHost, tokenSrcInfoDevice, tokenSrcInfoSize, GetOutputFilePath("assist_info_for_combine", rankId));
    FinalizeData(
        expertTokenNumsHost, expertTokenNumsDevice, expertTokenNumsSize, GetOutputFilePath("expert_token_nums", rankId));
    FinalizeData(sendCountsHost, sendCountsDevice, sendCountSize, GetOutputFilePath("ep_recv_count", rankId));

    ACL_CHECK(aclrtFree(reinterpret_cast<void *>(disaptchWorkspaceGM)));

    // free combine io
    FinalizeData(expandXInHost, expandXInDevice);
    FinalizeData(expertScalestHost, expertScalesDevice);

    FinalizeData(xOutHost, xOutDevice, xOutSize, GetOutputFilePath("x", rankId));

    // release resource
    ACL_CHECK(aclshmem_finalize());
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    std::cout << "[SUCCESS] demo run success in relative_pe_id " << rankId << std::endl;
    return 0;
}

static int RunWorker(int rankNum, int rankId, int bs, uint64_t /*shmemSpaceSize*/)
{
    return runDispatchAndCombine(rankNum, rankId, bs);
}

int main(int argc, char* argv[])
{
    return MoeDemoForkMain(argc, argv, RunWorker, SHMEM_SPACE_SIZE);
}