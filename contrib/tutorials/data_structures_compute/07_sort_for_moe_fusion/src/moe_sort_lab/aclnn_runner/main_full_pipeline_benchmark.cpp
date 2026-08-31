#include "common/acl_utils.h"
#include "common/bin_utils.h"

#include <aclnn_moe_top_k_lite.h>
#include <aclnn_moe_sort_quick_sort_lite.h>
#include <aclnn_moe_sort_heap_sort_lite.h>
#include <aclnn_moe_token_permute_lite.h>
#include <aclnn_moe_token_unpermute_lite.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr size_t FP16_BYTES = 2;
static constexpr size_t I32_BYTES = 4;
static constexpr int64_t LITE_FIXED_TOP_K = 2;

struct PipelineTiming {
    double routeMs = 0.0;
    double buildOrderAndCopyMs = 0.0;
    double permuteMs = 0.0;
    double unpermuteMs = 0.0;
    double deviceOpTotalMs = 0.0;
    double endToEndMs = 0.0;
};

enum class RouteKind {
    TopK,
    QuickSort,
    HeapSort,
};

static const char *KindName(RouteKind kind)
{
    switch (kind) {
        case RouteKind::TopK: return "topk";
        case RouteKind::QuickSort: return "quicksort";
        case RouteKind::HeapSort: return "heapsort";
    }
    return "unknown";
}

static const char *KindTitle(RouteKind kind)
{
    switch (kind) {
        case RouteKind::TopK: return "TopK full flow";
        case RouteKind::QuickSort: return "QuickSort full flow";
        case RouteKind::HeapSort: return "HeapSort full flow";
    }
    return "unknown";
}

static double RunTopK(void *logitsDev, void *topkIdxDev, void *topkProbsDev,
                      int64_t T, int64_t E, int64_t K, aclrtStream stream)
{
    aclTensor *logitsT = CreateTensor({T, E}, ACL_FLOAT16, ACL_FORMAT_ND, logitsDev);
    aclTensor *topkIdxT = CreateTensor({T, K}, ACL_INT32, ACL_FORMAT_ND, topkIdxDev);
    aclTensor *topkProbsT = CreateTensor({T, K}, ACL_FLOAT16, ACL_FORMAT_ND, topkProbsDev);
    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    CHECK_ACL(aclnnMoeTopKLiteGetWorkspaceSize(logitsT, topkIdxT, topkProbsT, &ws, &ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;

    // Create events for device-side timing
    aclrtEvent startEvt, endEvt;
    CHECK_ACL(aclrtCreateEvent(&startEvt));
    CHECK_ACL(aclrtCreateEvent(&endEvt));

    // Warmup (3 iterations)
    for (int i = 0; i < 3; ++i) {
        CHECK_ACL(aclnnMoeTopKLite(workspace, ws, ex, stream));
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // Timed iterations (10x)
    constexpr int ITERS = 10;
    CHECK_ACL(aclrtRecordEvent(startEvt, stream));
    for (int i = 0; i < ITERS; ++i) {
        CHECK_ACL(aclnnMoeTopKLite(workspace, ws, ex, stream));
    }
    CHECK_ACL(aclrtRecordEvent(endEvt, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    float elapsedMs = 0.0f;
    CHECK_ACL(aclrtEventElapsedTime(&elapsedMs, startEvt, endEvt));

    aclrtDestroyEvent(startEvt);
    aclrtDestroyEvent(endEvt);
    aclDestroyTensor(logitsT);
    aclDestroyTensor(topkIdxT);
    aclDestroyTensor(topkProbsT);
    if (workspace) {
        aclrtFree(workspace);
    }
    return static_cast<double>(elapsedMs) / ITERS;
}

static double RunQuickSortRoute(void *logitsDev, void *idxDev, void *probsDev,
                                int64_t T, int64_t E, int64_t K, aclrtStream stream)
{
    aclTensor *logitsT = CreateTensor({T, E}, ACL_FLOAT16, ACL_FORMAT_ND, logitsDev);
    aclTensor *idxT = CreateTensor({T, K}, ACL_INT32, ACL_FORMAT_ND, idxDev);
    aclTensor *probsT = CreateTensor({T, K}, ACL_FLOAT16, ACL_FORMAT_ND, probsDev);
    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    CHECK_ACL(aclnnMoeSortQuickSortLiteGetWorkspaceSize(logitsT, idxT, probsT, &ws, &ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;
    CHECK_ACL(aclnnMoeSortQuickSortLite(workspace, ws, ex, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    aclDestroyTensor(logitsT);
    aclDestroyTensor(idxT);
    aclDestroyTensor(probsT);
    if (workspace) {
        aclrtFree(workspace);
    }
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double RunHeapSortRoute(void *logitsDev, void *idxDev, void *probsDev,
                               int64_t T, int64_t E, int64_t K, aclrtStream stream)
{
    aclTensor *logitsT = CreateTensor({T, E}, ACL_FLOAT16, ACL_FORMAT_ND, logitsDev);
    aclTensor *idxT = CreateTensor({T, K}, ACL_INT32, ACL_FORMAT_ND, idxDev);
    aclTensor *probsT = CreateTensor({T, K}, ACL_FLOAT16, ACL_FORMAT_ND, probsDev);
    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    CHECK_ACL(aclnnMoeSortHeapSortLiteGetWorkspaceSize(logitsT, idxT, probsT, &ws, &ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;
    CHECK_ACL(aclnnMoeSortHeapSortLite(workspace, ws, ex, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    aclDestroyTensor(logitsT);
    aclDestroyTensor(idxT);
    aclDestroyTensor(probsT);
    if (workspace) {
        aclrtFree(workspace);
    }
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double RunRoute(RouteKind kind, void *logitsDev, void *idxDev, void *probsDev,
                       int64_t T, int64_t E, int64_t K, aclrtStream stream)
{
    if (kind == RouteKind::TopK) {
        return RunTopK(logitsDev, idxDev, probsDev, T, E, K, stream);
    }
    if (kind == RouteKind::QuickSort) {
        return RunQuickSortRoute(logitsDev, idxDev, probsDev, T, E, K, stream);
    }
    return RunHeapSortRoute(logitsDev, idxDev, probsDev, T, E, K, stream);
}

static double RunPermute(void *tokensDev, void *sortedOrderDev, void *probsDev,
                         void *permTokensDev, void *sortedIdxDev, void *permProbsDev,
                         int64_t T, int64_t H, int64_t K, aclrtStream stream)
{
    const int64_t total = T * K;
    aclTensor *tokensT = CreateTensor({T, H}, ACL_FLOAT16, ACL_FORMAT_ND, tokensDev);
    aclTensor *orderT = CreateTensor({total}, ACL_INT32, ACL_FORMAT_ND, sortedOrderDev);
    aclTensor *probsT = CreateTensor({T, K}, ACL_FLOAT16, ACL_FORMAT_ND, probsDev);
    aclTensor *permTokensT = CreateTensor({total, H}, ACL_FLOAT16, ACL_FORMAT_ND, permTokensDev);
    aclTensor *sortedIdxT = CreateTensor({total}, ACL_INT32, ACL_FORMAT_ND, sortedIdxDev);
    aclTensor *permProbsT = CreateTensor({total}, ACL_FLOAT16, ACL_FORMAT_ND, permProbsDev);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    CHECK_ACL(aclnnMoeTokenPermuteLiteGetWorkspaceSize(tokensT, orderT, probsT,
                                                       permTokensT, sortedIdxT, permProbsT,
                                                       &ws, &ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;
    CHECK_ACL(aclnnMoeTokenPermuteLite(workspace, ws, ex, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto t1 = std::chrono::high_resolution_clock::now();

    aclDestroyTensor(tokensT);
    aclDestroyTensor(orderT);
    aclDestroyTensor(probsT);
    aclDestroyTensor(permTokensT);
    aclDestroyTensor(sortedIdxT);
    aclDestroyTensor(permProbsT);
    if (workspace) {
        aclrtFree(workspace);
    }
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double RunUnpermute(void *expertOutDev, void *sortedIdxDev, void *permProbsDev, void *outDev,
                           int64_t T, int64_t H, int64_t K, aclrtStream stream)
{
    const int64_t total = T * K;
    aclTensor *expertT = CreateTensor({total, H}, ACL_FLOAT16, ACL_FORMAT_ND, expertOutDev);
    aclTensor *idxT = CreateTensor({total}, ACL_INT32, ACL_FORMAT_ND, sortedIdxDev);
    aclTensor *probT = CreateTensor({total}, ACL_FLOAT16, ACL_FORMAT_ND, permProbsDev);
    aclTensor *outT = CreateTensor({T, H}, ACL_FLOAT16, ACL_FORMAT_ND, outDev);

    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    CHECK_ACL(aclnnMoeTokenUnpermuteLiteGetWorkspaceSize(expertT, idxT, probT, outT, &ws, &ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;
    CHECK_ACL(aclnnMoeTokenUnpermuteLite(workspace, ws, ex, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto t1 = std::chrono::high_resolution_clock::now();

    aclDestroyTensor(expertT);
    aclDestroyTensor(idxT);
    aclDestroyTensor(probT);
    aclDestroyTensor(outT);
    if (workspace) {
        aclrtFree(workspace);
    }
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static std::vector<int32_t> BuildSortedOrderFromIndices(const std::vector<int32_t> &indices,
                                                        int64_t T, int64_t K)
{
    struct Pair {
        int32_t expert;
        int32_t pairId;
    };

    std::vector<Pair> pairs;
    pairs.reserve(static_cast<size_t>(T * K));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < K; ++k) {
            const int64_t pairId = t * K + k;
            pairs.push_back(Pair{indices[static_cast<size_t>(pairId)], static_cast<int32_t>(pairId)});
        }
    }

    std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) {
        if (a.expert != b.expert) {
            return a.expert < b.expert;
        }
        return a.pairId < b.pairId;
    });

    std::vector<int32_t> sortedOrder(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) {
        sortedOrder[i] = pairs[i].pairId;
    }
    return sortedOrder;
}

static double BuildOrderAndUpload(void *idxDev, void *sortedOrderDev,
                                  int64_t T, int64_t K,
                                  std::vector<int32_t> &idxHost,
                                  std::vector<int32_t> &sortedOrderHost)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    idxHost.assign(static_cast<size_t>(T * K), 0);
    CHECK_ACL(aclrtMemcpy(idxHost.data(), static_cast<size_t>(T * K * I32_BYTES),
                          idxDev, static_cast<size_t>(T * K * I32_BYTES),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    sortedOrderHost = BuildSortedOrderFromIndices(idxHost, T, K);
    CHECK_ACL(aclrtMemcpy(sortedOrderDev, static_cast<size_t>(T * K * I32_BYTES),
                          sortedOrderHost.data(), static_cast<size_t>(T * K * I32_BYTES),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static void SavePipelineOutputs(const std::string &dir,
                                const std::vector<int32_t> &indicesHost,
                                const std::vector<int32_t> &sortedOrderHost,
                                void *probsDev,
                                void *permTokensDev,
                                void *sortedIdxDev,
                                void *permProbsDev,
                                void *unpermuteOutDev,
                                int64_t T, int64_t H, int64_t K)
{
    const int64_t total = T * K;
    std::filesystem::create_directories(dir);

    std::vector<uint8_t> probsHost(static_cast<size_t>(T * K * FP16_BYTES));
    std::vector<uint8_t> permTokensHost(static_cast<size_t>(total * H * FP16_BYTES));
    std::vector<int32_t> sortedIdxHost(static_cast<size_t>(total));
    std::vector<uint8_t> permProbsHost(static_cast<size_t>(total * FP16_BYTES));
    std::vector<uint8_t> unpermuteOutHost(static_cast<size_t>(T * H * FP16_BYTES));

    CHECK_ACL(aclrtMemcpy(probsHost.data(), probsHost.size(), probsDev, probsHost.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(permTokensHost.data(), permTokensHost.size(), permTokensDev, permTokensHost.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(sortedIdxHost.data(), static_cast<size_t>(total * I32_BYTES), sortedIdxDev, static_cast<size_t>(total * I32_BYTES), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(permProbsHost.data(), permProbsHost.size(), permProbsDev, permProbsHost.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(unpermuteOutHost.data(), unpermuteOutHost.size(), unpermuteOutDev, unpermuteOutHost.size(), ACL_MEMCPY_DEVICE_TO_HOST));

    WriteBinary(dir + "/indices.bin", indicesHost.data(), indicesHost.size() * I32_BYTES);
    WriteBinary(dir + "/probs.bin", probsHost.data(), probsHost.size());
    WriteBinary(dir + "/sorted_order.bin", sortedOrderHost.data(), sortedOrderHost.size() * I32_BYTES);
    WriteBinary(dir + "/permuted_tokens.bin", permTokensHost.data(), permTokensHost.size());
    WriteBinary(dir + "/sorted_indices.bin", sortedIdxHost.data(), sortedIdxHost.size() * I32_BYTES);
    WriteBinary(dir + "/permuted_probs.bin", permProbsHost.data(), permProbsHost.size());
    WriteBinary(dir + "/unpermute_out.bin", unpermuteOutHost.data(), unpermuteOutHost.size());
}

static PipelineTiming RunFullPipeline(RouteKind kind,
                                      const std::string &dataDir,
                                      void *logitsDev,
                                      void *tokensDev,
                                      int64_t T, int64_t E, int64_t H, int64_t K,
                                      aclrtStream stream)
{
    const int64_t total = T * K;
    void *idxDev = MallocDevice(static_cast<size_t>(T * K * I32_BYTES));
    void *probsDev = MallocDevice(static_cast<size_t>(T * K * FP16_BYTES));
    void *sortedOrderDev = MallocDevice(static_cast<size_t>(total * I32_BYTES));
    void *permTokensDev = MallocDevice(static_cast<size_t>(total * H * FP16_BYTES));
    void *sortedIdxDev = MallocDevice(static_cast<size_t>(total * I32_BYTES));
    void *permProbsDev = MallocDevice(static_cast<size_t>(total * FP16_BYTES));
    void *unpermuteOutDev = MallocDevice(static_cast<size_t>(T * H * FP16_BYTES));

    std::vector<int32_t> indicesHost;
    std::vector<int32_t> sortedOrderHost;
    PipelineTiming timing;

    auto all0 = std::chrono::high_resolution_clock::now();

    timing.routeMs = RunRoute(kind, logitsDev, idxDev, probsDev, T, E, K, stream);

    // This teaching package does not include a separate device-side BuildSortedOrder op.
    // Therefore sortedOrder is built on Host from the selected expert indices and copied to Device.
    // It is reported separately so you can distinguish pure Ascend C kernel time from end-to-end time.
    timing.buildOrderAndCopyMs = BuildOrderAndUpload(idxDev, sortedOrderDev, T, K, indicesHost, sortedOrderHost);

    timing.permuteMs = RunPermute(tokensDev, sortedOrderDev, probsDev,
                                  permTokensDev, sortedIdxDev, permProbsDev,
                                  T, H, K, stream);

    // Expert computation is mocked as identity: expertOut == permutedTokens.
    // This keeps the experiment focused on routing + token movement.
    timing.unpermuteMs = RunUnpermute(permTokensDev, sortedIdxDev, permProbsDev, unpermuteOutDev,
                                      T, H, K, stream);

    auto all1 = std::chrono::high_resolution_clock::now();
    timing.deviceOpTotalMs = timing.routeMs + timing.permuteMs + timing.unpermuteMs;
    timing.endToEndMs = std::chrono::duration<double, std::milli>(all1 - all0).count();

    SavePipelineOutputs(dataDir + "/output/full_pipeline/" + std::string(KindName(kind)),
                        indicesHost, sortedOrderHost,
                        probsDev, permTokensDev, sortedIdxDev, permProbsDev, unpermuteOutDev,
                        T, H, K);

    aclrtFree(idxDev);
    aclrtFree(probsDev);
    aclrtFree(sortedOrderDev);
    aclrtFree(permTokensDev);
    aclrtFree(sortedIdxDev);
    aclrtFree(permProbsDev);
    aclrtFree(unpermuteOutDev);

    return timing;
}

static void PrintTimingRow(const char *name, const PipelineTiming &t)
{
    std::cout << std::left << std::setw(18) << name
              << std::right << std::setw(12) << t.routeMs
              << std::setw(14) << t.buildOrderAndCopyMs
              << std::setw(12) << t.permuteMs
              << std::setw(12) << t.unpermuteMs
              << std::setw(14) << t.deviceOpTotalMs
              << std::setw(14) << t.endToEndMs
              << "\n";
}

int main(int argc, char **argv)
{
    try {
        if (argc != 5) {
            std::cerr << "usage: main_full_pipeline_benchmark <data_dir> <num_tokens> <hidden_size> <top_k>\n";
            return 1;
        }
        const std::string dataDir = argv[1];
        const int64_t T = std::stoll(argv[2]);
        const int64_t H = std::stoll(argv[3]);
        const int64_t K = std::stoll(argv[4]);

        if (K != LITE_FIXED_TOP_K) {
            throw std::runtime_error("Lite teaching package currently supports top_k=2 only. Regenerate data with --top_k 2.");
        }

        std::filesystem::create_directories(dataDir + "/output/full_pipeline");

        auto logitsHost = ReadBinary(dataDir + "/input/logits.bin");
        auto tokensHost = ReadBinary(dataDir + "/input/tokens.bin");
        const int64_t E = static_cast<int64_t>(logitsHost.size()) / (T * FP16_BYTES);

        AclRuntimeGuard guard(0);
        aclrtStream stream = guard.stream();

        void *logitsDev = MallocDevice(logitsHost.size());
        void *tokensDev = MallocDevice(tokensHost.size());
        CHECK_ACL(aclrtMemcpy(logitsDev, logitsHost.size(), logitsHost.data(), logitsHost.size(), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(tokensDev, tokensHost.size(), tokensHost.data(), tokensHost.size(), ACL_MEMCPY_HOST_TO_DEVICE));

        std::cout << "=== MoE Lite Full-Flow Benchmark ===\n";
        std::cout << "T=" << T << " H=" << H << " K=" << K << " E=" << E << " total=" << (T * K) << "\n";
        std::cout << "Each row is one complete path: route selection -> build sortedOrder -> permute -> identity expert -> unpermute.\n";
        std::cout << "buildOrder+copy is Host D2H + CPU sortedOrder build + H2D because this package has no separate device-side BuildSortedOrder op.\n\n";

        PipelineTiming topk = RunFullPipeline(RouteKind::TopK, dataDir, logitsDev, tokensDev, T, E, H, K, stream);
        PipelineTiming quick = RunFullPipeline(RouteKind::QuickSort, dataDir, logitsDev, tokensDev, T, E, H, K, stream);
        PipelineTiming heap = RunFullPipeline(RouteKind::HeapSort, dataDir, logitsDev, tokensDev, T, E, H, K, stream);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << std::left << std::setw(18) << "flow"
                  << std::right << std::setw(12) << "route"
                  << std::setw(14) << "order+copy"
                  << std::setw(12) << "permute"
                  << std::setw(12) << "unpermute"
                  << std::setw(14) << "device_total"
                  << std::setw(14) << "end_to_end"
                  << "\n";
        PrintTimingRow(KindTitle(RouteKind::TopK), topk);
        PrintTimingRow(KindTitle(RouteKind::QuickSort), quick);
        PrintTimingRow(KindTitle(RouteKind::HeapSort), heap);

        std::cout << "\nOutput files are saved under: " << dataDir << "/output/full_pipeline/{topk,quicksort,heapsort}\n";
        std::cout << "Use scripts/verify_full_pipeline.py to compare the three complete flows with golden results.\n";

        aclrtFree(logitsDev);
        aclrtFree(tokensDev);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 99;
    }
}
