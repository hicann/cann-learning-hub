#include "common/acl_utils.h"
#include "common/bin_utils.h"

#include <aclnn_moe_top_k_lite.h>
#include <aclnn_moe_sort_quick_sort_lite.h>
#include <aclnn_moe_sort_heap_sort_lite.h>
#include <aclnn_moe_token_permute_lite.h>
#include <aclnn_moe_token_unpermute_lite.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr size_t FP16_BYTES = 2;
static constexpr size_t I32_BYTES = 4;
static constexpr int64_t LITE_FIXED_TOP_K = 2;

static double RunTopK(void *logitsDev, void *topkIdxDev, void *topkProbsDev,
                      int64_t T, int64_t E, int64_t K, aclrtStream stream)
{
    aclTensor *logitsT = CreateTensor({T, E}, ACL_FLOAT16, ACL_FORMAT_ND, logitsDev);
    aclTensor *topkIdxT = CreateTensor({T, K}, ACL_INT32, ACL_FORMAT_ND, topkIdxDev);
    aclTensor *topkProbsT = CreateTensor({T, K}, ACL_FLOAT16, ACL_FORMAT_ND, topkProbsDev);
    uint64_t ws = 0;
    aclOpExecutor *ex = nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    CHECK_ACL(aclnnMoeTopKLiteGetWorkspaceSize(logitsT, topkIdxT, topkProbsT, &ws, &ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;
    CHECK_ACL(aclnnMoeTopKLite(workspace, ws, ex, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto t1 = std::chrono::high_resolution_clock::now();
    aclDestroyTensor(logitsT);
    aclDestroyTensor(topkIdxT);
    aclDestroyTensor(topkProbsT);
    if (workspace) {
        aclrtFree(workspace);
    }
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
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

static bool SameI32(const std::vector<int32_t> &a, const std::vector<int32_t> &b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    try {
        if (argc != 5) {
            std::cerr << "usage: main_benchmark <data_dir> <num_tokens> <hidden_size> <top_k>\n";
            return 1;
        }
        std::string dataDir = argv[1];
        int64_t T = std::stoll(argv[2]);
        int64_t H = std::stoll(argv[3]);
        int64_t K = std::stoll(argv[4]);
        int64_t total = T * K;

        if (K != LITE_FIXED_TOP_K) {
            throw std::runtime_error("Lite teaching package currently supports top_k=2 only. Regenerate data with --top_k 2.");
        }

        std::filesystem::create_directories(dataDir + "/output");

        auto logitsHost = ReadBinary(dataDir + "/input/logits.bin");
        auto tokensHost = ReadBinary(dataDir + "/input/tokens.bin");
        auto sortedOrderHost = ReadBinary(dataDir + "/input/sorted_order.bin");
        auto expertOutHost = ReadBinary(dataDir + "/input/expert_out.bin");

        int64_t E = static_cast<int64_t>(logitsHost.size()) / (T * FP16_BYTES);

        AclRuntimeGuard guard(0);
        aclrtStream stream = guard.stream();

        void *logitsDev = MallocDevice(logitsHost.size());
        void *tokensDev = MallocDevice(tokensHost.size());
        void *sortedOrderDev = MallocDevice(sortedOrderHost.size());
        void *expertOutDev = MallocDevice(expertOutHost.size());

        void *topkIdxDev = MallocDevice(T * K * I32_BYTES);
        void *topkProbsDev = MallocDevice(T * K * FP16_BYTES);
        void *quickIdxDev = MallocDevice(T * K * I32_BYTES);
        void *quickProbsDev = MallocDevice(T * K * FP16_BYTES);
        void *heapIdxDev = MallocDevice(T * K * I32_BYTES);
        void *heapProbsDev = MallocDevice(T * K * FP16_BYTES);

        void *permTokensDev = MallocDevice(total * H * FP16_BYTES);
        void *sortedIdxDev = MallocDevice(total * I32_BYTES);
        void *permProbsDev = MallocDevice(total * FP16_BYTES);
        void *unpermuteOutDev = MallocDevice(T * H * FP16_BYTES);

        CHECK_ACL(aclrtMemcpy(logitsDev, logitsHost.size(), logitsHost.data(), logitsHost.size(), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(tokensDev, tokensHost.size(), tokensHost.data(), tokensHost.size(), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(sortedOrderDev, sortedOrderHost.size(), sortedOrderHost.data(), sortedOrderHost.size(), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(expertOutDev, expertOutHost.size(), expertOutHost.data(), expertOutHost.size(), ACL_MEMCPY_HOST_TO_DEVICE));

        std::cout << "=== MoE Lite Parallel Routing Benchmark ===\n";
        std::cout << "T=" << T << " H=" << H << " K=" << K << " E=" << E << " total=" << total << "\n\n";
        std::cout << "Routing selection algorithms are parallel alternatives, not a serial chain.\n";
        std::cout << "Each algorithm reads the same logits and writes [T,K] expert indices plus probabilities.\n\n";

        double topkMs = RunTopK(logitsDev, topkIdxDev, topkProbsDev, T, E, K, stream);
        std::cout << "Selection TopK:       " << topkMs << " ms\n";

        double quickMs = RunQuickSortRoute(logitsDev, quickIdxDev, quickProbsDev, T, E, K, stream);
        std::cout << "Full QuickSort TopK:  " << quickMs << " ms\n";

        double heapMs = RunHeapSortRoute(logitsDev, heapIdxDev, heapProbsDev, T, E, K, stream);
        std::cout << "Heap Extract TopK:    " << heapMs << " ms\n";

        std::vector<int32_t> topkIdxHost(T * K), quickIdxHost(T * K), heapIdxHost(T * K);
        std::vector<uint8_t> topkProbsHost(T * K * FP16_BYTES);
        std::vector<uint8_t> quickProbsHost(T * K * FP16_BYTES);
        std::vector<uint8_t> heapProbsHost(T * K * FP16_BYTES);
        CHECK_ACL(aclrtMemcpy(topkIdxHost.data(), T * K * I32_BYTES, topkIdxDev, T * K * I32_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(quickIdxHost.data(), T * K * I32_BYTES, quickIdxDev, T * K * I32_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(heapIdxHost.data(), T * K * I32_BYTES, heapIdxDev, T * K * I32_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(topkProbsHost.data(), T * K * FP16_BYTES, topkProbsDev, T * K * FP16_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(quickProbsHost.data(), T * K * FP16_BYTES, quickProbsDev, T * K * FP16_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(heapProbsHost.data(), T * K * FP16_BYTES, heapProbsDev, T * K * FP16_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));

        std::cout << "TopK vs QuickSort indices match: " << (SameI32(topkIdxHost, quickIdxHost) ? "YES" : "NO") << "\n";
        std::cout << "TopK vs HeapSort indices match:  " << (SameI32(topkIdxHost, heapIdxHost) ? "YES" : "NO") << "\n\n";

        // Permute/Unpermute are shared MoE data-movement stages. They are not added to the
        // three routing-selection timings above. This demo uses the precomputed sortedOrder
        // generated from the TopK reference so that the token movement kernels can still be verified.
        double permuteMs = RunPermute(tokensDev, sortedOrderDev, topkProbsDev,
                                      permTokensDev, sortedIdxDev, permProbsDev,
                                      T, H, K, stream);
        std::cout << "Shared Permute:       " << permuteMs << " ms\n";

        double unpermuteMs = RunUnpermute(expertOutDev, sortedIdxDev, permProbsDev, unpermuteOutDev,
                                          T, H, K, stream);
        std::cout << "Shared Unpermute:     " << unpermuteMs << " ms\n\n";

        std::vector<uint8_t> permTokensHost(total * H * FP16_BYTES);
        std::vector<int32_t> sortedIdxHost(total);
        std::vector<uint8_t> permProbsHost(total * FP16_BYTES);
        std::vector<uint8_t> unpermuteOutHost(T * H * FP16_BYTES);
        CHECK_ACL(aclrtMemcpy(permTokensHost.data(), total * H * FP16_BYTES, permTokensDev, total * H * FP16_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(sortedIdxHost.data(), total * I32_BYTES, sortedIdxDev, total * I32_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(permProbsHost.data(), total * FP16_BYTES, permProbsDev, total * FP16_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(unpermuteOutHost.data(), T * H * FP16_BYTES, unpermuteOutDev, T * H * FP16_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));

        WriteBinary(dataDir + "/output/topk_indices_lite.bin", topkIdxHost.data(), topkIdxHost.size() * I32_BYTES);
        WriteBinary(dataDir + "/output/topk_probs_lite.bin", topkProbsHost.data(), topkProbsHost.size());
        WriteBinary(dataDir + "/output/quicksort_indices_lite.bin", quickIdxHost.data(), quickIdxHost.size() * I32_BYTES);
        WriteBinary(dataDir + "/output/quicksort_probs_lite.bin", quickProbsHost.data(), quickProbsHost.size());
        WriteBinary(dataDir + "/output/heapsort_indices_lite.bin", heapIdxHost.data(), heapIdxHost.size() * I32_BYTES);
        WriteBinary(dataDir + "/output/heapsort_probs_lite.bin", heapProbsHost.data(), heapProbsHost.size());
        WriteBinary(dataDir + "/output/permuted_tokens_lite.bin", permTokensHost.data(), permTokensHost.size());
        WriteBinary(dataDir + "/output/sorted_indices_lite.bin", sortedIdxHost.data(), sortedIdxHost.size() * I32_BYTES);
        WriteBinary(dataDir + "/output/permuted_probs_lite.bin", permProbsHost.data(), permProbsHost.size());
        WriteBinary(dataDir + "/output/unpermute_out_lite.bin", unpermuteOutHost.data(), unpermuteOutHost.size());

        std::cout << "=== Routing Summary: lower is better ===\n";
        std::cout << "Selection TopK:       " << topkMs << " ms\n";
        std::cout << "Full QuickSort TopK:  " << quickMs << " ms\n";
        std::cout << "Heap Extract TopK:    " << heapMs << " ms\n";
        std::cout << "Shared Permute:       " << permuteMs << " ms\n";
        std::cout << "Shared Unpermute:     " << unpermuteMs << " ms\n";

        aclrtFree(logitsDev);
        aclrtFree(tokensDev);
        aclrtFree(sortedOrderDev);
        aclrtFree(expertOutDev);
        aclrtFree(topkIdxDev);
        aclrtFree(topkProbsDev);
        aclrtFree(quickIdxDev);
        aclrtFree(quickProbsDev);
        aclrtFree(heapIdxDev);
        aclrtFree(heapProbsDev);
        aclrtFree(permTokensDev);
        aclrtFree(sortedIdxDev);
        aclrtFree(permProbsDev);
        aclrtFree(unpermuteOutDev);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 99;
    }
}
