#include "common/acl_utils.h"
#include "common/bin_utils.h"

#include <aclnn_bracket_match_lite.h>
#include <aclnn_suffix_eval_lite.h>
#include <aclnn_infix_to_postfix_lite.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static constexpr uint32_t BLOCK_DIM = 8;
static constexpr uint32_t EXPR_LEN = 128;
static constexpr uint32_t TOKEN_LEN = 64;
static constexpr int WARMUP_ITERS = 3;
static constexpr int TIMED_ITERS = 10;

template <typename Fn>
static double RunOpTimed(aclrtStream stream, int warmup, int iters, Fn &&fn) {
    aclrtEvent startEvt, endEvt;
    CHECK_ACL(aclrtCreateEvent(&startEvt));
    CHECK_ACL(aclrtCreateEvent(&endEvt));
    for (int i = 0; i < warmup; ++i) fn();
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtRecordEvent(startEvt, stream));
    for (int i = 0; i < iters; ++i) fn();
    CHECK_ACL(aclrtRecordEvent(endEvt, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    float elapsedMs = 0.0f;
    CHECK_ACL(aclrtEventElapsedTime(&elapsedMs, startEvt, endEvt));
    aclrtDestroyEvent(startEvt);
    aclrtDestroyEvent(endEvt);
    return static_cast<double>(elapsedMs) / iters;
}

// === BracketMatch Benchmark ===
static bool RunBracketMatch(const std::string &dataDir, aclrtStream stream) {
    auto inputBin = ReadBinary(dataDir + "/input/bracket_input.bin");
    auto refBin = ReadBinary(dataDir + "/input/bracket_ref.bin");

    int64_t totalLen = BLOCK_DIM * EXPR_LEN;
    void *xDev = MallocDevice(inputBin.size());
    void *yDev = MallocDevice(BLOCK_DIM * sizeof(int32_t));
    CHECK_ACL(aclrtMemcpy(xDev, inputBin.size(), inputBin.data(),
                          inputBin.size(), ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *xT = CreateTensor({totalLen}, ACL_INT8, ACL_FORMAT_ND, xDev);
    aclTensor *yT = CreateTensor({(int64_t)BLOCK_DIM}, ACL_INT32, ACL_FORMAT_ND, yDev);

    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK_ACL(aclnnBracketMatchLiteGetWorkspaceSize(xT, EXPR_LEN, yT, &ws, &ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;

    double ms = RunOpTimed(stream, WARMUP_ITERS, TIMED_ITERS, [&]() {
        CHECK_ACL(aclnnBracketMatchLite(workspace, ws, ex, stream));
    });
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<int32_t> results(BLOCK_DIM);
    CHECK_ACL(aclrtMemcpy(results.data(), BLOCK_DIM * sizeof(int32_t),
                          yDev, BLOCK_DIM * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<int32_t> refs(BLOCK_DIM);
    memcpy(refs.data(), refBin.data(), BLOCK_DIM * sizeof(int32_t));

    std::cout << "[BracketMatch] time=" << std::fixed << std::setprecision(4) << ms << " ms\n";
    bool allPass = true;
    for (uint32_t i = 0; i < BLOCK_DIM; ++i) {
        bool pass = (results[i] == refs[i]);
        if (!pass) allPass = false;
        std::cout << "  expr[" << i << "]: result=" << results[i]
                  << " ref=" << refs[i] << (pass ? " OK" : " FAIL") << "\n";
    }
    std::cout << "  " << (allPass ? "PASS" : "FAIL") << "\n";

    aclDestroyTensor(xT); aclDestroyTensor(yT);
    aclrtFree(xDev); aclrtFree(yDev);
    if (workspace) aclrtFree(workspace);
    return allPass;
}

// === SuffixEval Benchmark ===
static bool RunSuffixEval(const std::string &dataDir, aclrtStream stream) {
    auto inputBin = ReadBinary(dataDir + "/input/suffix_input.bin");
    auto refBin = ReadBinary(dataDir + "/input/suffix_ref.bin");

    int64_t totalLen = BLOCK_DIM * TOKEN_LEN;
    void *xDev = MallocDevice(inputBin.size());
    void *yDev = MallocDevice(BLOCK_DIM * sizeof(float));
    CHECK_ACL(aclrtMemcpy(xDev, inputBin.size(), inputBin.data(),
                          inputBin.size(), ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *xT = CreateTensor({totalLen}, ACL_INT32, ACL_FORMAT_ND, xDev);
    aclTensor *yT = CreateTensor({(int64_t)BLOCK_DIM}, ACL_FLOAT, ACL_FORMAT_ND, yDev);

    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK_ACL(aclnnSuffixEvalLiteGetWorkspaceSize(xT, TOKEN_LEN, yT, &ws, &ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;

    double ms = RunOpTimed(stream, WARMUP_ITERS, TIMED_ITERS, [&]() {
        CHECK_ACL(aclnnSuffixEvalLite(workspace, ws, ex, stream));
    });
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<float> results(BLOCK_DIM);
    CHECK_ACL(aclrtMemcpy(results.data(), BLOCK_DIM * sizeof(float),
                          yDev, BLOCK_DIM * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> refs(BLOCK_DIM);
    memcpy(refs.data(), refBin.data(), BLOCK_DIM * sizeof(float));

    std::cout << "[SuffixEval]   time=" << std::fixed << std::setprecision(4) << ms << " ms\n";
    bool allPass = true;
    for (uint32_t i = 0; i < BLOCK_DIM; ++i) {
        float err = std::fabs(results[i] - refs[i]);
        bool pass = (err < 0.001f);
        if (!pass) allPass = false;
        std::cout << "  expr[" << i << "]: result=" << std::setprecision(4)
                  << results[i] << " ref=" << refs[i]
                  << " err=" << err << (pass ? " OK" : " FAIL") << "\n";
    }
    std::cout << "  " << (allPass ? "PASS" : "FAIL") << "\n";

    aclDestroyTensor(xT); aclDestroyTensor(yT);
    aclrtFree(xDev); aclrtFree(yDev);
    if (workspace) aclrtFree(workspace);
    return allPass;
}

// === InfixToPostfix Benchmark ===
static bool RunInfixToPostfix(const std::string &dataDir, aclrtStream stream) {
    auto inputBin = ReadBinary(dataDir + "/input/infix_input.bin");
    auto refBin = ReadBinary(dataDir + "/input/postfix_ref.bin");

    int64_t totalLen = BLOCK_DIM * EXPR_LEN;
    void *xDev = MallocDevice(inputBin.size());
    void *yDev = MallocDevice(inputBin.size());
    CHECK_ACL(aclrtMemcpy(xDev, inputBin.size(), inputBin.data(),
                          inputBin.size(), ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *xT = CreateTensor({totalLen}, ACL_INT8, ACL_FORMAT_ND, xDev);
    aclTensor *yT = CreateTensor({totalLen}, ACL_INT8, ACL_FORMAT_ND, yDev);

    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK_ACL(aclnnInfixToPostfixLiteGetWorkspaceSize(xT, EXPR_LEN, yT, &ws, &ex));
    void *workspace = ws > 0 ? MallocDevice(ws) : nullptr;

    double ms = RunOpTimed(stream, WARMUP_ITERS, TIMED_ITERS, [&]() {
        CHECK_ACL(aclnnInfixToPostfixLite(workspace, ws, ex, stream));
    });
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<int8_t> results(totalLen);
    CHECK_ACL(aclrtMemcpy(results.data(), totalLen, yDev, totalLen, ACL_MEMCPY_DEVICE_TO_HOST));

    std::cout << "[InfixToPostfix] time=" << std::fixed << std::setprecision(4) << ms << " ms\n";
    bool allPass = true;
    for (uint32_t i = 0; i < BLOCK_DIM; ++i) {
        bool pass = true;
        for (uint32_t j = 0; j < EXPR_LEN; ++j) {
            int8_t r = results[i * EXPR_LEN + j];
            int8_t ref = ((int8_t*)refBin.data())[i * EXPR_LEN + j];
            if (r != ref) { pass = false; break; }
            if (r == '#') break;
        }
        if (!pass) allPass = false;
        std::cout << "  expr[" << i << "]: ";
        for (uint32_t j = 0; j < EXPR_LEN; ++j) {
            char c = (char)results[i * EXPR_LEN + j];
            if (c == '#') break;
            std::cout << c;
        }
        std::cout << (pass ? " OK" : " FAIL") << "\n";
    }
    std::cout << "  " << (allPass ? "PASS" : "FAIL") << "\n";

    aclDestroyTensor(xT); aclDestroyTensor(yT);
    aclrtFree(xDev); aclrtFree(yDev);
    if (workspace) aclrtFree(workspace);
    return allPass;
}

int main(int argc, char **argv) {
    try {
        std::string dataDir = (argc >= 2) ? argv[1] : "data";

        AclRuntimeGuard guard(0);
        aclrtStream stream = guard.stream();

        std::cout << "=== Stack Expression Evaluation Lab Benchmark ===\n";
        std::cout << "BLOCK_DIM=" << BLOCK_DIM << " EXPR_LEN=" << EXPR_LEN
                  << " TOKEN_LEN=" << TOKEN_LEN << "\n\n";

        bool p1 = RunBracketMatch(dataDir, stream);
        std::cout << "\n";
        bool p2 = RunSuffixEval(dataDir, stream);
        std::cout << "\n";
        bool p3 = RunInfixToPostfix(dataDir, stream);

        int passCount = (p1 ? 1 : 0) + (p2 ? 1 : 0) + (p3 ? 1 : 0);
        std::cout << "\n=== Result: " << passCount << "/3 PASS ===\n";
        return (passCount == 3) ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 99;
    }
}
