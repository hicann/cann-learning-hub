// RoPE SIMD RTC Host（Ascend 910B3 / CANN 9.0）。
// 关键链路：aclrtcCompileProg -> aclrtBinaryLoadFromData ->
// aclrtBinaryGetFunction -> aclrtLaunchKernelWithConfig -> stream sync。
// Host 把 [12,64] interleaved 输入拆成 even/odd pair-planar 布局，设备输出再与
// position={0,1,20,95} 的 CPU reference 比较。RTC 编译、Kernel 和 reference
// 分开计时；stdout 最终只输出一行 ROPE_RESULT。
// 用法：rope_simd_rtc --kernel <kernel.cpp> [--warmup N] [--repeat N]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "acl/acl_rt_compile.h"

// ---------------------------------------------------------------------------
// 常量与教学契约（pair-planar：12 heads x 32 pairs = 384 连续 pair）
// ---------------------------------------------------------------------------
static constexpr int32_t kTotalPairs = 384;
static constexpr int32_t kHeads = 12;
static constexpr int32_t kPairsPerHead = 32;
static constexpr int32_t kDim = 2 * kPairsPerHead;  // 每 head 维数 64（32 对）
static constexpr int32_t kTotalDim = kHeads * kDim; // interleaved [12,64] = 768
static constexpr float kDefaultTolerance = 1e-5f;
static constexpr int kDefaultWarmup = 2;
static constexpr int kDefaultRepeat = 5;
static const std::vector<int32_t> kPositions = {0, 1, 20, 95};

// ---------------------------------------------------------------------------
// 错误处理：结构化异常式（KISS），所有失败最终进入统一释放段。
// ---------------------------------------------------------------------------
struct RopeFailure {
    const char* point;
    explicit RopeFailure(const char* p) : point(p) {}
};

#define CHECK_ACL(x)                                                            \
    do {                                                                        \
        aclError __ret = (x);                                                   \
        if (__ret != ACL_ERROR_NONE) {                                          \
            fprintf(stderr, "[rope_simd_rtc] %s:%d ACL error %d from %s\n",     \
                    __FILE__, __LINE__, static_cast<int>(__ret), #x);           \
            const char* __msg = aclGetRecentErrMsg();                           \
            if (__msg != nullptr) {                                             \
                fprintf(stderr, "[rope_simd_rtc] %s\n", __msg);                 \
            }                                                                   \
            throw RopeFailure(#x);                                              \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// 数据生成与 CPU reference
// ---------------------------------------------------------------------------

// 输入链：先生成确定性 interleaved [12,64] FP32 输入，再 split 为
// pair-planar 连续 even/odd 数组；cos/sin 按 head 复制。
static void GenCase(int32_t pos, std::vector<float>& xInterleaved,
                    std::vector<float>& xEven, std::vector<float>& xOdd,
                    std::vector<float>& cos, std::vector<float>& sin)
{
    xInterleaved.resize(kTotalDim);
    for (int32_t k = 0; k < kTotalDim; ++k) {
        xInterleaved[k] = static_cast<float>(((k * 37 + 11) % 100) - 50) / 100.0f;
    }
    // interleaved -> pair-planar split（Host 完成布局转换）
    xEven.resize(kTotalPairs);
    xOdd.resize(kTotalPairs);
    for (int32_t h = 0; h < kHeads; ++h) {
        for (int32_t i = 0; i < kPairsPerHead; ++i) {
            xEven[h * kPairsPerHead + i] = xInterleaved[h * kDim + 2 * i];
            xOdd[h * kPairsPerHead + i]  = xInterleaved[h * kDim + 2 * i + 1];
        }
    }
    cos.resize(kTotalPairs);
    sin.resize(kTotalPairs);
    for (int32_t h = 0; h < kHeads; ++h) {
        for (int32_t i = 0; i < kPairsPerHead; ++i) {
            const double theta = 1.0 / std::pow(10000.0, (2.0 * i) / static_cast<double>(kDim));
            const double angle = static_cast<double>(pos) * theta;
            cos[h * kPairsPerHead + i] = static_cast<float>(std::cos(angle));
            sin[h * kPairsPerHead + i] = static_cast<float>(std::sin(angle));
        }
    }
}

// CPU reference：double 精度实现同一 pair-planar 语义。
static void ReferenceRope(const std::vector<float>& xEven, const std::vector<float>& xOdd,
                          const std::vector<float>& cos, const std::vector<float>& sin,
                          std::vector<double>& refEven, std::vector<double>& refOdd)
{
    refEven.resize(kTotalPairs);
    refOdd.resize(kTotalPairs);
    for (int32_t p = 0; p < kTotalPairs; ++p) {
        refEven[p] = static_cast<double>(xEven[p]) * static_cast<double>(cos[p])
                   - static_cast<double>(xOdd[p]) * static_cast<double>(sin[p]);
        refOdd[p] = static_cast<double>(xEven[p]) * static_cast<double>(sin[p])
                  + static_cast<double>(xOdd[p]) * static_cast<double>(cos[p]);
    }
}

// pair-planar even/odd -> interleaved 布局重排（double reference 版）。
static void ReassembleInterleaved(const std::vector<double>& refEven,
                                  const std::vector<double>& refOdd,
                                  std::vector<double>& refInterleaved)
{
    refInterleaved.resize(kTotalDim);
    for (int32_t h = 0; h < kHeads; ++h) {
        for (int32_t i = 0; i < kPairsPerHead; ++i) {
            refInterleaved[h * kDim + 2 * i]     = refEven[h * kPairsPerHead + i];
            refInterleaved[h * kDim + 2 * i + 1] = refOdd[h * kPairsPerHead + i];
        }
    }
}

// pair-planar even/odd -> interleaved 布局重排（float 设备输出版）。
static void ReassembleInterleaved(const float* devEven, const float* devOdd,
                                  std::vector<float>& devInterleaved)
{
    devInterleaved.resize(kTotalDim);
    for (int32_t h = 0; h < kHeads; ++h) {
        for (int32_t i = 0; i < kPairsPerHead; ++i) {
            devInterleaved[h * kDim + 2 * i]     = devEven[h * kPairsPerHead + i];
            devInterleaved[h * kDim + 2 * i + 1] = devOdd[h * kPairsPerHead + i];
        }
    }
}

// 设备 float 输出与 double reference 的最大绝对误差（interleaved 布局逐元素）。
static float MaxAbsError(const std::vector<double>& ref, const float* dev, int32_t n)
{
    double maxErr = 0.0;
    for (int32_t p = 0; p < n; ++p) {
        const double err = std::fabs(static_cast<double>(dev[p]) - ref[p]);
        if (err > maxErr) {
            maxErr = err;
        }
    }
    return static_cast<float>(maxErr);
}

// 亚微秒精度：duration<double, micro> 保留小数微秒，不做整数截断。
static std::chrono::duration<double, std::micro> NowUs()
{
    return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
        std::chrono::steady_clock::now().time_since_epoch());
}

// ---------------------------------------------------------------------------
// 参数解析
// ---------------------------------------------------------------------------
static bool ParseArgs(int argc, char* argv[], std::string& kernelPath, int& warmup, int& repeat)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--kernel" && i + 1 < argc) {
            kernelPath = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
        } else {
            return false;
        }
    }
    return !kernelPath.empty() && warmup >= 0 && repeat >= 1;
}

static bool ReadFileToString(const std::string& path, std::string& out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return !out.empty();
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------
struct RopeState {
    std::string kernelPath;
    int warmup = kDefaultWarmup;
    int repeat = kDefaultRepeat;
    bool ok = false;
    bool aclInited = false;
    bool deviceSet = false;
    bool progCreated = false;
    const char* errPoint = "init";
    const char* compileStatus = "fail";
    std::string kernelSrc;
    std::vector<char> elfBin;
    size_t elfSize = 0;
    aclrtStream stream = nullptr;
    void* dXEven = nullptr;
    void* dXOdd = nullptr;
    void* dCos = nullptr;
    void* dSin = nullptr;
    void* dOutEven = nullptr;
    void* dOutOdd = nullptr;
    uint8_t* outEvenHost = nullptr;
    uint8_t* outOddHost = nullptr;
    aclrtcProg prog = nullptr;
    aclrtBinHandle binHandle = nullptr;
    aclrtFuncHandle funcHandle = nullptr;
    aclrtArgsHandle argsHandle = nullptr;
    aclrtParamHandle paramHandle = nullptr;
    int casesTotal = static_cast<int>(kPositions.size());
    int casesPassed = 0;
    float maxError = 0.0f;
    double deviceMeanUs = 0.0;
    double referenceMeanUs = 0.0;
    double deviceToReferenceRatio = 0.0;
    double compileSeconds = 0.0;
};

static constexpr size_t BufferBytes()
{
    return static_cast<size_t>(kTotalPairs) * sizeof(float);
}

static void InitializeRuntime(RopeState* state)
{
    if (!ReadFileToString(state->kernelPath, state->kernelSrc)) {
        fprintf(stderr, "[rope_simd_rtc] cannot read kernel source: %s\n",
                state->kernelPath.c_str());
        throw RopeFailure("read_kernel_source");
    }
    CHECK_ACL(aclInit(nullptr));
    state->aclInited = true;
    CHECK_ACL(aclrtSetDevice(0));
    state->deviceSet = true;
    CHECK_ACL(aclrtCreateStream(&state->stream));
}

static void CompileKernel(RopeState* state)
{
    CHECK_ACL(aclrtcCreateProg(&state->prog, state->kernelSrc.c_str(),
                               "rope_simd", 0, nullptr, nullptr));
    state->progCreated = true;
    const char* options[] = {"--npu-arch=dav-2201"};
    const auto begin = NowUs();
    const aclError result = aclrtcCompileProg(state->prog, 1, options);
    state->compileSeconds = (NowUs() - begin).count() / 1e6;
    if (result != ACL_ERROR_NONE) {
        fprintf(stderr, "[rope_simd_rtc] ACL error %d from aclrtcCompileProg\n",
                static_cast<int>(result));
        const char* message = aclGetRecentErrMsg();
        if (message != nullptr) fprintf(stderr, "[rope_simd_rtc] %s\n", message);
        throw RopeFailure("aclrtcCompileProg");
    }
    state->compileStatus = "ok";
    fprintf(stderr, "[rope_simd_rtc] aclrtcCompileProg ok, %.9f s\n",
            state->compileSeconds);
    CHECK_ACL(aclrtcGetBinDataSize(state->prog, &state->elfSize));
    state->elfBin.resize(state->elfSize);
    CHECK_ACL(aclrtcGetBinData(state->prog, state->elfBin.data()));
}

static void AllocateBuffers(RopeState* state)
{
    const size_t bytes = BufferBytes();
    CHECK_ACL(aclrtMalloc(&state->dXEven, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&state->dXOdd, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&state->dCos, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&state->dSin, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&state->dOutEven, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&state->dOutOdd, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void**>(&state->outEvenHost), bytes));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void**>(&state->outOddHost), bytes));
}

static void LoadKernel(RopeState* state)
{
    aclrtBinaryLoadOptions loadOption{};
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_LAZY_MAGIC;
    option.value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
    loadOption.numOpt = 1;
    loadOption.options = &option;
    CHECK_ACL(aclrtBinaryLoadFromData(state->elfBin.data(), state->elfSize,
                                      &loadOption, &state->binHandle));
    CHECK_ACL(aclrtBinaryGetFunction(state->binHandle, "rope_simd", &state->funcHandle));
}

static void AppendPointerArg(RopeState* state, void** address)
{
    CHECK_ACL(aclrtKernelArgsAppend(state->argsHandle, address,
                                    sizeof(uintptr_t), &state->paramHandle));
}

static void BindArguments(RopeState* state)
{
    CHECK_ACL(aclrtKernelArgsInit(state->funcHandle, &state->argsHandle));
    AppendPointerArg(state, reinterpret_cast<void**>(&state->dXEven));
    AppendPointerArg(state, reinterpret_cast<void**>(&state->dXOdd));
    AppendPointerArg(state, reinterpret_cast<void**>(&state->dCos));
    AppendPointerArg(state, reinterpret_cast<void**>(&state->dSin));
    AppendPointerArg(state, reinterpret_cast<void**>(&state->dOutEven));
    AppendPointerArg(state, reinterpret_cast<void**>(&state->dOutOdd));
    CHECK_ACL(aclrtKernelArgsFinalize(state->argsHandle));
}

static void CopyInputs(RopeState* state, const std::vector<float>& xEven,
                       const std::vector<float>& xOdd,
                       const std::vector<float>& cos,
                       const std::vector<float>& sin)
{
    const size_t bytes = BufferBytes();
    CHECK_ACL(aclrtMemcpy(state->dXEven, bytes, xEven.data(), bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(state->dXOdd, bytes, xOdd.data(), bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(state->dCos, bytes, cos.data(), bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(state->dSin, bytes, sin.data(), bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
}

static void LaunchAndSync(RopeState* state)
{
    CHECK_ACL(aclrtLaunchKernelWithConfig(state->funcHandle, 1, state->stream,
                                          nullptr, state->argsHandle, nullptr));
    CHECK_ACL(aclrtSynchronizeStream(state->stream));
}

static void RunCorrectnessCase(int32_t position, RopeState* state)
{
    std::vector<float> xInterleaved, xEven, xOdd, cos, sin;
    std::vector<double> refEven, refOdd, refInterleaved;
    GenCase(position, xInterleaved, xEven, xOdd, cos, sin);
    ReferenceRope(xEven, xOdd, cos, sin, refEven, refOdd);
    ReassembleInterleaved(refEven, refOdd, refInterleaved);
    CopyInputs(state, xEven, xOdd, cos, sin);
    LaunchAndSync(state);
    const size_t bytes = BufferBytes();
    CHECK_ACL(aclrtMemcpy(state->outEvenHost, bytes, state->dOutEven, bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(state->outOddHost, bytes, state->dOutOdd, bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> deviceInterleaved;
    ReassembleInterleaved(reinterpret_cast<const float*>(state->outEvenHost),
                          reinterpret_cast<const float*>(state->outOddHost),
                          deviceInterleaved);
    const float error = MaxAbsError(refInterleaved, deviceInterleaved.data(), kTotalDim);
    state->maxError = std::max(state->maxError, error);
    const bool passed = error <= kDefaultTolerance;
    if (passed) ++state->casesPassed;
    fprintf(stderr, "[rope_simd_rtc] case pos=%d max_abs_error=%.6e %s\n",
            position, error, passed ? "PASS" : "FAIL");
}

static void RunCorrectness(RopeState* state)
{
    for (const int32_t position : kPositions) RunCorrectnessCase(position, state);
}

static void RunWarmup(RopeState* state)
{
    for (int i = 0; i < state->warmup; ++i) LaunchAndSync(state);
}

static double MeasureDevice(RopeState* state, const std::vector<float>& xEven,
                            const std::vector<float>& xOdd,
                            const std::vector<float>& cos,
                            const std::vector<float>& sin)
{
    CopyInputs(state, xEven, xOdd, cos, sin);
    double total = 0.0;
    for (int repeat = 0; repeat < state->repeat; ++repeat) {
        const auto begin = NowUs();
        LaunchAndSync(state);
        total += (NowUs() - begin).count();
    }
    return total / static_cast<double>(state->repeat);
}

static double MeasureReference(RopeState* state, const std::vector<float>& xEven,
                               const std::vector<float>& xOdd,
                               const std::vector<float>& cos,
                               const std::vector<float>& sin)
{
    std::vector<double> refEven(kTotalPairs), refOdd(kTotalPairs);
    double total = 0.0;
    for (int repeat = 0; repeat < state->repeat; ++repeat) {
        const auto begin = NowUs();
        ReferenceRope(xEven, xOdd, cos, sin, refEven, refOdd);
        total += (NowUs() - begin).count();
    }
    return total / static_cast<double>(state->repeat);
}

static void MeasureTimings(RopeState* state)
{
    for (const int32_t position : kPositions) {
        std::vector<float> interleaved, xEven, xOdd, cos, sin;
        GenCase(position, interleaved, xEven, xOdd, cos, sin);
        state->deviceMeanUs += MeasureDevice(state, xEven, xOdd, cos, sin);
        state->referenceMeanUs += MeasureReference(state, xEven, xOdd, cos, sin);
    }
    state->deviceMeanUs /= static_cast<double>(state->casesTotal);
    state->referenceMeanUs /= static_cast<double>(state->casesTotal);
    if (state->referenceMeanUs > 0.0) {
        state->deviceToReferenceRatio = state->deviceMeanUs / state->referenceMeanUs;
    }
}

static void ReleaseResources(RopeState* state)
{
    if (state->binHandle != nullptr) aclrtBinaryUnLoad(state->binHandle);
    if (state->dOutOdd != nullptr) aclrtFree(state->dOutOdd);
    if (state->dOutEven != nullptr) aclrtFree(state->dOutEven);
    if (state->dSin != nullptr) aclrtFree(state->dSin);
    if (state->dCos != nullptr) aclrtFree(state->dCos);
    if (state->dXOdd != nullptr) aclrtFree(state->dXOdd);
    if (state->dXEven != nullptr) aclrtFree(state->dXEven);
    if (state->outOddHost != nullptr) aclrtFreeHost(state->outOddHost);
    if (state->outEvenHost != nullptr) aclrtFreeHost(state->outEvenHost);
    if (state->stream != nullptr) aclrtDestroyStream(state->stream);
    if (state->deviceSet) aclrtResetDevice(0);
    if (state->aclInited) aclFinalize();
    if (state->progCreated) aclrtcDestroyProg(&state->prog);
}

static void PrintResult(const RopeState& state)
{
    if (state.ok) {
        fprintf(stdout,
                "ROPE_RESULT status=PASS cases=%d/%d max_error=%.6e tolerance=%.6e "
                "compile=ok compile_seconds=%.9f device_mean_us=%.9f reference_mean_us=%.9f "
                "device_to_reference_ratio=%.9f fallback=0 path=ASCENDC_SIMD_RTC device_id=0\n",
                state.casesPassed, state.casesTotal, state.maxError, kDefaultTolerance,
                state.compileSeconds, state.deviceMeanUs, state.referenceMeanUs,
                state.deviceToReferenceRatio);
        return;
    }
    fprintf(stdout,
            "ROPE_RESULT status=FAIL cases=%d/%d max_error=%.6e tolerance=%.6e "
            "compile=%s compile_seconds=%.9f device_mean_us=%.9f reference_mean_us=%.9f "
            "device_to_reference_ratio=%.9f fallback=0 path=ASCENDC_SIMD_RTC device_id=0 error=%s\n",
            state.casesPassed, state.casesTotal, state.maxError, kDefaultTolerance,
            state.compileStatus, state.compileSeconds, state.deviceMeanUs,
            state.referenceMeanUs, state.deviceToReferenceRatio, state.errPoint);
}

static void RunExperiment(RopeState* state)
{
    InitializeRuntime(state);
    CompileKernel(state);
    AllocateBuffers(state);
    LoadKernel(state);
    BindArguments(state);
    RunCorrectness(state);
    RunWarmup(state);
    MeasureTimings(state);
    state->ok = state->casesPassed == state->casesTotal &&
                state->maxError <= kDefaultTolerance;
}

int main(int argc, char* argv[])
{
    RopeState state;
    if (!ParseArgs(argc, argv, state.kernelPath, state.warmup, state.repeat)) {
        fprintf(stderr,
                "[rope_simd_rtc] usage: %s --kernel <kernel.cpp> [--warmup N] [--repeat N]\n",
                argv[0]);
        return 1;
    }
    try {
        RunExperiment(&state);
    } catch (const RopeFailure& error) {
        state.errPoint = error.point;
    }
    ReleaseResources(&state);
    PrintResult(state);
    return state.ok ? 0 : 1;
}
