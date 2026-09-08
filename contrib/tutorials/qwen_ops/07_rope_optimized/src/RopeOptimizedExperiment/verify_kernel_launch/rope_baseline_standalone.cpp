// RoPE 通用版 — ACLRT_LAUNCH_KERNEL 独立验证 (Host 端)
//
// 用法: build.sh 编译后直接运行 binary 进行正确性与性能验证
// 需要 input/input_x.bin, input/input_cos.bin, input/input_sin.bin

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"

// ── Tiling 结构 (与 kernel 共用) ─────────────────────────────────
#pragma pack(push, 1)
struct RoPeTiling {
    uint32_t totalTokens = 0;
    uint32_t headDim     = 0;
    uint32_t coreNum     = 1;
    uint32_t rowsPerCore = 0;
    uint32_t seqLen      = 0;
    uint32_t numHeads    = 1;
    uint32_t trigTokens  = 0;
    uint32_t compactTrig = 0;
    uint32_t tileSize    = 0;   // >1 enables DataCopy+Vector path
};
#pragma pack(pop)

// ── ACLRT launch ────────────────────────────────────────────────
#include "aclrtlaunch_rope_baseline_kernel.h"

#define CHECK_ACL(call)                                                 \
    do {                                                                \
        const aclError ret = (call);                                     \
        if (ret != ACL_SUCCESS) {                                        \
            std::fprintf(stderr, "[ERROR] ACL fail %s:%d ret=%d\n",     \
                __FILE__, __LINE__, static_cast<int>(ret));              \
            return -1;                                                  \
        }                                                               \
    } while (0)

// ── 文件 I/O ─────────────────────────────────────────────────────
inline bool ReadFile(const std::string &fn, size_t sz, void *buf, size_t bufSz) {
    if (!buf || bufSz < sz) return false;
    std::ifstream ifs(fn, std::ios::binary);
    if (!ifs) return false;
    ifs.read(reinterpret_cast<char *>(buf), static_cast<std::streamsize>(sz));
    return ifs.gcount() == static_cast<std::streamsize>(sz);
}
inline bool WriteFile(const std::string &fn, const void *buf, size_t sz) {
    if (!buf) return false;
    std::ofstream ofs(fn, std::ios::binary);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char *>(buf), static_cast<std::streamsize>(sz));
    return static_cast<bool>(ofs);
}

static double Mean(const std::vector<double> &v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
static double Median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1U) ? v[n / 2U] : 0.5 * (v[n / 2U - 1U] + v[n / 2U]);
}

// ── 命令行参数 ───────────────────────────────────────────────────
struct Options {
    uint32_t totalTokens = 128, headDim = 64, blockDim = 8;
    uint32_t warmup = 10, repeat = 50, rounds = 5;
    uint32_t tileSize = 0;  // 0 = auto-compute
};
static bool ParseUint(const char *t, uint32_t *v) {
    if (!t || !v) return false;
    char *e = nullptr;
    unsigned long x = std::strtoul(t, &e, 10);
    if (e == t || *e != '\0' || x == 0 || x > UINT32_MAX) return false;
    *v = static_cast<uint32_t>(x); return true;
}
static bool ParseOptions(int argc, char **argv, Options *o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") return false;
        if (i + 1 >= argc) return false;
        const char *v = argv[++i]; bool ok = false;
        if (a == "--tokens")        ok = ParseUint(v, &o->totalTokens);
        else if (a == "--head-dim") ok = ParseUint(v, &o->headDim);
        else if (a == "--block-dim") ok = ParseUint(v, &o->blockDim);
        else if (a == "--warmup")   ok = ParseUint(v, &o->warmup);
        else if (a == "--repeat")   ok = ParseUint(v, &o->repeat);
        else if (a == "--rounds")   ok = ParseUint(v, &o->rounds);
        else if (a == "--tile-size") ok = ParseUint(v, &o->tileSize);
        else return false;
        if (!ok) return false;
    }
    return true;
}

int32_t main(int argc, char **argv) {
    Options o;
    if (!ParseOptions(argc, argv, &o)) {
        std::printf("Usage: %s --tokens N --head-dim N --block-dim N "
                    "--warmup N --repeat N --rounds N [--tile-size N]\n", argv[0]);
        return argc > 1 ? -1 : 0;
    }

    uint32_t bd  = std::min(o.blockDim, o.totalTokens);
    uint32_t rpc = (o.totalTokens + bd - 1U) / bd;
    uint64_t nEl = static_cast<uint64_t>(o.totalTokens) * o.headDim;
    size_t xb    = static_cast<size_t>(nEl) * sizeof(float);

    // Auto-compute tileSize if not specified
    if (o.tileSize == 0) {
        o.tileSize = (o.headDim > 0)
            ? std::max(1u, 240u * 1024u / (10u * o.headDim * 4u))
            : 1u;
        if (o.tileSize > 64u) o.tileSize = 64u;
    }

    RoPeTiling th;
    th.totalTokens = o.totalTokens; th.headDim = o.headDim;
    th.coreNum = bd; th.rowsPerCore = rpc;
    th.seqLen = o.totalTokens; th.numHeads = 1;
    th.trigTokens = o.totalTokens; th.compactTrig = 0;
    th.tileSize = o.tileSize;

    std::printf("[INFO] RoPE universal cos/sin version\n");
    std::printf("[INFO] totalTokens=%u headDim=%u coreNum=%u rowsPerCore=%u tileSize=%u\n",
        o.totalTokens, o.headDim, bd, rpc, o.tileSize);
    std::printf("[INFO] warmup=%u repeat=%u rounds=%u\n",
        o.warmup, o.repeat, o.rounds);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xH = nullptr, *cH = nullptr, *sH = nullptr, *yH = nullptr, *tlH = nullptr;
    uint8_t *xD = nullptr, *cD = nullptr, *sD = nullptr, *yD = nullptr, *wsD = nullptr, *tlD = nullptr;

    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&xH), xb));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&cH), xb));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&sH), xb));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&yH), xb));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&tlH), sizeof(th)));

    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&xD), xb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&cD), xb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&sD), xb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&yD), xb, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&wsD), 32, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&tlD), sizeof(th), ACL_MEM_MALLOC_NORMAL_ONLY));

    if (!ReadFile("./input/input_x.bin", xb, xH, xb) ||
        !ReadFile("./input/input_cos.bin", xb, cH, xb) ||
        !ReadFile("./input/input_sin.bin", xb, sH, xb)) {
        std::fprintf(stderr, "[ERROR] input_x.bin / input_cos.bin / input_sin.bin not found\n");
        return -1;
    }
    std::memcpy(tlH, &th, sizeof(th));

    CHECK_ACL(aclrtMemcpy(xD, xb, xH, xb, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(cD, xb, cH, xb, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(sD, xb, sH, xb, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tlD, sizeof(th), tlH, sizeof(th), ACL_MEMCPY_HOST_TO_DEVICE));

    // Warmup
    for (uint32_t i = 0; i < o.warmup; ++i)
        ACLRT_LAUNCH_KERNEL(rope_baseline_kernel)(bd, stream, xD, cD, sD, yD, wsD, tlD);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // Benchmark
    std::vector<double> du, wu; du.reserve(o.rounds); wu.reserve(o.rounds);
    for (uint32_t r = 0; r < o.rounds; ++r) {
        aclrtEvent sE = nullptr, eE = nullptr;
        CHECK_ACL(aclrtCreateEvent(&sE)); CHECK_ACL(aclrtCreateEvent(&eE));
        auto ws = std::chrono::steady_clock::now();
        CHECK_ACL(aclrtRecordEvent(sE, stream));
        for (uint32_t i = 0; i < o.repeat; ++i)
            ACLRT_LAUNCH_KERNEL(rope_baseline_kernel)(bd, stream, xD, cD, sD, yD, wsD, tlD);
        CHECK_ACL(aclrtRecordEvent(eE, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto we = std::chrono::steady_clock::now();
        float ms = 0; CHECK_ACL(aclrtEventElapsedTime(&ms, sE, eE));
        CHECK_ACL(aclrtDestroyEvent(sE)); CHECK_ACL(aclrtDestroyEvent(eE));
        double dUs = static_cast<double>(ms) * 1000.0 / o.repeat;
        double wUs = std::chrono::duration<double, std::micro>(we - ws).count() / o.repeat;
        du.push_back(dUs); wu.push_back(wUs);
        std::printf("[BENCH] round=%u device=%.3f us wall=%.3f us/launch\n",
            r + 1U, dUs, wUs);
    }
    auto mmD = std::minmax_element(du.begin(), du.end());
    std::printf("[RESULT] device mean=%.3f median=%.3f min=%.3f max=%.3f us\n",
        Mean(du), Median(du), *mmD.first, *mmD.second);

    CHECK_ACL(aclrtMemcpy(yH, xb, yD, xb, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_y.bin", yH, xb);
    std::printf("[OK] output/output_y.bin\n");

    CHECK_ACL(aclrtFree(xD)); CHECK_ACL(aclrtFree(cD)); CHECK_ACL(aclrtFree(sD));
    CHECK_ACL(aclrtFree(yD)); CHECK_ACL(aclrtFree(wsD)); CHECK_ACL(aclrtFree(tlD));
    CHECK_ACL(aclrtFreeHost(xH)); CHECK_ACL(aclrtFreeHost(cH));
    CHECK_ACL(aclrtFreeHost(sH)); CHECK_ACL(aclrtFreeHost(yH)); CHECK_ACL(aclrtFreeHost(tlH));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0)); CHECK_ACL(aclFinalize());
    std::printf("[OK] finished.\n"); return 0;
}
