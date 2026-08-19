#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "acl/acl.h"
#ifndef GQA_STANDALONE_CUSTOM_LAUNCH_HEADER
#include "aclrtlaunch_gqa_attention_baseline_kernel.h"
#endif
#include "gqa_attention_tiling.h"

#ifndef GQA_STANDALONE_LABEL
#define GQA_STANDALONE_LABEL "baseline"
#endif

#define CHECK_ACL(call) do { const aclError ret = (call); if (ret != ACL_SUCCESS) { std::fprintf(stderr, "[ERROR] %s ret=%d\n", #call, static_cast<int>(ret)); return 1; } } while (0)

struct Options { uint32_t batch = 1, queryHeads = 8, kvHeads = 2, queryLen = 32, keyLen = 32, headDim = 64, blockDim = 8, warmup = 10, repeat = 50, rounds = 5; bool causal = true; };
static bool ParseUint(const char *text, uint32_t *value) { char *end = nullptr; const unsigned long v = std::strtoul(text, &end, 10); if (end == text || *end || v == 0 || v > UINT32_MAX) return false; *value = static_cast<uint32_t>(v); return true; }
static bool Parse(Options *o, int argc, char **argv) {
    for (int i = 1; i < argc; ++i) { const std::string key(argv[i]); if (key == "--help" || i + 1 >= argc) return false; const char *v = argv[++i]; bool ok = false;
        if (key == "--batch") ok = ParseUint(v, &o->batch); else if (key == "--q-heads") ok = ParseUint(v, &o->queryHeads); else if (key == "--kv-heads") ok = ParseUint(v, &o->kvHeads);
        else if (key == "--q-len") ok = ParseUint(v, &o->queryLen); else if (key == "--kv-len") ok = ParseUint(v, &o->keyLen); else if (key == "--head-dim") ok = ParseUint(v, &o->headDim);
        else if (key == "--block-dim") ok = ParseUint(v, &o->blockDim); else if (key == "--warmup") ok = ParseUint(v, &o->warmup); else if (key == "--repeat") ok = ParseUint(v, &o->repeat);
        else if (key == "--rounds") ok = ParseUint(v, &o->rounds); else if (key == "--causal") { o->causal = std::strtoul(v, nullptr, 10) != 0; ok = true; }
        if (!ok) return false;
    } return o->queryHeads % o->kvHeads == 0;
}
static void Reference(const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v, std::vector<float> *out, const Options &o) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(o.headDim));
    for (uint32_t b = 0; b < o.batch; ++b) for (uint32_t h = 0; h < o.queryHeads; ++h) for (uint32_t p = 0; p < o.queryLen; ++p) {
        const uint32_t kvh = h / (o.queryHeads / o.kvHeads); const uint32_t qbase = ((b * o.queryHeads + h) * o.queryLen + p) * o.headDim;
        const uint32_t kvbase = (b * o.kvHeads + kvh) * o.keyLen * o.headDim; const int32_t causalVisible = static_cast<int32_t>(p) + static_cast<int32_t>(o.keyLen) - static_cast<int32_t>(o.queryLen) + 1; const uint32_t visible = o.causal ? (causalVisible <= 0 ? 0 : std::min(o.keyLen, static_cast<uint32_t>(causalVisible))) : o.keyLen;
        if (visible == 0) { for (uint32_t d = 0; d < o.headDim; ++d) (*out)[qbase + d] = 0.0f; continue; }
        std::vector<float> scores(visible); float maxScore = -INFINITY;
        for (uint32_t i = 0; i < visible; ++i) { float dot = 0; for (uint32_t d = 0; d < o.headDim; ++d) dot += q[qbase+d] * k[kvbase+i*o.headDim+d]; scores[i] = dot * scale; maxScore = std::max(maxScore, scores[i]); }
        float sum = 0; for (float &x : scores) { x = std::exp(x - maxScore); sum += x; }
        for (uint32_t d = 0; d < o.headDim; ++d) { float x = 0; for (uint32_t i = 0; i < visible; ++i) x += scores[i] * v[kvbase+i*o.headDim+d]; (*out)[qbase+d] = x / sum; }
    }
}
int main(int argc, char **argv) {
    Options o; if (!Parse(&o, argc, argv)) { std::printf("Usage: %s [--batch N --q-heads N --kv-heads N --q-len N --kv-len N --head-dim N --block-dim N --warmup N --repeat N --rounds N --causal 0|1]\n", argv[0]); return argc > 1; }
    const uint32_t qSize = o.batch*o.queryHeads*o.queryLen*o.headDim, kvSize = o.batch*o.kvHeads*o.keyLen*o.headDim, totalQueries = o.batch*o.queryHeads*o.queryLen;
    const uint32_t blocks = std::min(o.blockDim, totalQueries); GqaAttentionBaselineTiling tiling; tiling.batch=o.batch; tiling.queryHeads=o.queryHeads; tiling.kvHeads=o.kvHeads; tiling.queryLen=o.queryLen; tiling.keyLen=o.keyLen; tiling.headDim=o.headDim; tiling.totalQueries=totalQueries; tiling.coreNum=blocks; tiling.queriesPerCore=(totalQueries+blocks-1)/blocks; tiling.causal=o.causal; tiling.scale=1.0f/std::sqrt(static_cast<float>(o.headDim));
    std::mt19937 rng(42); std::normal_distribution<float> dist(0, 1); std::vector<float> q(qSize), k(kvSize), v(kvSize), golden(qSize), output(qSize); for (float &x:q) x=dist(rng); for(float &x:k)x=dist(rng); for(float &x:v)x=dist(rng); Reference(q,k,v,&golden,o);
    std::printf("[INFO] GQA %s B=%u Hq=%u Hkv=%u Sq=%u Sk=%u D=%u causal=%u blockDim=%u\n", GQA_STANDALONE_LABEL, o.batch,o.queryHeads,o.kvHeads,o.queryLen,o.keyLen,o.headDim,o.causal,blocks);
    CHECK_ACL(aclInit(nullptr)); CHECK_ACL(aclrtSetDevice(0)); aclrtStream stream=nullptr; CHECK_ACL(aclrtCreateStream(&stream)); void *qd=nullptr,*kd=nullptr,*vd=nullptr,*od=nullptr,*ws=nullptr,*td=nullptr; const size_t qb=qSize*sizeof(float), kvb=kvSize*sizeof(float);
    CHECK_ACL(aclrtMalloc(&qd,qb,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK_ACL(aclrtMalloc(&kd,kvb,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK_ACL(aclrtMalloc(&vd,kvb,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK_ACL(aclrtMalloc(&od,qb,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK_ACL(aclrtMalloc(&ws,32,ACL_MEM_MALLOC_NORMAL_ONLY)); CHECK_ACL(aclrtMalloc(&td,sizeof(tiling),ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMemcpy(qd,qb,q.data(),qb,ACL_MEMCPY_HOST_TO_DEVICE)); CHECK_ACL(aclrtMemcpy(kd,kvb,k.data(),kvb,ACL_MEMCPY_HOST_TO_DEVICE)); CHECK_ACL(aclrtMemcpy(vd,kvb,v.data(),kvb,ACL_MEMCPY_HOST_TO_DEVICE)); CHECK_ACL(aclrtMemcpy(td,sizeof(tiling),&tiling,sizeof(tiling),ACL_MEMCPY_HOST_TO_DEVICE));
    for (uint32_t i = 0; i < o.warmup; ++i) {
        ACLRT_LAUNCH_KERNEL(gqa_attention_baseline_kernel)(blocks, stream, qd, kd, vd, od, ws, td);
    }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    std::vector<double> times;
    for(uint32_t r=0;r<o.rounds;++r){ aclrtEvent start=nullptr,stop=nullptr; CHECK_ACL(aclrtCreateEvent(&start)); CHECK_ACL(aclrtCreateEvent(&stop)); CHECK_ACL(aclrtRecordEvent(start,stream)); for(uint32_t i=0;i<o.repeat;++i) ACLRT_LAUNCH_KERNEL(gqa_attention_baseline_kernel)(blocks,stream,qd,kd,vd,od,ws,td); CHECK_ACL(aclrtRecordEvent(stop,stream)); CHECK_ACL(aclrtSynchronizeStream(stream)); float ms=0; CHECK_ACL(aclrtEventElapsedTime(&ms,start,stop)); CHECK_ACL(aclrtDestroyEvent(start)); CHECK_ACL(aclrtDestroyEvent(stop)); times.push_back(ms*1000.0/o.repeat); std::printf("[BENCH] round=%u device=%.3f us\n",r+1,times.back()); }
    CHECK_ACL(aclrtMemcpy(output.data(),qb,od,qb,ACL_MEMCPY_DEVICE_TO_HOST)); double maxAbs=0,meanAbs=0; for(uint32_t i=0;i<qSize;++i){const double diff=std::abs(output[i]-golden[i]);maxAbs=std::max(maxAbs,diff);meanAbs+=diff;} meanAbs/=qSize; std::sort(times.begin(),times.end()); std::printf("[RESULT] mean=%.3f median=%.3f min=%.3f max=%.3f us\n",std::accumulate(times.begin(),times.end(),0.0)/times.size(),times[times.size()/2],times.front(),times.back()); std::printf("[CHECK] max_abs_diff=%.8e mean_abs_diff=%.8e\n",maxAbs,meanAbs); std::printf("[%s] correctness threshold atol=3e-3\n",maxAbs<=3e-3?"PASS":"FAIL");
    CHECK_ACL(aclrtFree(qd));CHECK_ACL(aclrtFree(kd));CHECK_ACL(aclrtFree(vd));CHECK_ACL(aclrtFree(od));CHECK_ACL(aclrtFree(ws));CHECK_ACL(aclrtFree(td));CHECK_ACL(aclrtDestroyStream(stream));CHECK_ACL(aclrtResetDevice(0));CHECK_ACL(aclFinalize()); return maxAbs<=3e-3?0:1;
}
