// Reuse the identical ACL benchmark/reference harness so only kernel implementation differs.
#include "aclrtlaunch_gqa_attention_optimized_kernel.h"
#include "gqa_attention_tiling.h"
#define GQA_STANDALONE_CUSTOM_LAUNCH_HEADER
#define GQA_ATTENTION_BASELINE_TILING_H
#define GqaAttentionBaselineTiling GqaAttentionOptimizedTiling
#define gqa_attention_baseline_kernel gqa_attention_optimized_kernel
#define GQA_STANDALONE_LABEL "optimized"
#undef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_gqa_attention_optimized_kernel
#include "../../GqaAttentionBaselineExperiment/verify_kernel_launch/gqa_attention_baseline_standalone.cpp"
