# 任务三参考答案：Push 模式 KV Cache 传输实现
# 需昇腾 NPU + llm-datadist 环境

# ============ push_prefill.py ============
"""
from llm_datadist import (
    LLMDataDist, LLMRole, LLMConfig, CacheDesc,
    Placement, DataType, BlocksCacheKey, RegisterMemStatus,
    LLMStatusCode
)
import torch
import torch_npu
import time

PREFILL_CLUSTER_ID = 0
num_layers = 2
batch_size = 1
num_heads = 8
seq_len = 128
head_dim = 128

# TODO-1: 初始化 LLMDataDist（Prefill 角色）
datadist = LLMDataDist(LLMRole.PROMPT, PREFILL_CLUSTER_ID)
llm_config = LLMConfig()
llm_config.device_id = 0
llm_config.enable_cache_manager = True
options = llm_config.generate_options()
datadist.init(options)
cache_manager = datadist.get_cache_manager()

# TODO-2: 注册 KV Cache 内存
kv_cache = torch.zeros(num_layers, 2, batch_size, num_heads, seq_len, head_dim,
                      dtype=torch.float16, device="npu:0")
cache_key = BlocksCacheKey(0, "push_cache_key")
cache_manager.register_blocks_cache(cache_key, kv_cache.data_ptr(), kv_cache.nbytes)

# TODO-3: 建立单向链接到 Decode 节点
datadist.link_clusters([1], timeout=5000)

# TODO-4: 执行 Push 传输
cache_desc = CacheDesc()
cache_desc.num_tensors = num_layers * 2
cache_desc.shape = [batch_size, num_heads, seq_len, head_dim]
cache_desc.dtype = DataType.DT_FP16
cache_desc.placement = Placement.DEVICE
cache = cache_manager.allocate_cache(cache_desc)

# 等待链接就绪后执行 Push
time.sleep(2)
cache_manager.push_blocks(cache_key, [1], cache, timeout=10000)
print("Push transfer completed!")

# TODO-5: 资源释放
datadist.unlink_clusters([1], timeout=5000, force=True)
cache_manager.unregister_cache(cache_key)
cache_manager.deallocate_cache(cache)
datadist.finalize()
print("Prefill push completed!")
"""

# ============ push_decode.py ============
"""
from llm_datadist import (
    LLMDataDist, LLMRole, LLMConfig, CacheDesc,
    Placement, DataType, BlocksCacheKey, RegisterMemStatus,
    LLMStatusCode
)
import torch
import torch_npu
import time

DECODE_CLUSTER_ID = 1
PREFILL_CLUSTER_ID = 0
num_layers = 2
batch_size = 1
num_heads = 8
seq_len = 128
head_dim = 128

# TODO-1: 初始化 LLMDataDist（Decode 角色）
datadist = LLMDataDist(LLMRole.DECODER, DECODE_CLUSTER_ID)
llm_config = LLMConfig()
llm_config.device_id = 1
llm_config.enable_cache_manager = True
llm_config.transfer_backend = "hixl"
options = llm_config.generate_options()
datadist.init(options)
cache_manager = datadist.get_cache_manager()

# TODO-2: 注册 KV Cache 内存（用于接收 Push 数据）
kv_cache = torch.zeros(num_layers, 2, batch_size, num_heads, seq_len, head_dim,
                      dtype=torch.float16, device="npu:1")
cache_key = BlocksCacheKey(0, "push_cache_key")
cache_manager.register_blocks_cache(cache_key, kv_cache.data_ptr(), kv_cache.nbytes)

# TODO-3: 接收 Push 数据
# Push 模式下 Decode 端被动接收，需确保 cache 已注册并等待传输完成
datadist.link_clusters([0], timeout=5000)

cache_desc = CacheDesc()
cache_desc.num_tensors = num_layers * 2
cache_desc.shape = [batch_size, num_heads, seq_len, head_dim]
cache_desc.dtype = DataType.DT_FP16
cache_desc.placement = Placement.DEVICE
cache = cache_manager.allocate_cache(cache_desc)

# 等待 Prefill 端 Push 完成
time.sleep(5)
print("Decode received push data!")

# TODO-4: 资源释放
datadist.unlink_clusters([0], timeout=5000, force=True)
cache_manager.unregister_cache(cache_key)
cache_manager.deallocate_cache(cache)
datadist.finalize()
print("Decode cleanup completed!")
"""
