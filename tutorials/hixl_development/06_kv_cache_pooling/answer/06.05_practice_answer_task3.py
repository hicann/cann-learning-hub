# 任务三参考答案：多轮对话 KV Cache 池化复用
# 需昇腾 NPU + Mooncake 环境

from mooncake.store import MooncakeDistributedStore
import torch
import torch_npu
torch.npu.set_device(0)

# ========== 1. 初始化 Store ==========
store = MooncakeDistributedStore()

store.setup(
    local_hostname="127.0.0.1",
    metadata_server="http://127.0.0.1:8080",
    global_segment_size=1024 * 1024 * 1024,
    local_buffer_size=256 * 1024 * 1024,
    protocol="ascend",
    rdma_devices="",
    master_server_addr="127.0.0.1:50051"
)

ROUND_SIZE_MB = 4

def make_kv_cache(size_mb=ROUND_SIZE_MB):
    return torch.ones(size_mb * 512 * 1024, dtype=torch.float16, device="npu")


# ========== 2. 第 1 轮：写入初始 KV Cache ==========
print("=== Round 1: Store initial KV Cache ===")
kv_r1 = make_kv_cache()

store.register_buffer(kv_r1.data_ptr(), kv_r1.nbytes)
put_ret = store.batch_put_from(
    ["dialogue_r1"],
    [kv_r1.data_ptr()],
    [kv_r1.nbytes]
)
assert put_ret[0] == 0, f"R1 put failed: {put_ret[0]}"

print(f"  R1: stored {kv_r1.nbytes / 1024 / 1024:.1f} MB to key 'dialogue_r1'")


# ========== 3. 第 2 轮：读取 R1，追加新数据，更新存储 ==========
print("\n=== Round 2: Retrieve R1, extend and update ===")

recv_r1 = torch.zeros_like(kv_r1)
store.register_buffer(recv_r1.data_ptr(), recv_r1.nbytes)
get_ret = store.batch_get_into(
    ["dialogue_r1"],
    [recv_r1.data_ptr()],
    [recv_r1.nbytes]
)
assert get_ret[0] > 0, f"R2 get failed: {get_ret[0]}"

kv_r2_new = make_kv_cache()
kv_combined = torch.cat([recv_r1, kv_r2_new])

store.register_buffer(kv_combined.data_ptr(), kv_combined.nbytes)
put_ret2 = store.batch_put_from(
    ["dialogue_r2"],
    [kv_combined.data_ptr()],
    [kv_combined.nbytes]
)
assert put_ret2[0] == 0, f"R2 put failed: {put_ret2[0]}"

print(f"  R2: retrieved R1 ({recv_r1.nbytes / 1024 / 1024:.1f} MB), "
      f"extended with new data ({kv_r2_new.nbytes / 1024 / 1024:.1f} MB), "
      f"stored combined ({kv_combined.nbytes / 1024 / 1024:.1f} MB) to 'dialogue_r2'")


# ========== 4. 第 3 轮：读取并验证完整性 ==========
print("\n=== Round 3: Verify cache integrity ===")

expected_size = kv_combined.nbytes
recv_r2 = torch.zeros(expected_size // 2, dtype=torch.float16, device="npu")

store.register_buffer(recv_r2.data_ptr(), recv_r2.nbytes)
get_ret2 = store.batch_get_into(
    ["dialogue_r2"],
    [recv_r2.data_ptr()],
    [recv_r2.nbytes]
)

cache_hit = get_ret2[0] > 0
print(f"  R3: retrieved 'dialogue_r2' ({recv_r2.nbytes / 1024 / 1024:.1f} MB)")
print(f"  Cache hit: {'✅ YES' if cache_hit else '❌ NO'}")

if torch.equal(recv_r2, kv_combined):
    print("  Data integrity: ✅ PASS — R3 数据与 R2 写入数据一致")
else:
    print("  Data integrity: ❌ FAIL — 数据不一致")


# ========== 5. 清理 ==========
store.close()
print("\n✅ Multi-round dialogue cache simulation completed!")
