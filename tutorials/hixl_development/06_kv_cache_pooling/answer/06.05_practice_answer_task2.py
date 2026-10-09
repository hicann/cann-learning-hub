# 综合编程实践·任务二参考答案：Mooncake Store 批量读写与部分失败处理
# 需昇腾 NPU + Mooncake 环境，需先启动 mooncake_master

from mooncake.store import MooncakeDistributedStore
import torch
import torch_npu

torch.npu.set_device(0)  # 逻辑设备号

# ========== 1. 初始化 Store ==========
store = MooncakeDistributedStore()

store.setup(
    local_hostname="127.0.0.1",
    metadata_server="http://127.0.0.1:8080/metadata",
    global_segment_size=1024 * 1024 * 1024,  # 1 GB：kv_2(1.5GB) 超过配额，
                                             # 写入必然失败（返回负数错误码）
    local_buffer_size=0,
    protocol="ascend",
    rdma_devices="",
    master_server_addr="127.0.0.1:50051"
)

# ========== 2. 准备 3 份不同大小的 KV 数据 ==========
buf_0 = torch.ones(512 * 1024, dtype=torch.float16, device="npu")          # 1MB
buf_1 = torch.ones(2 * 1024 * 1024, dtype=torch.float16, device="npu")     # 4MB
buf_2 = torch.ones(768 * 1024 * 1024, dtype=torch.float16, device="npu")   # 1.5GB：超过
                                                                             # global_segment_size(1GB)
                                                                             # 配额，写入必然失败

keys = ["batch_kv_0", "batch_kv_1", "batch_kv_2"]
buffers = [buf_0, buf_1, buf_2]

for b in buffers:
    store.register_buffer(b.data_ptr(), b.nbytes)

# ========== 3. 批量写入（D2H）并逐个检查返回值 ==========
results = store.batch_put_from(keys, [b.data_ptr() for b in buffers], [b.nbytes for b in buffers])

ok_keys = []
ok_buffers = []
for i, ret in enumerate(results):
    if ret == 0:  # put：0 表示写入成功
        ok_keys.append(keys[i])
        ok_buffers.append(buffers[i])
    print(f"  put {keys[i]}: {'OK' if ret == 0 else f'FAIL(code={ret})'}")

print(f"\n写入成功 {len(ok_keys)}/{len(keys)} 个 key")

# ========== 4. 只读取写入成功的 key（H2D） ==========
dst_buffers = [torch.zeros_like(b) for b in ok_buffers]

for d in dst_buffers:
    store.register_buffer(d.data_ptr(), d.nbytes)

get_results = store.batch_get_into(
    ok_keys,
    [d.data_ptr() for d in dst_buffers],
    [d.nbytes for d in dst_buffers],
)

# ========== 5. 逐个验证并汇总 ==========
# get 返回值语义：正数=成功读取的字节数，负数=错误码
for i, ret in enumerate(get_results):
    if ret > 0:
        match = torch.equal(ok_buffers[i], dst_buffers[i])
        print(f"  get {ok_keys[i]}: OK ({ret} bytes), match={'PASS' if match else 'FAIL'}")
    else:
        print(f"  get {ok_keys[i]}: FAIL(code={ret})")

store.close()
