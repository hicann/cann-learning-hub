# HCCL + Ascend C 分布式 SpMV 实验手册

## 1. 当前后端事实

本工程使用 ACL+HCCL Device Buffer 完成 Broadcast/AllGather，并由每个 Rank 的 Ascend C RTC Kernel 计算 local CSR SpMV。流程为：Host 输入 → H2D → HCCL Broadcast(Device) → Ascend C local SpMV(Device) → HCCL AllGather(Device) → D2H → CPU reference。正式结果必须同时出现 `Actual Compute Backend=Ascend C FP32 RTC` 与 `Communication Backend=ACL+HCCL`。

## 2. 查询并设置通信地址

使用 rank table 前先查询本机服务端 IP 与各 Device 的 HCCN IP：

```bash
hostname -I
for d in 0 1 2 3; do hccn_tool -i "$d" -ip -g; done
export SERVER_IP="<本机实际服务端IP>"
export DEVICE_IPS_2="<device0_ip> <device1_ip>"
export DEVICE_IPS_4="<device0_ip> <device1_ip> <device2_ip> <device3_ip>"
```

若单机环境没有 `hccn_tool`，不要设置上述三个变量，运行脚本会使用 root-info 初始化。变量设置不完整时 Notebook 会忽略它们并使用 root-info。

## 3. 编译与运行

```bash
cd src/hccl_spmv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHCCL_SPMV_STUB=OFF
cmake --build build -j
python3 scripts/generate_rank_table.py \
  --server-ip <管理网或服务端实际IP> \
  --device-ip <HCCN_DEVICE_IP_0> <HCCN_DEVICE_IP_1> \
  --output rank_table_2p.json
bash scripts/run_scaling.sh --rank-table rank_table_2p.json \
  --npus-list 2 --matrix U1 --warmup 3 --repeat 10
```

配置后必须确认 CMake 输出 `HCCL backend=1`；正式模式缺少依赖会直接失败。只有显式 `-DHCCL_SPMV_STUB=ON` 才能生成 Host Stub，且不能作为 HCCL 实测。2/4 卡分别使用各自匹配的 rank table（`rank_table_2p.json`/`rank_table_4p.json`），不要用一个 4-rank JSON 配合 `--npus-list 2,4`，也可不传 rank table 走单机 root-info fallback；并保持矩阵、预热和重复次数一致。以单 Rank CPU SpMV 为 reference，使用程序的最大相对误差和 `1e-6` 阈值验证结果。

## 4. 记录与分析

每个 Rank 记录 Rank ID、Device ID、World Size；将 compute、HCCL communication、transfer、total time 分开。不得把历史 CSV 当作本次实测。

| Ranks/Devices | Compute backend | Comm backend | Compute (ms) | HCCL (ms) | Transfer (ms) | Total (ms) | Error |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Ascend C RTC | | | | | | |
| 2 | Ascend C RTC | ACL+HCCL | | | | | |
| 4 | Ascend C RTC | ACL+HCCL | | | | | |

练习：解释为什么增加 NPU 数量仍可能因 HCCL、同步和负载不均衡而无法线性扩展。
