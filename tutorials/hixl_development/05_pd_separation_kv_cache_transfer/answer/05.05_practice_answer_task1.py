# 任务一参考答案：KV Cache 传输可行性分析

# 模型配置
models = [
    {"name": "LLaMA-7B",  "num_layers": 32, "num_heads": 32, "head_dim": 128, "dtype_size": 2},
    {"name": "LLaMA-65B", "num_layers": 80, "num_heads": 64, "head_dim": 128, "dtype_size": 2},
    {"name": "Qwen-72B",  "num_layers": 80, "num_heads": 64, "head_dim": 128, "dtype_size": 2},
]

# 链路配置
links = [
    {"name": "HCCS",      "bandwidth_gbps": 56},
    {"name": "RDMA",      "bandwidth_gbps": 100},
    {"name": "FabricMem", "bandwidth_gbps": 119},
]

batch_size = 8
seq_len = 2048
tpot_ms = 20


def calc_kv_cache_size_gb(num_layers, batch_size, seq_len, num_heads, head_dim, dtype_size):
    size_bytes = num_layers * 2 * batch_size * seq_len * num_heads * head_dim * dtype_size
    return size_bytes / (1024 ** 3)


def calc_transfer_time_sec(size_gb, bandwidth_gbps):
    return size_gb / bandwidth_gbps


def check_feasibility(transfer_time_sec, tpot_ms, seq_len):
    threshold_sec = tpot_ms * seq_len / 1000
    return (transfer_time_sec < threshold_sec, threshold_sec)


if __name__ == "__main__":
    print("=" * 70)
    print("KV Cache 传输可行性分析")
    print("=" * 70)

    for model in models:
        cache_size = calc_kv_cache_size_gb(
            model["num_layers"], batch_size, seq_len,
            model["num_heads"], model["head_dim"], model["dtype_size"]
        )
        print(f"\n模型: {model['name']}")
        print(f"  配置: layers={model['num_layers']}, heads={model['num_heads']}, "
              f"head_dim={model['head_dim']}, batch={batch_size}, seq_len={seq_len}")
        print(f"  KV Cache 大小: {cache_size:.2f} GB")

        for link in links:
            transfer_time = calc_transfer_time_sec(cache_size, link["bandwidth_gbps"])
            feasible, threshold = check_feasibility(transfer_time, tpot_ms, seq_len)
            status = "✅ 可行" if feasible else "❌ 不可行"
            print(f"    {link['name']:12s} | 带宽 {link['bandwidth_gbps']:3d} GB/s | "
                  f"传输时间 {transfer_time:.4f}s | 阈值 {threshold:.4f}s | {status}")
