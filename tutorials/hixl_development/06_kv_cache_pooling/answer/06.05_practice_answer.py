# 综合编程实践·任务一参考答案：KV Cache 大小计算与缓存命中分析

def calc_kv_cache(num_layers, batch_size, seq_len, num_heads, head_dim, dtype_size=2):
    """计算 KV Cache 大小（GB）"""
    bytes_total = num_layers * 2 * batch_size * seq_len * num_heads * head_dim * dtype_size
    return bytes_total / (1024**3)


# ========== 测试：不同模型的 KV Cache 大小 ==========
models = [
    ("LLaMA-7B",  32, 32, 4096, 32, 128),
    ("LLaMA-13B", 40, 32, 4096, 40, 128),
    ("LLaMA-65B", 80, 32, 4096, 64, 128),
]

print(f"{'模型':<15} {'KV Cache (FP16)':>16} {'单卡64GB够？':>12}")
print("-" * 48)
for name, layers, batch, seq, heads, dim in models:
    kv_gb = calc_kv_cache(layers, batch, seq, heads, dim)
    fits = "够" if kv_gb < 64 else "不够"
    print(f"{name:<15} {kv_gb:>13.1f} GB {fits:>12}")


# ========== 多轮对话缓存命中分析 ==========
print("\n" + "=" * 48)
print("多轮对话缓存命中分析")
print("=" * 48)

dialogue = [
    ("第1轮", "什么是昇腾", 5),
    ("第2轮", "什么是昇腾，和英伟达比呢", 12),
    ("第3轮", "什么是昇腾，和英伟达比呢，支持哪些模型", 20),
]

for i in range(len(dialogue)):
    label, prompt, total_tokens = dialogue[i]
    if i == 0:
        hit_tokens = 0
        compute_tokens = total_tokens
    else:
        prev_tokens = dialogue[i - 1][2]
        hit_tokens = prev_tokens
        compute_tokens = total_tokens - hit_tokens

    hit_rate = (hit_tokens / total_tokens * 100) if total_tokens > 0 else 0
    print(f"{label}: 总 {total_tokens} token, 命中 {hit_tokens}, 新算 {compute_tokens}, 命中率 {hit_rate:.0f}%")
