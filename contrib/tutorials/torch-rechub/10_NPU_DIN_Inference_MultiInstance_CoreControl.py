# -*- coding: utf-8 -*-
"""单文件 PyTorch DIN 昇腾 NPU 多进程控核推理示例。

模型结构、固定演示权重和输入数据均内置，不读取 checkpoint、模型定义或输入文件。
运行环境需预装 CANN、PyTorch、torch_npu、TorchAir 和 NumPy。

架构说明：
- 多进程（非多线程）并行：每个 worker 进程拥有独立的 NPU context 与默认 stream。
- torchair 编译图（torch.compile + npu backend）在 worker 进程内编译并绑定到
  该进程的默认 stream，不存在跨 stream 执行问题。
- torch.npu.set_device_limit 在每个 worker 进程内调用，提供设备级控核。
- 使用 spawn 启动方法确保子进程不继承父进程的 NPU 运行时状态。
- 支持 autofuse 算子融合：通过 AUTOFUSE_FLAGS 环境变量启用 reduce/concat 融合。
- 支持 1000 请求基准测试：warmup 后处理指定数量请求并统计 latency。
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
except ImportError as exc:
    raise RuntimeError(
        "请在已安装 CANN、PyTorch、torch_npu 与 TorchAir 的昇腾环境运行。"
    ) from exc


# -------------------- 可调运行参数 --------------------
DEVICE_ID = 0
NUM_INSTANCES = 4
BATCH_SIZE = 4
WARMUP_STEPS = 3
OP_AICORE_NUM = 7
OP_VECTORCORE_NUM = 14
# 基准测试请求数：warmup 后处理的总请求数（按 batch 切分）。
NUM_BENCHMARK_REQUESTS = 1000
# autofuse 配置：启用 reduce 与 concat 算子自动融合。
# 该环境变量需在 NPU/torchair 初始化前设置，由 GE 编译器读取。
AUTOFUSE_FLAGS = "--enable_autofuse=true;--autofuse_enable_pass=reduce,concat"
OUTPUT_DIR = Path("./embedded_pytorch_din_outputs")

# -------------------- 内置 DIN 规格 --------------------
DIN_SPEC = {
    "n_users": 8,
    "n_items": 32,
    "n_cates": 12,
    "seq_max_len": 8,
    "embed_dim": 16,
    "attention_dims": [64, 32],
    "prediction_dims": [64, 32],
}

class EmbeddedDIN(torch.nn.Module):
    """不依赖 torch-rechub 的紧凑 DIN 推理模型。"""

    def __init__(self, spec: dict):
        super().__init__()
        dim = int(spec["embed_dim"])
        self.user_embedding = torch.nn.Embedding(int(spec["n_users"]) + 1, dim)
        self.item_embedding = torch.nn.Embedding(
            int(spec["n_items"]) + 1, dim, padding_idx=0
        )
        self.cate_embedding = torch.nn.Embedding(
            int(spec["n_cates"]) + 1, dim, padding_idx=0
        )
        self.attention = self._mlp(dim * 8, spec["attention_dims"], 1)
        self.predictor = self._mlp(dim * 5, spec["prediction_dims"], 1)

    @staticmethod
    def _mlp(input_dim: int, hidden_dims: List[int], output_dim: int):
        layers: List[torch.nn.Module] = []
        current = input_dim
        for width in hidden_dims:
            layers.extend((torch.nn.Linear(current, int(width)), torch.nn.ReLU()))
            current = int(width)
        layers.append(torch.nn.Linear(current, output_dim))
        return torch.nn.Sequential(*layers)

    def forward(self, user_id, target_item, target_cate, history_item, history_cate):
        user = self.user_embedding(user_id)
        target = torch.cat(
            (self.item_embedding(target_item), self.cate_embedding(target_cate)), dim=-1
        )
        history = torch.cat(
            (self.item_embedding(history_item), self.cate_embedding(history_cate)), dim=-1
        )
        query = target.unsqueeze(1).expand_as(history)
        attention_features = torch.cat(
            (query, history, query - history, query * history), dim=-1
        )
        attention_logits = self.attention(attention_features).squeeze(-1)
        mask = history_item.ne(0)
        attention_logits = attention_logits.masked_fill(~mask, -1.0e9)
        attention_scores = torch.softmax(attention_logits, dim=1)
        interest = torch.sum(history * attention_scores.unsqueeze(-1), dim=1)
        prediction_features = torch.cat((user, target, interest), dim=-1)
        return torch.sigmoid(self.predictor(prediction_features)).squeeze(-1)


def install_embedded_demo_weights(model: torch.nn.Module) -> None:
    """用脚本内固定公式写入演示权重；不读取 checkpoint，也不进行训练。"""
    with torch.no_grad():
        for parameter_index, parameter in enumerate(model.parameters()):
            position = torch.arange(parameter.numel(), dtype=torch.float32)
            values = 0.035 * torch.sin(position * 0.017 + parameter_index * 0.31)
            parameter.copy_(values.reshape(parameter.shape))
        model.item_embedding.weight[0].zero_()
        model.cate_embedding.weight[0].zero_()


@dataclass
class Batch:
    batch_id: int
    valid_size: int
    arrays: Tuple[np.ndarray, ...]


def generate_single_random_batch(
    batch_size: int, spec: dict, seed: int = 42
) -> Batch:
    """生成单个随机输入 batch，用于 warmup 与基准测试。

    使用随机值（非 zeros）以避免编译器/硬件对全零输入的特殊优化。
    该 batch 同时用于 warmup 和所有推理请求，确保 input 数据和 shape
    完全一致，消除输入变化对 latency 的干扰。

    Args:
        batch_size: batch 样本数
        spec: DIN 规格（决定 ID 范围与序列长度）
        seed: 随机种子，确保可复现

    Returns:
        单个 Batch，包含随机生成的输入数组
    """
    rng = np.random.default_rng(seed)
    n_users = int(spec["n_users"])
    n_items = int(spec["n_items"])
    n_cates = int(spec["n_cates"])
    seq_max_len = int(spec["seq_max_len"])

    # user_id: [1, n_users]
    user_ids = rng.integers(1, n_users + 1, size=batch_size, dtype=np.int64)
    # target_item: [1, n_items]
    target_items = rng.integers(1, n_items + 1, size=batch_size, dtype=np.int64)
    # target_cate: [1, n_cates]
    target_cates = rng.integers(1, n_cates + 1, size=batch_size, dtype=np.int64)
    # history_item/history_cate: [0, n_items]/[0, n_cates]，0 表示 padding
    history_items = rng.integers(0, n_items + 1, size=(batch_size, seq_max_len), dtype=np.int64)
    history_cates = rng.integers(0, n_cates + 1, size=(batch_size, seq_max_len), dtype=np.int64)
    arrays = (user_ids, target_items, target_cates, history_items, history_cates)
    return Batch(batch_id=0, valid_size=batch_size, arrays=arrays)


def replicate_batch(template: Batch, num_copies: int) -> List[Batch]:
    """将单个 batch 复制多份，用于基准测试。

    所有副本共享相同的输入数组（引用同一份数据），
    确保 warmup 与所有推理请求使用完全相同的 input。

    Args:
        template: 模板 batch
        num_copies: 复制份数

    Returns:
        batch 列表，每个 batch 的 arrays 指向同一份数据
    """
    return [
        Batch(batch_id=i, valid_size=template.valid_size, arrays=template.arrays)
        for i in range(num_copies)
    ]


# -------------------- 进程内全局缓存 --------------------
# 每个 worker 进程在首次任务时初始化一次，后续任务复用已编译图。
_WORKER_STATE: Dict[str, object] = {}
# 跨进程编译锁：序列化 TBE 算子编译，避免并发编译子进程冲突。
_COMPILE_LOCK: mp.Lock = None


def _get_worker_graph():
    """懒加载并缓存当前进程的编译图与设备设置。

    使用 _COMPILE_LOCK 序列化 torch.compile 调用：
    TBE（Tensor Boost Engine）在编译算子时会启动内部子进程，
    多个 worker 进程同时编译会导致 TBE 子进程管理冲突，
    报错 "TBE Subprocess[task_distribute] raise error, main process disappeared"。
    """
    if "graph" in _WORKER_STATE:
        return _WORKER_STATE["graph"]

    # 设置 autofuse 环境变量：必须在 NPU context 与 torchair 初始化前设置，
    # GE 编译器在首次编译时读取该变量以启用算子融合 pass。
    os.environ["AUTOFUSE_FLAGS"] = AUTOFUSE_FLAGS

    device = f"npu:{DEVICE_ID}"
    torch.npu.set_device(device)
    torch.set_grad_enabled(False)

    # 设备级控核：每个进程独立设置（进程间共享同一物理设备）。
    torch.npu.set_device_limit(
        DEVICE_ID, cube_num=OP_AICORE_NUM, vector_num=OP_VECTORCORE_NUM
    )

    compiler_config = CompilerConfig()
    # 不使用 "reduce-overhead"（NPU graph capture）模式：
    # 该模式会启用图捕获/重放，与多进程共享设备的同步语义存在冲突。
    # 默认模式提供算子融合、常量折叠等图优化，但不进行 graph replay。
    backend = torchair.get_npu_backend(compiler_config=compiler_config)

    model = EmbeddedDIN(DIN_SPEC)
    install_embedded_demo_weights(model)
    model.eval().to(device)

    # 序列化编译：同一时刻只有一个 worker 进程执行 torch.compile，
    # 避免 TBE 编译子进程并发冲突。
    if _COMPILE_LOCK is not None:
        _COMPILE_LOCK.acquire()
    try:
        graph = torch.compile(model, backend=backend, dynamic=False, fullgraph=True)
    finally:
        if _COMPILE_LOCK is not None:
            _COMPILE_LOCK.release()

    _WORKER_STATE["graph"] = graph
    _WORKER_STATE["device"] = device
    return graph


def _worker_init(compile_lock: mp.Lock, warmup_arrays: Tuple[np.ndarray, ...]) -> None:
    """Pool initializer：提前触发 NPU context 与编译图初始化。

    在进程启动时即完成设备设置、控核配置与图编译，避免首个任务承担编译开销。
    同时执行一次 dummy 推理，使编译图绑定到本进程默认 stream。
    编译过程通过 compile_lock 序列化，避免 TBE 子进程并发冲突。

    warmup 使用与正式推理相同的输入数据与 shape，确保编译产物与
    后续推理完全一致（torchair 编译图对输入 shape 敏感）。

    Args:
        compile_lock: 跨进程编译锁
        warmup_arrays: warmup 用的输入数组（来自首个 benchmark batch）
    """
    global _COMPILE_LOCK
    _COMPILE_LOCK = compile_lock

    graph = _get_worker_graph()
    device = _WORKER_STATE["device"]
    # 使用与正式推理相同的输入数据触发编译并绑定 stream，
    # 确保 shape 一致，避免编译图与后续推理 shape 不匹配。
    warmup_inputs = tuple(torch.from_numpy(arr).to(device) for arr in warmup_arrays)
    with torch.inference_mode():
        graph(*warmup_inputs)
    torch.npu.synchronize()


def _worker_run_batch(payload: Tuple[int, int, Tuple[np.ndarray, ...]]) -> Tuple[int, int, np.ndarray, float]:
    """在 worker 进程内执行单个 batch 推理。

    使用 torch.npu.Event 进行 NPU 侧计时，elapsed_time() 直接返回毫秒，
    避免 perf_counter（秒级）* 1000 的转换，且更准确反映设备执行时间。

    Args:
        payload: (batch_id, valid_size, arrays_tuple)

    Returns:
        (batch_id, valid_size, predictions_array, latency_ms)
    """
    batch_id, valid_size, arrays = payload
    graph = _get_worker_graph()
    device = _WORKER_STATE["device"]

    with torch.inference_mode():
        inputs = tuple(torch.from_numpy(arr).to(device) for arr in arrays)
        # NPU Event 计时：elapsed_time() 直接返回毫秒
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        prediction = graph(*inputs)
        end_event.record()
        torch.npu.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        values = prediction.detach().float().cpu().numpy().reshape(-1)
    return batch_id, valid_size, values, latency_ms


def _worker_warmup(warmup_arrays: Tuple[np.ndarray, ...]) -> int:
    """warmup 任务：在 worker 进程内运行一次 warmup batch。

    使用与正式推理相同的输入数据，确保 warmup 与后续推理的
    input 数据和 shape 完全一致。

    Args:
        warmup_arrays: warmup 用的输入数组（来自首个 benchmark batch）

    Returns:
        0（占位返回值）
    """
    graph = _get_worker_graph()
    device = _WORKER_STATE["device"]
    warmup_inputs = tuple(torch.from_numpy(arr).to(device) for arr in warmup_arrays)
    with torch.inference_mode():
        graph(*warmup_inputs)
        torch.npu.synchronize()
    return 0


def run_multiprocess_inference(
    batches: List[Batch],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """使用 multiprocessing.Pool 在多进程间并行推理。

    每个进程独立持有 NPU context、编译图与默认 stream，互不干扰。
    编译阶段通过 Lock 序列化，避免 TBE 子进程并发冲突。
    warmup 使用与正式推理相同的输入数据与 shape，确保编译产物一致。

    Args:
        batches: 待推理的 batch 列表

    Returns:
        (predictions, latencies_ms, wall_seconds)
    """
    # 使用 spawn 启动方法：子进程不继承父进程内存状态，
    # 重新初始化 Python 解释器与 NPU 运行时，确保干净的进程隔离。
    ctx = mp.get_context("spawn")

    payloads = [(b.batch_id, b.valid_size, b.arrays) for b in batches]

    # warmup 数据：使用首个 benchmark batch 的输入数据，
    # 确保 warmup 与正式推理的 input 数据和 shape 完全一致。
    warmup_arrays = batches[0].arrays

    # 跨进程编译锁：序列化 torch.compile，避免 TBE 编译子进程并发冲突。
    compile_lock = ctx.Lock()

    started = time.perf_counter()
    pool = ctx.Pool(
        processes=NUM_INSTANCES,
        initializer=_worker_init,
        initargs=(compile_lock, warmup_arrays),
    )
    try:
        # warmup：每个进程执行 WARMUP_STEPS 次 warmup batch 推理
        # 使用与正式推理相同的输入数据，确保 shape 与数据分布一致
        warmup_payloads = [warmup_arrays] * NUM_INSTANCES
        for step in range(WARMUP_STEPS):
            warmup_results = pool.map(_worker_warmup, warmup_payloads)
            print(f"warm-up {step + 1}/{WARMUP_STEPS} complete (procs={len(warmup_results)})")

        # 正式推理：将所有 batch 分发到进程池
        # chunksize=1 确保任务均匀分布，避免某个进程负载过重
        results = pool.map(_worker_run_batch, payloads, chunksize=1)
    finally:
        # 显式清理：先同步所有 NPU 操作，再终止进程池。
        # 避免 TBE 子进程在父进程退出后报 "main process disappeared"。
        pool.close()
        pool.join()

    wall_seconds = time.perf_counter() - started
    results.sort(key=lambda item: item[0])
    predictions = np.concatenate(
        [values[:valid_size] for _, valid_size, values, _ in results]
    )
    latencies = np.asarray([latency for *_, latency in results], dtype=np.float64)
    return predictions, latencies, wall_seconds


def main() -> None:
    if not hasattr(torch.npu, "set_device_limit"):
        raise RuntimeError("当前 torch_npu 不包含 torch.npu.set_device_limit。")
    if not hasattr(torchair, "get_npu_backend"):
        raise RuntimeError("当前 TorchAir 不包含 torchair.get_npu_backend。")
    if torch.npu.device_count() <= DEVICE_ID:
        raise RuntimeError(f"NPU {DEVICE_ID} 不可用。")

    # 父进程仅做数据准备与结果聚合，不初始化 NPU context（避免与子进程冲突）。

    # -------------------- 生成单个随机输入 --------------------
    # 先随机生成一个 input batch，warmup 和后续 1000 次推理都使用这同一份输入，
    # 确保 warmup 与正式推理的 input 数据和 shape 完全一致。
    print("=" * 60)
    print(f"生成单个随机输入 batch (batch_size={BATCH_SIZE}, seed=42)")
    print("=" * 60)
    single_batch = generate_single_random_batch(
        batch_size=BATCH_SIZE, spec=DIN_SPEC, seed=42
    )
    print(f"input shapes: user_id={single_batch.arrays[0].shape}, "
          f"history_item={single_batch.arrays[3].shape}")

    # 将单个 batch 复制 NUM_BENCHMARK_REQUESTS 份，所有副本共享同一份输入数据
    benchmark_batches = replicate_batch(single_batch, NUM_BENCHMARK_REQUESTS)

    # -------------------- warmup + 1000 次推理 --------------------
    print()
    print("=" * 60)
    print(f"warmup + {NUM_BENCHMARK_REQUESTS} 次推理（同一输入）")
    print("=" * 60)
    predictions, latencies, wall_seconds = run_multiprocess_inference(
        benchmark_batches
    )
    if not np.isfinite(predictions).all():
        raise RuntimeError("推理输出包含非有限值。")

    num_batches = len(benchmark_batches)
    summary = {
        "framework": "PyTorch",
        "model": "embedded_demo_din",
        "device": f"npu:{DEVICE_ID}",
        "instances": NUM_INSTANCES,
        "parallelism": "multiprocessing",
        "batch_size": BATCH_SIZE,
        "num_benchmark_requests": NUM_BENCHMARK_REQUESTS,
        "num_batches": num_batches,
        "samples": int(len(predictions)),
        "wall_seconds": float(wall_seconds),
        "throughput_samples_per_second": float(len(predictions) / wall_seconds),
        "throughput_batches_per_second": float(num_batches / wall_seconds),
        "batch_latency_mean_ms": float(latencies.mean()),
        "batch_latency_p50_ms": float(np.percentile(latencies, 50)),
        "batch_latency_p90_ms": float(np.percentile(latencies, 90)),
        "batch_latency_p95_ms": float(np.percentile(latencies, 95)),
        "batch_latency_p99_ms": float(np.percentile(latencies, 99)),
        "batch_latency_max_ms": float(latencies.max()),
        "batch_latency_min_ms": float(latencies.min()),
        "batch_latency_std_ms": float(latencies.std()),
        "op_aicore_num": OP_AICORE_NUM,
        "op_vectorcore_num": OP_VECTORCORE_NUM,
        "autofuse_flags": AUTOFUSE_FLAGS,
        "input_consistent": True,
        "input_seed": 42,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 保存 latency 分布
    with (OUTPUT_DIR / "din_benchmark_latencies.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("batch_id", "latency_ms"))
        for idx, lat in enumerate(latencies):
            writer.writerow((idx, float(lat)))
    with (OUTPUT_DIR / "din_inference_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"first 10 predictions: {predictions[:10]}")
    print(f"latency (ms): mean={latencies.mean():.2f}, p50={np.percentile(latencies, 50):.2f}, "
          f"p95={np.percentile(latencies, 95):.2f}, p99={np.percentile(latencies, 99):.2f}, "
          f"max={latencies.max():.2f}")
    print(f"throughput: {len(predictions) / wall_seconds:.2f} samples/s, "
          f"{num_batches / wall_seconds:.2f} batches/s")
    print(f"output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
