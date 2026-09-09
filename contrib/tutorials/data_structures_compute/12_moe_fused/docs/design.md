# moe_router_fused 融合算子设计文档（纯标量实现，定稿）

> 实验 12：MoE 融合算子和性能分析 —— 融合算子设计
> 目标芯片：Ascend 910B3（ascend910b，arch 3510）；CANN 9.0.0
> 精度：输入/输出 FP16，内部计算 FP32
> 参考实现：同级章节 `08_engineering_deployment_and_perf_analysis` 的 `attention_custom`（纯标量范式）

## 0. 实现路线沿革（为什么是纯标量）

本算子先后尝试了三条实现路线，前两条均被本环境（910B + CANN 9.0.0）的
工具链/硬件缺陷阻断，最终采用与 08 章 `attention_custom` 完全同构的纯标量路线：

| 路线 | 方案 | 阻断原因（均实测） |
| -- | -- | -- |
| A. CV 高阶库 | Matmul 高阶库（Cube）+ TopK 高阶库 | KFC 模式下 `IterateAll` DDR 越界崩溃（MTE 546）；`GetTensorC(LocalTensor)` 返回 C0/NZ 分形布局导致输出全零；FP32 C baseN 只能 16、TopK inner 必须 32 倍数等硬约束使 E≤64 场景 tiling 复杂化 |
| B. 纯向量（AIV） | DataCopy 搬运 + 向量 Mul/Add 树归约 + 标量 softmax/topk | ① 向量 Cast 不支持 BF16↔FP32（接口被迫改 FP16）；② `SyncAll()` 触发 MIX 编译模式（AIC 侧向量指令空操作 + 双任务竞写）；③ reduce 族指令（vcadd/vpadd/vcgadd）在 910B AIV 不产出结果、高阶 ReduceSum 挂起；④「循环内标量读 + 向量指令混用」编译器缺陷：向量写入数据经标量 `GetValue` 回读命中竞态/读零，blockDim=1 也无法通过 |
| C. 纯标量（最终） | 标量 GM 读 + 标量 FMA + UB 普通数组，0 向量指令、0 队列、0 workspace | 无——已验证 8 组用例 × 3 轮确定性 PASS（见 §10）。唯一代价：计算吞吐受标量流水限制（§7） |

纯标量范式的依据：08 章 `attention_custom` 在本环境实测 PASS
（S=512 maxAbsErr≈6e-5），其注释明确「刻意只用标量 GM 访问 + UB 普通数组，
最朴素、最稳定的 API 子集」；`practice_vectorize_answer.md` 亦记载
「本实验环境的 CANN 工具链对『循环内标量读取 + 向量指令混用』存在编译器缺陷」。

## 1. 算子定义

**moe_router_fused**（单 kernel，纯标量）：MoE Router 路径 Step 1~4 的融合：

```
scores      = x @ W_gate                (标量点积)
gate_scores = softmax(scores, dim=-1)   (逐行标量 max/exp/sum)
topk_scores, topk_idx = topk(gate_scores, K, dim=-1, largest=True)  (K 轮 max，选择问题)
topk_weights = topk_scores / sum(topk_scores, dim=-1)   (标量 renorm)
```

| 项 | 规格 |
| -- | -- |
| 输入 x | [N, D]，FP16，ND（行主序） |
| 输入 w_gate | [D, E]，FP16，ND（行主序） |
| 属性 k | INT，Top-K 的 K（1 ≤ K ≤ realE） |
| 属性 e | INT，真实专家数 realE（≤ E，缺省 = E；[realE, E) 为 padding 列，不参与计算） |
| 输出 topk_idx | [N, K]，INT32，ND |
| 输出 topk_weights | [N, K]，FP16，ND（行内和为 1） |

约束：N ≥ 1；1 ≤ realE ≤ E ≤ 32（kernel 内标量数组上限 kMaxE=32）；D ≥ 1（纯标量无对齐要求）。

**接口为什么是 FP16 而非设计初期的 BF16**：向量 Cast 指令不支持 BF16↔FP32
（路线 B 实测 "not support bf16 type cast"）。纯标量路线虽然可用 C++ 转换绕开，
但为与已再生的测试数据、08 章参考实现保持一致，接口定为 FP16。

对照基线（M5）：`matmul → softmax → topk → div` 四个 torch_npu 内置算子序列。

## 2. 数据流设计

### 2.1 算子级数据流（融合前 vs 融合后）

```
未融合（4 kernel，中间结果全部落 GM）：
  GM: x ─┐
         ├─[matmul]─→ GM: scores ─→[softmax]─→ GM: gate ─→[topk]─→ GM: topk_scores ─→[div]─→ GM: weights
  GM: W ─┘                                        GM: topk_idx ──────────────────────────→ 输出

融合（1 kernel，纯标量，中间结果全部留在 UB 普通数组）：
  GM: x ─┐
         ├─[标量点积]─→ UB: scoresRow ─→[标量 softmax]─→[K 轮 max topk]─→[renorm]─→ GM: topk_idx
  GM: W ─┘   (GetValue)    (类成员数组)                                       (SetValue)  GM: topk_weights
```

核心收益（数据结构视角）：`scores / gate_scores / topk_scores` 三个中间张量
（形状 [N,E]、[N,E]、[N,K]）**不再写回 GM、不再从 GM 读回**；4 次 kernel
发射/调度合并为 1 次。

### 2.2 kernel 内数据流（单核视角，每行 n）

```
① scores[n,e] = Σ_d x[n,d] * w_gate[d,e]        e ∈ [0, realE)
   标量 GM 读：xGlobal.GetValue(n*D+d)（行连续）、wGlobal.GetValue(d*E+e)（跨行步长 E）
   FP16→FP32 转换后 FP32 累加；顺带记录行内 max（softmax 第一遍）
② softmax：v = FastExp(scores[e] - rowMax)；行和归一（原地，scoresRow 变 gate 行）
③ TopK：K 轮「严格大于取最大 + 记录索引 + 掩蔽已选」（-1e30 哨兵）
④ renorm：sumK = Σ_k topkVal[k]；逐元素写回：
   idxGlobal.SetValue(n*K+k, topkIdxArr[k])   (INT32)
   wtGlobal.SetValue(n*K+k, half(topkVal[k]/sumK))
```

实现要点：
- 不使用 TPipe/TQue/DataCopy/任何向量指令——08 章实测这些高级特性在本环境
  存在缺陷（MIX 崩溃、TQue/printf 异常、UB GetValue 与大循环组合误编译等）；
- FastExp 为 08 章同款多项式近似（输入范围 [-20, 0]，BF16/FP16 输出精度内）；
- topk 并列处理：严格 `>` 保留最小索引，与参考实现一致（测试数据规避近似平局，
  比对侧另有 |Δw| < 1e-4 的 tie 容忍）。

## 3. Tiling 方案：多核「连续块」行切分（关键设计点）

### 3.1 为什么不能用跨步行进切分（踩坑实录）

直觉的多核切分是 08 章的跨步行进式：`for (n = blockIdx; n < N; n += blockDim)`。
**该切分在本算子下导致非确定性输出错误（实测）**：

- 910B 上 kernel 标量 GM 读/写（`GetValue`/`SetValue`）一律经 L2 缓存
  （`kernel_tensor_impl.h`: 3510 路径经 `ExtractL2CacheGmAddr`；
  cache line = 64B，`kernel_utils_constants.h: CACHE_LINE_SIZE = 64`）；
- **多核并发标量写同一条 64B cache line 会非确定性地丢失部分写**
  （核间对同一 line 的并发写无一致性保证）；
- 本算子输出极小（每行仅 K*4B idx + K*2B wt），跨步行进切分下各核输出行
  按字节交错，所有核必然并发写同一批 line → idx/wt 元素随机丢写成 0；
- 实测表征：blockDim=1 稳定 PASS；blockDim=16/40 时每轮约 30~60% 输出元素
  丢失，且逐轮不同（计算本身正确，仅写回丢失）。
- 08 章 attention 不受影响的原因：其每行输出 64 half = 128B ≥ 2 条 line，
  行间切分下没有两个核写同一条 line。

另注：`SetL2CacheHint(CACHE_MODE_DISABLE)` 在 3510 上无效——`SetValue`/
`GetValue` 内部会把地址高位（cache 模式位）屏蔽，hint 为死代码（实测）。

### 3.2 修复方案：连续块切分 + 64B 对齐块边界

保证**每条输出 cache line 只被一个核写入**：

```
R_align    = 64 / gcd(2K, 64)        // 行对齐单位：使 R_align 行的 wt 字节数 = 64B 整数倍
                                     //（idx 字节数随之对齐；K=2→16 行，K=4→8 行，K=8→4 行）
if N < R_align:  blockDim = 1, rowsPerCore = N              // 不足一段，单核
else:            blockDim = min(GetCoreNumAiv(), N / R_align)
                 rowsPerCore = R_align * ceil(N / (blockDim * R_align))  // 向上取整，均衡负载
                 blockDim = ceil(N / rowsPerCore)                        // 反推核数
core c ∈ [0, blockDim):  n ∈ [c*rowsPerCore, (c+1)*rowsPerCore)   // 尾核收尾到 N
```

- 块边界行号均为 R_align 倍数 → idx 段（K*4B/行）与 wt 段（K*2B/行）的
  字节边界都落在 64B 对齐处 → 无 line 跨核；尾核区间终点为 N（最后一条
  line 只被尾核写，允许不对齐）；
- rowsPerCore 向上取整（而非向下）保证任意两核行数差 < R_align——向下取整会把
  所有余数堆给尾核（实测 N=1024/E=32 尾核独占 3.7× 行数，总耗时翻 2.7×）；
- 910B3 AIV 核数 = GetCoreNumAiv() = 40；N=4096/K=2 时 blockDim=37、rowsPerCore=112；
- 小 N（N < R_align）自动退化单核：此时 launch 开销主导，少核无性能损失。

### 3.3 Tiling 数据结构（op_kernel/moe_router_fused_tiling.h）

```cpp
struct MoeRouterFusedTilingData {
    uint32_t N, D, E, realE, K;   // 形状与属性
    uint32_t blockDim;            // 参与计算的核数
    uint32_t rowsPerCore;         // 每核行数（64B 对齐；尾核承担剩余行）
};
```

纯标量 tiling：host 侧不依赖 MultiCoreMatmulTiling / TopKTilingFunc，不申请
workspace（声明 LibApi workspace 会被标记 matmul 库依赖、触发 MIX 编译模式）。

## 4. UB 预算（单核）

| 项 | 大小 |
| -- | -- |
| scoresRow[kMaxE=32] FP32 | 128 B |
| topkVal[32] FP32 + topkIdxArr[32] INT32 | 256 B |
| 合计（类成员数组） | < 0.5 KB |

远低于 910B 单核 UB 192KB 与 32KB 栈帧上限（08 章实测约束）。
无队列缓冲、无 workspace——这是纯标量范式的附带收益：预算核算从 ~104KB
（路线 A 估算）降到可忽略，设计审查面大幅缩小。

## 5. HBM 流量估算表（融合前 vs 融合后）

记 N/D/E/K 为形状，cores = blockDim，元素字节：FP16=2B、FP32=4B、INT32=4B。
中间张量 = scores[N,E]、gate_scores[N,E]、topk_scores[N,K]（BF16/FP16，与 baseline 一致）。

| 张量 | Baseline（4 算子） | Fused（1 kernel） | 说明 |
| -- | -- | -- | -- |
| x 读取 | N·D·2 | N·D·2 | 相同 |
| W_gate 读取 | D·E·2（+多核复制） | cores·D·E·2（逻辑值；W 仅 KB 级，L2 192MB 命中后 HBM 增量≈0） | 纯标量下每核逐元素重读 W，L2 是唯一复用层 |
| scores 写+读（softmax） | 2·N·E·2 | **0** | 留 UB 数组 |
| gate 写+读（topk） | 2·N·E·2 | **0** | 原地复用 scoresRow |
| topk_scores 写+读（div） | 2·N·K·2 | **0** | topkVal 数组 |
| topk_idx 写 | N·K·4 | N·K·4 | 相同（最终输出） |
| topk_weights 写 | N·K·2 | N·K·2 | 相同（最终输出） |
| **中间张量 GM 流量小计** | **16NE + 8NK (B)** | **0** | ↓100% |

数值例（N=128, D=512, E=16, K=2）：中间张量流量 baseline ≈ 34.8KB → fused 0；
数值例（N=4096, D=2048, E=8, K=2）：中间张量流量 baseline ≈ 576KB → fused 0。

**注意**：流量收益是「融合」本身的收益，与计算路线无关；但流量不是本实现的
性能瓶颈（见 §7）。

## 6. TopK 实现选型（数据结构视角）

| 方案 | 算法 | 复杂度（每行） | 结论 |
| -- | -- | -- | -- |
| A. K 轮 max+mask（本实现） | K 次线性扫描取最大 + 哨兵掩蔽 | O(K·E)，E≤32、K≤8 | **主实现**：纯标量路径下与 TopK 库同为选择类算法；平局取最小索引，确定性 |
| B. Ascend C TopK 高阶原语 | 基数选择 | O(E) | 路线 A（CV）采用过；约束 inner%32==0 需 repack，且随 CV 路线一起放弃 |
| C. 全排序 | O(E·log E) | 不采用 | Top-K 是 selection 问题不是 sorting 问题（本实验的数据结构论点） |

本实验的「数据结构」主线：Top-K = 选择问题（selection），K 轮 max 是最朴素
的选择算法实例；融合的可行性判据 = 子图中间张量容量 ≤ UB 且无跨核依赖
（本算子中间量 ≤ 0.5KB/行，N 维切分无跨核边）——数据结构形态决定融合形态。

## 7. 性能形态分析（纯标量的代价，实测）

- 逐行计算量：D·E 次标量 GM 读（x 行 D 次 + W 列 D·E 次）+ D·E 次标量 FMA；
- 08 章标定：纯标量 GM 读有效延迟约 50ns/次/核（attention S=512 85ms ≈ 67M 读 / 40 核）；
- **实测耗时**（`test/main.cpp --bench` 事件计时，含 3 次预热，括号内为 blockDim）：

| 用例 | 形状 (N·D·E·K) | fused 实测 | 4 算子 baseline (M1) | 比值 |
| -- | -- | -- | -- | -- |
| case_16_256_8_2 | 16·256·8·2 | 0.335 ms (bd=1) | —（发射开销下限 ~0.23ms） | ~1.5× |
| case_128_512_16_2 | 128·512·16·2 | 2.36 ms (bd=8) | 0.226 ms | ~10× 慢 |
| case_1024_1024_32_4 | 1024·1024·32·4 | 105.7 ms (bd=32) | 0.263 ms | ~400× 慢 |
| case_4096_2048_8_2 | 4096·2048·8·2 | 60.6 ms (bd=37) | 0.310 ms | ~200× 慢 |

- **加速比 KPI（小 N ≥1.5×、大 N ≥1.1×）不达**——这是选择纯标量路线时已接受的
  权衡（见 §0）；
- 归因：融合消除了中间张量 GM 往返与 3 次发射（数据结构收益为真），但把计算搬到
  了标量流水——**标量 GM 读延迟 × 次数成为新瓶颈**（arithmetic intensity 过低：
  每字节访存只配 ~1 次标量运算）。这正是性能分析章节的论点：**融合改变访存形态，
  不改变计算吞吐；吞吐由执行管线（标量/向量/Cube）决定**；
- 实测还暴露并修复了一个多核负载均衡问题：若「前 (blockDim-1) 核取整段、尾核收尾」，
  N=1024/E=32 时尾核独占 3.7× 行数、总耗时由尾核决定（281ms）；改为向上取整
  rowsPerCore + 反推核数后均衡（105.7ms，§3.2）；
- 完整 A/B 对比 + msprof 采集（aiv_vec_ratio≈0、PipeUtilization）留待 M5 正式报告。

## 8. 数值精度与边界处理

| 项 | 处理 |
| -- | -- |
| 内部精度 | 点积/softmax/TopK/renorm 全程 FP32；输入 FP16 读入即转 FP32；输出权重转 FP16 |
| softmax 溢出 | 先减行内 max，exp 输入 ≤ 0，无上溢；FastExp 截断 [-20, 0] |
| 平局（tie） | 严格 `>` 保留最小索引（与 torch.topk+参考实现口径一致）；比对侧 tie 容忍 |Δw|<1e-4 |
| E 尾列 | realE 属性区分有效专家，padding 列不参与 softmax/topk |
| N 尾块 | 尾核承担剩余行（§3.2），行循环以 N 为界，无越界读写 |
| 权重阈值 | idx 逐元素精确匹配（tie 容忍内除外）；wt rtol=atol=1e-2（FP16 量化误差，实测 max_abs ≈ 7e-4） |

## 9. 工程结构

```
12_moe_fused/
├── docs/design.md                     # 本文档
├── data/case_<N>_<D>_<E>_<K>/         # 8 组测试数据（gitignore，gen_test_data.py 再生）
│   ├── meta.json                      # N/D/E(真实)/E_pad/K
│   ├── x_fp16.bin / wgate_fp16.bin    # FP16 输入
│   └── ref_topk_idx.npy / ref_topk_weights.npy  # FP32 参考
├── src/custom_op/                     # 算子工程（08 章模板同构）
│   ├── op_host/moe_router_fused.cpp   # TilingFunc（§3.2 切分）+ 算子注册（FP16/INT32）
│   ├── op_kernel/moe_router_fused.cpp # 纯标量 kernel
│   ├── op_kernel/moe_router_fused_tiling.h
│   ├── test/main.cpp                  # aclnn C++ 测试（npy 解析 + tie-aware 比对 + --dump）
│   ├── test/run.sh                    # 部署 opp_pkg + 编译运行单用例
│   └── build.sh                       # cmake preset 构建 + 打包 .run
└── tools/
    ├── moe_ref.py                     # FP32 参考实现（M1）
    ├── gen_test_data.py               # 测试数据生成（M1）
    ├── test_moe_router.py             # 参数化正确性回归（M4）
    └── run_all.sh                     # 一键回归
```

## 10. 验收清单（实测结果）

- [x] 8 组用例（含边界形态：N=16 最小、N=100/129 非对齐、E=4 小专家、
      E=32/K=4 大专家、N=4096/D=2048 最大）× 3 轮确定性 **全部 PASS**；
      idx 逐元素 100% 匹配、wt max_abs ≤ 7.6e-4（阈值 1e-2）
- [x] blockDim=1（单核）与多核（blockDim=6~40）两种形态均 PASS
- [x] 对照实验：08 章 attention_custom（同范式）本环境复测 PASS，排除环境漂移
- [x] 多核写丢失根因定位与修复：跨步行进→连续块 + 64B 对齐（§3，含反例证据链）
- [x] UB 预算 < 0.5KB（§4）；tiling 无越界（块划分覆盖全 N，尾核收尾）
- [x] 精度链路：FP16 in → FP32 内部 → FP16 out，softmax 防溢出（§8）
- [x] HBM 流量估算表（§5）与 TopK 选型论证（§6）
- [ ] M5 性能实测与 msprof 对比（benchmark 脚本与报告，进行中）
