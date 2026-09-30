# 第4章：GEMM 实现、流水与任务遍历

本目录是可独立使用的课程章节。主用例为 `(M,N,K)=(512,36864,256)`；K 切分说明使用 `(512,36864,2048)`。

- [4.1 章节介绍](04.01_intro.ipynb)
- [4.2 基础 GEMM 实现](04.02_basic_gemm.ipynb)：内存与流水、T.gemm 参数、分核与 K 切分。
- [4.3 流水优化](04.03_pipeline.ipynb)：8 核下比较单、双、三 buffer。
- [4.4 Persistent](04.04_persistent.ipynb)：8 核任务遍历、索引映射及正确性。
- [4.5 实现与调优实战](04.05_practice.ipynb)：四个阶段，依次建立 36 核基线、开启双 buffer、调整 BM、联合使用 A 常驻与 Persistent 列优先任务分配；共五个配置。

## 阅读方式

直接打开上方 Notebook 阅读正文、代码与保留的运行输出；自行执行需要按第1.4节准备950环境。折叠仅用于方便阅读，不影响代码单元的执行顺序。

## 环境与运行

在已安装 TileLang 的环境中运行，并按第1.4节配置好 CANN、PyTorch 和 torch_npu。

### 在终端运行

从仓库根目录执行：

```bash
cd tutorials/tilelang_operator_development/04_gemm
source src/env.sh
python src/check_contracts.py
python src/run_gemm.py basic
python src/run_gemm.py k_demo
python src/run_gemm.py pipeline
python src/run_gemm.py persistent
python answer/04.05_practice.py
```

默认使用当前 Python 环境中已安装的 TileLang，无需额外指定源码目录。若需要使用另一份已编译好的 TileLang，可在执行 `source src/env.sh` 前设置 `TILELANG_COURSE_FORK`，指定其源码目录。这是可选配置；`src/env.sh` 只设置当前 shell 的环境与临时缓存，不修改安装内容。

### 在 Notebook 中运行

Notebook 应在本目录作为工作目录的环境中打开，且从已执行 `source src/env.sh` 的终端启动 Jupyter。代码单元会实际运行正确性验证，需要可用 NPU；没有 NPU 时可阅读正文和已记录的输出，但不代表完成本机验证。Markdown 中的局部片段用于说明，完整实现见相应 `src` 文件。

### 输出文件

运行产生的缓存、日志和性能采集文件默认写入系统临时目录，不属于课程提交文件；可在 source 前设置 `COURSE_OUTPUT_DIR` 指定输出位置。CANN 安装位置不同时，可用 `CANN_ENV_SCRIPT` 指定环境脚本。

## 正确性与性能

正确性验证覆盖随机、零、常量及规则模式四类输入，每类运行两次，检查全部输出，`rtol=atol=1e-4`。运行失败会返回非零状态。正文性能表保留为历史教学示例，不能视为本机本次结果。

```bash
source src/env.sh
msprof --output="${COURSE_OUTPUT_DIR}/profile_s2" --task-time=on --ai-core=off --runtime-api=on \
  python src/profile_gemm.py 512 36864 256 2 contiguous > "${COURSE_OUTPUT_DIR}/profile_s2.log" 2>&1
python src/summarize_profile.py "${COURSE_OUTPUT_DIR}/profile_s2" --blocks 8 --log "${COURSE_OUTPUT_DIR}/profile_s2.log"

msprof --output="${COURSE_OUTPUT_DIR}/practice_step5" --task-time=on --ai-core=off --runtime-api=on \
  python src/profile_practice.py 5 > "${COURSE_OUTPUT_DIR}/practice_step5.log" 2>&1
python src/summarize_profile.py "${COURSE_OUTPUT_DIR}/practice_step5" --blocks 36 --log "${COURSE_OUTPUT_DIR}/practice_step5.log"
```

每次采集请使用新的输出目录，或向汇总脚本传入某次采集的 `op_summary` CSV。汇总脚本先检查同次采集日志含四条 `full-output PASS`、正常结束标记且没有异常，再检查 128 次 kernel 调用，剔除 8 次精度调用及每批 10 次预热，保留 90 个样本。

`profile_practice.py` 参数 1～5 对应 BM64 单 buffer、BM64 双 buffer、BM128 双 buffer、BM256 双 buffer、BM256/BN128 的 A 常驻加列优先；加 `--validate-only` 仅验证正确性。

## 文件结构

- `src/`：算子工厂、校验及性能采集入口。
- `answer/`：课后答案和可运行验证入口。
- `images/`：正文图示。
- `src/original/`：上游扩展示例，供理解 swizzle 等机制；不替代本章 8/36 核教学实现。

核心文件：[8 核 GEMM](src/gemm_kernel.py)、[K 切分](src/gemm_k_demo.py)、[36 核 GEMM](src/gemm_practice36.py)、[A 常驻与列优先](src/gemm_reuse_persistent.py)。
