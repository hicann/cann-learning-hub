# acl-c：Ascend ACL C/C++ 算子库调用实验

本项目只调用已经安装的 ACL/ops-sparse 算子库，不实现算子 kernel，也不依赖 Python。两个实验可以分别编译：

```text
acl-c/
├── SpMV-acl/                 # ops-sparse aclsparseSpMV（CSR）
│   ├── CMakeLists.txt
│   ├── include/acl_utils.hpp
│   ├── src/main.cpp
│   ├── scripts/{build,run}.sh
│   └── results/
└── GEMM-acl/                 # ACLNN aclnnGemm（FP32）
    ├── CMakeLists.txt
    ├── include/acl_utils.hpp
    ├── src/main.cpp
    ├── scripts/{build,run}.sh
    └── results/
```

## 本地构建

本地没有 CANN 时，脚本会构建一个明确提示“需要 Ascend 环境”的 stub，用于验证 CMake/目录流程；它不会伪装成 ACL 计算结果。昇腾主机上先加载 CANN 环境，再执行真实构建：

### SpMV 依赖：安装与 CANN 配套的 ops-sparse

`ops-sparse` 是独立组件，CANNLab 镜像可能只有源码而没有已安装的头文件和动态库。先检查常用安装位置：

```bash
for root in "${OPS_SPARSE_ROOT:-}" "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}"; do
  [[ -n "$root" ]] || continue
  [[ -f "$root/include/cann_ops_sparse.h" && -f "$root/lib64/libops_sparse.so" ]] && echo "OPS_SPARSE_ROOT=$root"
done
```

若没有输出，优先使用 CANNLab 中与当前 CANN 镜像配套的 `ops-sparse` 源码；环境未预置时，从官方仓库获取与当前 CANN 版本配套的分支或标签。Atlas A3（Ascend910_9362）使用官方构建标识 `ascend910_93`：

```bash
source "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/set_env.sh"
git clone https://gitcode.com/cann/ops-sparse.git
cd ops-sparse
bash build.sh --pkg --soc=ascend910_93 --ops=spmv
RUN_PKG="$(find build_out -maxdepth 1 -type f -name 'cann-A3-ops-sparse-*linux*.run' -print -quit)"
test -n "$RUN_PKG"
"$RUN_PKG" --install --install-path="$HOME/Ascend"
export OPS_SPARSE_ROOT="$HOME/Ascend/cann"
test -f "$OPS_SPARSE_ROOT/include/cann_ops_sparse.h"
test -f "$OPS_SPARSE_ROOT/lib64/libops_sparse.so"
```

源码与 CANN 必须配套；镜像版本变化时按官方 README 选择对应 tag，不要混用头文件和动态库。将 `OPS_SPARSE_ROOT` 设置为实际的 ops-sparse 安装根目录。

```bash
source "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/set_env.sh"
export OPS_SPARSE_ROOT=/path/to/ops-sparse   # 按实际安装目录修改

cd SpMV-acl && bash scripts/build.sh
cd ../GEMM-acl && bash scripts/build.sh
```

也可以显式指定 toolkit：`cmake -S . -B build -DASCEND_HOME=/path/to/ascend-toolkit -DOPS_SPARSE_ROOT=/path/to/ops-sparse`。

## 远程昇腾测试

```bash
git clone git@gitcode.com:maeveyixue/acl-c.git
cd acl-c
source "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/set_env.sh"
export OPS_SPARSE_ROOT=/path/to/ops-sparse

cd SpMV-acl
bash scripts/build.sh
bash scripts/run.sh --warmup 10 --repeat 100

cd ../GEMM-acl
bash scripts/build.sh
bash scripts/run.sh --m 1024 --k 1024 --n 1024 --warmup 10 --repeat 100
```

SpMV 默认沿用已有 `SpMV` 项目的 U1 规模（100,000×100,000、1,000,000 nnz），可用 `--rows/--cols/--nnz` 选择其它规模。它在 CPU 生成 CSR `row_offsets/col_indices/values` 与 `x`，将三组 CSR 数组和向量拷贝到 Device，构造 `aclsparseCreateCsr`/`aclsparseCreateDnVec` 描述符，调用 `aclsparseSpMV`，同步并回读 `y`。SpMV 不查询、不分配 workspace：`aclsparseSpMV` 最后一个参数传 `nullptr`，使用 API 默认 workspace。`ops-sparse` 当前公开接口名是 `aclsparseSpMV`，不是 aclnn tensor 接口。

GEMM 生成固定种子 FP32 的 `A[M,K]`、`B[K,N]`，CPU 计算 reference，然后通过 `aclnnGemmGetWorkspaceSize` 规划并分配 workspace、`aclnnGemm` 执行，回读 C 并计算相对 L2 误差。workspace 查询与 executor 只属于 GEMM 的 ACLNN 链。

输出格式分别为：

```text
Actual Backend=ACL/CANN ops-sparse
Device ID=0
Workspace bytes=0 (API default workspace)
Timing scope=aclsparseSpMV launch + stream synchronize
Matrix:
rows = 100000
cols = 100000
nnz = 1000000

ACL SpMV:
time = ... ms

CPU Reference Error=...
Correctness=...

Actual Backend=ACL/CANN
Device ID=0
Workspace bytes=...
Timing scope=aclnnGemm launch + stream synchronize
ACL GEMM:
  time = ... ms
M=... K=... N=...
CPU Reference Error=...
Correctness=...
```

## 设备、版本与链接检查

```bash
npu-smi info
cat "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/version.info"
find "${OPS_SPARSE_ROOT}" -maxdepth 2 -type f \( -name 'cann_ops_sparse.h' -o -name 'libops_sparse.so' \)
ldd SpMV-acl/build/bin/spmv_acl | grep -E 'ops_sparse|ascendcl|c_sec'
ldd GEMM-acl/build/bin/gemm_acl | grep -E 'opapi|ascendcl'
nm -D "${OPS_SPARSE_ROOT}/lib64/libops_sparse.so" | grep aclsparseSpMV
nm -D "${ASCEND_HOME_PATH}/lib64/libopapi.so" | grep aclnnGemm
```

如果 `nm` 找不到符号，通常是库目录未加入 `OPS_SPARSE_ROOT`/`ASCEND_HOME_PATH`，或没有先 `source set_env.sh`。如果头文件与库来自不同 CANN/ops-sparse 版本，请使用同一套安装包。

## 远程实测记录（历史数据，非当前设备实测）

以下数字来自整改前的远程测试环境（CANN 9.0.0、Ascend 设备，固定随机种子，SpMV `100000×100000 / 1000000 nnz`，GEMM `M=K=N=1024`，warmup=10，repeat=100），只作为历史参考，不能冒充当前设备实测结果。实际运行请以本次实验的输出为准。

SpMV 实测：

```text
ACL SpMV:
time = 76.679907 ms
Correctness:
error = 7.581571451e-08
```

该误差小于 `1e-6`，SpMV 正确性通过。历史记录中，平均一次 ACL SpMV（包含 Device 同步）耗时约 76.68 ms；该数字只代表当时的远程测试环境，不代表当前设备实测。

GEMM ：

```text
ACL GEMM:
  time = 0.056814 ms
M=1024
K=1024
N=1024
Correctness:
  error = 6.409110591e-07
```

该 GEMM 运行成功，历史记录中 `cubeMathType=0`（KEEP_DTYPE，严格 FP32），正确性通过，误差小于 `1e-6`；同样只代表当时的远程测试环境。

## Git 提交

```bash
git status
git add .
git commit -m "Implement ACL C API SpMV and GEMM examples"
git push origin main
```
