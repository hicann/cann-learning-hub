# ACL C/C++ GEMM 与 SpMV 实验手册

## 1. 实验目标与真实性要求

目标是分别完成 GEMM 和稀疏 SpMV 的真实 ACL/CANN 调用。正式实验必须配置 `-DACL_C_STUB=OFF`；缺少 CANN、ops-sparse 或设备时配置/运行失败是正确行为，Host Stub 不能作为 NPU 结果。

资源生命周期为：`aclInit` → `aclrtSetDevice` → `aclrtCreateStream` → Device 内存/H2D → Tensor 或稀疏描述符 →（仅 GEMM）Workspace 查询与分配 → 算子执行 → `aclrtSynchronizeStream` → D2H → 销毁描述符、内存、Stream、Device、ACL。

GEMM 的关键链是 `aclnnGemmGetWorkspaceSize` → workspace allocation → `aclnnGemm` → synchronize。SpMV 使用 `aclsparseCreateCsr`、`aclsparseCreateDnVec`、`aclsparseSpMV`，随后同步、D2H 并与 `cpu_spmv` 比较。注意区分：只有 GEMM 需要 `aclnnGemmGetWorkspaceSize` 查询并分配 workspace；SpMV 不查询 workspace，`aclsparseSpMV` 的最后一个参数直接传 `nullptr`（API 默认 workspace），对应输出 `Workspace bytes=0 (API default workspace)`。

## 2. 安装与 CANN 配套的 ops-sparse

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

## 3. 直接编译与独立运行

```bash
source "${ASCEND_HOME:?请设置 ASCEND_HOME 为 CANN Toolkit 根目录}/set_env.sh"

cd src/acl_operator_calls/GEMM-acl
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DACL_C_STUB=OFF
cmake --build build -j
./build/bin/gemm_acl --m 512 --n 512 --k 512 --warmup 3 --repeat 10 --device 0

cd ../SpMV-acl
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DACL_C_STUB=OFF
cmake --build build -j
./build/bin/spmv_acl --rows 4096 --cols 4096 --nnz 65536 --warmup 3 --repeat 10 --device 0
```

## 4. 验证、性能与练习

每次运行记录 `Actual Backend`（SpMV 为 `ACL/CANN ops-sparse`，GEMM 为 `ACL/CANN`）、Device ID、时间字段（SpMV 为 `ACL SpMV time`，GEMM 为 `ACL GEMM time`，二者都包含 Stream 同步）和 CPU reference relative L2 error；以程序退出码为首要判据，并使用 `1e-6` 作为实验记录阈值（与程序 Correctness 门槛一致）。

| Operator | Actual backend | Device | ACL time (ms) | CPU reference error | Pass |
|---|---:|---:|---:|:---:|
| GEMM | | | | | |
| SpMV | | | | | |

练习：说明为何异步算子计时必须包含 Stream 同步，以及为何只有 GEMM 需要 `aclnnGemmGetWorkspaceSize` 在执行前查询并分配 workspace，而 SpMV 直接传 `nullptr` 使用 API 默认 workspace。答案仅链接至 `answer/`。
