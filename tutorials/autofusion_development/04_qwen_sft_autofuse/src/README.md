# Qwen3-1.7B SFT AutoFuse Quickstart

本例在 Linux aarch64 主机的两张 Ascend NPU 上运行 AF OFF/ON 短训练和 profiling，并生成中文
`report.md`。结果是单轮 Quick 方向性证据，不等价于多轮正式收益。

以下命令均在本目录执行，并建议在同一个 Bash 终端中按顺序运行：

```bash
cd autofuse/examples/af_sft_quickstart
```

## 前置条件

- Linux aarch64。
- Python 3.12。
- 两张不同且空闲的 Ascend NPU。
- 至少 30 GiB 空闲磁盘。
- 完整 CANN，且 Toolkit、AutoFuse、Runtime、HCCL、OPP 构建日期不早于 `20260715`。

驱动、固件和 CANN 由系统管理员预先安装。本例只检查，不安装或修改系统组件。

## 1. 检查环境

将 `CANN_ROOT` 改为目标机器上的实际路径：

```bash
export CANN_ROOT=/usr/local/Ascend/ascend-toolkit/latest
bash check_env.sh --cann-root "$CANN_ROOT" --devices 0,1 --python python3.12
```

全部项目必须显示 `[PASS]`。出现 `[FAIL]` 时按输出的 `action` 修复并重试；检查失败时停止，不要继续安装或训练。

## 2. 安装 Python 依赖

```bash
python3.12 -m venv third_party/venv
source third_party/venv/bin/activate
source "$CANN_ROOT/set_env.sh"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.12.0 torch-npu==2.12.0rc1 triton==3.7.0 \
  torchdata==0.11.0 torchao==0.17.0 datasets==5.0.0 \
  pandas==2.2.3 numpy==1.26.4 scipy==1.13.1 \
  tokenizers==0.23.1 safetensors==0.7.0 fsspec==2026.4.0 \
  tyro==1.0.15 tensorboard==2.20.0 wandb==0.28.1 \
  einops==0.8.2 pillow==12.3.0 PyYAML==6.0.3 \
  modelscope==1.38.1 huggingface-hub==1.24.0 pyarrow==21.0.0
python -m pip install \
  decorator==5.1.1 psutil==6.0.0 loguru==0.7.3 matplotlib==3.11.1 \
  msguard==0.0.8 openpyxl==3.1.5 \
  opentelemetry-exporter-otlp-proto-grpc==1.33.1 \
  opentelemetry-exporter-otlp-proto-http==1.33.1 \
  tzdata==2026.3 plotly==6.9.0 argparse==1.4.0 pybind11==3.0.4 \
  attrs==24.2.0
```

不需要安装 `triton-ascend`。不要运行 `pip check`，公共 `torch` 的 CUDA 依赖元数据不适用于本 Ascend 用例。

## 3. 安装源码项目

```bash
mkdir -p third_party/src third_party/wheels

git clone https://gitcode.com/GitHub_Trending/to/torchtitan.git third_party/src/torchtitan
git -C third_party/src/torchtitan checkout --detach ac13e536c84e7f6647b14fa9375c3c8a8a2b8578

git clone https://gitcode.com/cann/torchtitan-npu.git third_party/torchtitan-npu
git -C third_party/torchtitan-npu checkout --detach 5830760386d590722c4acf694383f4e5c4c0ada1

git clone https://gitcode.com/Ascend/torchair.git third_party/src/torchair
git -C third_party/src/torchair checkout --detach 3c9418c2804fbb93c2ca5f0c6b9055cbc873f7d1

python -m pip wheel --no-deps --wheel-dir third_party/wheels third_party/src/torchtitan
python -m pip wheel --no-deps --wheel-dir third_party/wheels third_party/torchtitan-npu
python -m pip wheel --no-deps --wheel-dir third_party/wheels \
  third_party/src/torchair/experimental/_inductor_npu_ext/python

shopt -s nullglob
torchtitan_wheels=(third_party/wheels/torchtitan-*.whl)
torchtitan_npu_wheels=(third_party/wheels/torchtitan_npu-*.whl)
inductor_wheels=(third_party/wheels/inductor_npu_ext-*.whl)
[[ ${#torchtitan_wheels[@]} -eq 1 ]] || { printf 'expected one torchtitan wheel\n' >&2; exit 1; }
[[ ${#torchtitan_npu_wheels[@]} -eq 1 ]] || { printf 'expected one torchtitan-npu wheel\n' >&2; exit 1; }
[[ ${#inductor_wheels[@]} -eq 1 ]] || { printf 'expected one inductor-npu-ext wheel\n' >&2; exit 1; }

python -m pip install --no-deps --force-reinstall \
  "${torchtitan_wheels[0]}" "${torchtitan_npu_wheels[0]}" "${inductor_wheels[0]}"
```

这里使用普通 wheel 安装，不使用 `pip install -e`。

## 4. 准备模型和数据

```bash
mkdir -p assets/hf assets/data/wordle

modelscope download --model Qwen/Qwen3-1.7B \
  --local_dir assets/hf/Qwen3-1.7B

export WORDLE_STAGING="$PWD/assets/.wordle-download"
rm -rf -- "$WORDLE_STAGING"
hf download willcb/V3-wordle data/train-00000-of-00001.parquet --repo-type dataset \
  --local-dir "$WORDLE_STAGING"
install -D -m 0644 \
  "$WORDLE_STAGING/data/train-00000-of-00001.parquet" \
  "$PWD/assets/data/wordle/train-00000-of-00001.parquet"
rm -rf -- "$WORDLE_STAGING"

export WORDLE_PARQUET="$PWD/assets/data/wordle/train-00000-of-00001.parquet"
cp ../af_sft_benchmark/assets.sha256 ./assets.sha256
sha256sum --check assets.sha256
```

11 个文件必须全部显示 `OK`，否则停止。

## 5. 检查 Python 和 NPU

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
env -u LD_PRELOAD -u PYTHONHOME -u CONDA_PREFIX python - <<'PY'
import torch, torch_npu, torchtitan, torchtitan_npu, triton
import inductor_npu_ext

count = torch.npu.device_count()
print(count)
assert count == 2
PY
```

预期打印 `2`。

## 6. 运行对比

```bash
export TORCHTITAN_NPU_DIR="$PWD/third_party/torchtitan-npu"
export NGPU=2

bash run_compare.sh \
  --cann-root "$CANN_ROOT" \
  --model "$PWD/assets/hf/Qwen3-1.7B" \
  --data "$WORDLE_PARQUET" \
  --devices 0,1 \
  --output-dir "$PWD/results"
```

脚本串行执行四次任务：AF OFF/ON 训练各 `10 steps`，AF OFF/ON profiling 各 `12 steps`。报告使用训练
`steps 6-10` 的墙钟均值。训练 seed `42`、profiling seed `43`，四次运行使用独立缓存。

## 7. 查看报告

每次运行会创建独立目录：

```text
results/<UTC timestamp>-<pid>-<attempt>/
├── af_off.log
├── af_on.log
├── profiling_off.log
├── profiling_on.log
├── profiling_off/
├── profiling_on/
└── report.md
```

训练状态：

- `AF_ON_FASTER`：AF ON 平均耗时更低。
- `NO_GAIN`：AF ON 没有更快。
- `RUN_FAILED`：训练失败或稳定步骤无效。

Profiling 状态：

- `COMPARABLE`：两侧 profiling 可比较。
- `UNAVAILABLE`：profiling 失败或结果缺失。
- `INCOMPARABLE`：结果存在，但不满足配对比较条件。

`COMPARABLE` 要求两侧命令成功，CSV 的 Step 5-9 完整、`Device_id` 一致且数值有效。阶段单位为 `us`：

- `Computing`：计算耗时。
- `Communication(Not Overlapped)`：未被计算覆盖的通信耗时。
- `Overlapped`：计算与通信重叠耗时，不能与其他阶段简单相加。
- `Free`：空闲耗时。
- `Stage`：完整阶段耗时。

训练墙钟是性能结论的主证据，耗时越低越好，收益为正值表示 AF ON 更快。Profiling 只解释阶段变化，
不能替代训练结论。

## 常见问题

### CANN 库版本混用

若 `torch_npu` 报 `undefined symbol`，通常是 `LD_LIBRARY_PATH` 中混入了其他 CANN。打开未加载其他 CANN 的
新终端，只设置目标 `CANN_ROOT`，再激活 venv 并加载目标 CANN。不要在同一终端依次加载多个 CANN：

```bash
cd autofuse/examples/af_sft_quickstart
export CANN_ROOT=/usr/local/Ascend/ascend-toolkit/latest
source third_party/venv/bin/activate
source "$CANN_ROOT/set_env.sh"
```

### Hugging Face 无法访问

官方站点不可达时，通过镜像重试 Wordle 下载：

```bash
HF_ENDPOINT=https://hf-mirror.com hf download \
  willcb/V3-wordle data/train-00000-of-00001.parquet --repo-type dataset \
  --local-dir "$WORDLE_STAGING"
```

镜像下载后仍必须执行 `sha256sum --check assets.sha256`。

### 出现 Triton warning

`torchtitan_npu` 可能提示 NPU 使用需要 `triton-ascend`。本 Qwen3-1.7B SFT 配置不使用相关可选算子；两卡
smoke 和训练通过即可，不需要额外安装该包。

### 资产摘要失败

摘要失败表示模型或数据与已验证资产不同。不要跳过校验；删除失败文件后重新下载，再执行
`sha256sum --check assets.sha256`。

## 清理和重跑

只重建 Python 和源码环境，保留模型、数据和报告：

```bash
deactivate 2>/dev/null || true
rm -rf third_party/venv third_party/src third_party/torchtitan-npu third_party/wheels
```

彻底从零重跑：

```bash
deactivate 2>/dev/null || true
rm -rf third_party assets results assets.sha256
```

不要删除或修改系统 CANN、驱动和固件。
