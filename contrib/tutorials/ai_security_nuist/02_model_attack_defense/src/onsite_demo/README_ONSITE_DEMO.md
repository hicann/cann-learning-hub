# AI 模型后门攻防课程：CANNLab 运行与彩排说明

本章节用于在 CANNLab MindSpore/Ascend 环境中演示后门攻击与后门检测流程。第一小节负责构造并训练 BadNets 后门模型，第二小节使用 STRIP++ 和 Neural Cleanse 思路展示检测结果。

## 目录结构

解压后的完整包通常是：

```text
01_model_backdoor_attack_and_detection/
├── 01.01_badnets_attack.ipynb          # 课堂运行 Notebook：BadNets 攻击
├── 01.02_neural_cleanse_detection.ipynb # 课堂运行 Notebook：后门检测
├── answer/                             # 参考答案说明
├── images/                             # 章节配图目录
└── src/
    ├── data/                           # GTSRB 演示数据
    ├── assets/                         # 正式结果、checkpoint 和检测报告
    ├── final_summary.json              # 正式攻击结果摘要
    ├── requirements.txt                # Python 依赖
    ├── dataset.py / model.py / ...     # 训练、模型与投毒源码
    └── onsite_demo/
        ├── configs/demo_config.json    # 现场 demo 配置
        ├── demo_lib/                   # Notebook 和脚本共用工具
        ├── outputs/                    # 运行输出目录
        ├── scripts/                    # 命令行彩排脚本
        └── README_ONSITE_DEMO.md        # 本说明文档
```

大部分命令需要在 `01_model_backdoor_attack_and_detection/` 章节目录执行；如果已经进入 `src/onsite_demo/` 目录，命令中的脚本路径需要去掉前缀 `src/onsite_demo/`。

## 推荐环境

- Python `3.11.4`（CANN 内核）
- MindSpore `2.2` 或 `2.3`，当前 notebook 也兼容已验证的 `2.7.x` 环境
- Ascend `Snt9B` 或同类 Ascend Notebook 资源
- CANN 与 MindSpore 版本匹配
- JupyterLab `4.x`
- 建议安装 `ipywidgets` 与 `jupyterlab_widgets`

MindSpore 与 CANN 强绑定。华为云 Notebook 镜像通常已经预装 MindSpore 和 CANN；如果手动安装失败，优先使用镜像自带 MindSpore，只补装普通 Python 包。

## 安装依赖

进入章节根目录后执行：

```bash
pip install -r src/requirements.txt
```

如需使用华为云 PyPI 镜像：

```bash
pip install -i https://repo.huaweicloud.com/repository/pypi/simple -r src/requirements.txt
```

如果 MindSpore 已经由云端镜像提供，但 `pip install -r src/requirements.txt` 因 MindSpore 包匹配失败而中断，可以先安装其余依赖：

```bash
pip install numpy pandas matplotlib Pillow ipython jupyterlab notebook ipywidgets jupyterlab_widgets
```

安装完成后重启 notebook kernel，再重新打开 notebook。JupyterLab `4.x` 一般不需要额外安装前端扩展。

## 配置说明

核心配置文件是 `src/onsite_demo/configs/demo_config.json`。

默认设置：

- `target_label = 0`：后门攻击目标类别。
- `cloud_live`：正式现场模式，默认每个触发器训练 `10` epoch。
- `fast`：快速检查模式，默认 `1` epoch，只用于流程验证。
- `square`：右下角半透明方块触发器。
- `checkerboard`：右下角棋盘格触发器。
- `strip_light_k = 8`：light STRIP++ 演示默认扰动次数。

课堂现场建议保持 `cloud_live` 默认配置；Notebook 通过环境变量 `ONSITE_DEMO_PROFILE` 生成 `RUN_PROFILE`，本地测试或时间紧张时再将其设为 `fast`。

## Notebook 运行方式

### 1. 单独运行实验4

在 JupyterLab 中打开项目根目录下的：

```text
01.01_badnets_attack.ipynb
```

按顺序运行全部单元。重点观察：

- 数据集样本浏览是否正常。
- square 与 checkerboard 触发器是否正确叠加在右下角。
- 训练曲线中的 `Train Loss`、`Clean Accuracy` 和 `ASR`。
- 单样本攻击中 clean 输入与 triggered 输入的预测是否从真实类别跳到目标类 `0`。
- 两种触发器对比表中的 clean accuracy、ASR 和 target confidence。

实验4会优先使用现场训练得到的 `demo_last.ckpt`。如果现场训练 checkpoint 不存在，展示逻辑会回退到包内正式 checkpoint，并在输出中说明 fallback 原因。

### 2. 单独运行实验5

在 JupyterLab 中打开项目根目录下的：

```text
01.02_neural_cleanse_detection.ipynb
```

按顺序运行全部单元。重点观察：

- light STRIP++ 是否能把带触发器样本判为可疑。
- 正式检测表中的 STRIP++ detection rate、FPR、ROC-AUC 和 PR-AUC。
- Neural Cleanse 的 `suspected_target_class` 是否为 `0`。
- `mad_anomaly_index` 是否超过阈值 `2.0`。
- 攻击与检测总览图中 `detection_passed` 是否为 `True`。

实验5的正式 Neural Cleanse 结论读取服务器完整 43 类反演结果，不依赖现场重新跑完整反演；这样可以避免现场显存和时间风险。

## Notebook 运行建议

- 第一次运行时选择 `Restart Kernel and Run All Cells`。
- 如果安装了 widgets，单样本攻击展示会优先使用交互控件。
- 如果 widgets 不显示，继续运行 notebook 中的 fallback 普通 cell。
- 如果 Ascend 显存不足，先重启 kernel，再只运行需要展示的 notebook。
- 检测 notebook 中的实时 light STRIP++ 默认使用独立 Python 子进程，以减少当前 kernel 中 MindSpore 显存池的影响。

## 命令行彩排

下面命令建议在章节根目录 `01_model_backdoor_attack_and_detection/` 下执行。

云端正式 10 epoch 彩排：

```bash
python src/onsite_demo/scripts/run_cloud_live_10epoch_acceptance.py --mode cloud_live --device-target auto
```

完整攻击与检测流程彩排：

```bash
python src/onsite_demo/scripts/run_full_onsite_demo.py --mode cloud_live --epochs 10 --batch-size 8 --device-target auto
```

结构与依赖检查：

```bash
python src/onsite_demo/scripts/validate_onsite_demo.py
```

如果只想快速验证流程，不跑完整训练：

```bash
python src/onsite_demo/scripts/run_full_onsite_demo.py --mode fast --epochs 1 --max-steps-per-epoch 1 --device-target auto
```

单独跑 square 触发器：

```bash
python src/onsite_demo/scripts/run_square_baseline.py --epochs 10 --batch-size 32 --device-target auto
```

单独跑 checkerboard 触发器：

```bash
python src/onsite_demo/scripts/run_checkerboard_improved.py --epochs 10 --batch-size 32 --device-target auto
```

单样本攻击 smoke test：

```bash
python src/onsite_demo/scripts/run_interactive_demo_smoke.py --mode enhanced --trigger-type square --random-pick --device-target auto
python src/onsite_demo/scripts/run_interactive_demo_smoke.py --mode enhanced --trigger-type checkerboard --random-pick --device-target auto
```

light STRIP++ 检测 smoke test：

```bash
python src/onsite_demo/scripts/run_detection_demo_smoke.py --mode enhanced --trigger-type square --k 8 --device-target auto
python src/onsite_demo/scripts/run_detection_demo_smoke.py --mode enhanced --trigger-type checkerboard --k 8 --device-target auto
```

如果已经在 `src/onsite_demo/` 目录内，命令改成：

```bash
python scripts/run_full_onsite_demo.py --mode fast --epochs 1 --max-steps-per-epoch 1 --device-target auto
```

## 输出文件

现场输出默认写入：

```text
src/onsite_demo/outputs/demo_runs/
```

每次运行会生成一个带时间戳的目录，例如：

```text
src/onsite_demo/outputs/demo_runs/notebook_YYYYMMDD_HHMMSS/
```

常见输出包括：

- `square_baseline/demo_train_log.json`
- `square_baseline/demo_training_curve.csv`
- `square_baseline/training_curve.png`
- `square_baseline/demo_last.ckpt`
- `square_baseline/demo_eval_summary.json`
- `checkerboard_improved/demo_train_log.json`
- `checkerboard_improved/demo_training_curve.csv`
- `checkerboard_improved/training_curve.png`
- `checkerboard_improved/demo_last.ckpt`
- `checkerboard_improved/demo_eval_summary.json`
- `comparison/comparison_summary.json`
- `comparison/comparison_table.csv`
- `comparison/comparison_report.md`
- `detection/*strip_light_result*.json`
- `detection/*strip_light_preview*.png`

demo subset 默认写入：

```text
src/onsite_demo/outputs/demo_subset/
```

正式服务器结果通常位于：

```text
src/final_summary.json
src/assets/results/metrics/final_summary.json
src/assets/results/detection/detection_complete_summary.json
src/assets/results/detection/detection_complete_table.csv
src/assets/results/detection/detection_complete_report.md
src/assets/evidence/server_artifacts/
```

如果某些正式结果目录在本地包内不可见，说明当前包可能只保留了现场 demo 子集或结果已放在上级 `src/assets/` 中。

## 结果解读

攻击阶段主要看三类指标：

- `clean_accuracy`：干净测试样本准确率。越高说明模型正常业务能力保留越好。
- `ASR` 或 `attack_success_rate`：带触发器样本被预测为目标类 `0` 的比例。越高说明后门越强。
- `avg_target_confidence_on_triggered`：触发后目标类平均置信度。越高说明模型对后门目标越坚定。

检测阶段主要看：

- `strip_detection_rate`：STRIP++ 对触发样本的检出率。
- `strip_fpr`：干净样本被误报为触发样本的比例，越低越好。
- `strip_roc_auc` 与 `strip_pr_auc`：样本级检测区分能力，越接近 `1` 越好。
- `suspected_target_class`：Neural Cleanse 反演出的可疑目标类。
- `mad_anomaly_index`：MAD 异常分数，超过阈值通常表示存在可疑后门目标类。

检测结果只能说明模型存在可疑后门特征或样本具有触发器特征，不代表模型已经修复，也不意味着后门已消除。

## 常见问题

### MindSpore 或 CANN 报错

先确认：

- 当前 kernel 是否使用云端 Ascend Notebook 镜像。
- MindSpore 版本与 CANN 匹配。
- notebook 开头的 CANN 路径是否与实际环境一致。
- 报错后是否已经重启 kernel。

### widgets 不显示

安装或升级：

```bash
pip install -U ipywidgets jupyterlab_widgets
```

然后重启 Jupyter kernel。即使 widgets 不显示，notebook 中的普通 cell fallback 仍可继续展示关键结果。
