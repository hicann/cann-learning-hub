# 章节源码与运行资源

本目录集中保存章节的全部工程资源；课程总览、学习目标、硬件和在线环境说明见上级课程 [README](../../README.md)。

## 内容

```text
src/
├── requirements.txt
├── data/                         # GTSRB 小规模演示数据
├── assets/                       # 正式 checkpoint、指标与检测报告
├── onsite_demo/                  # Notebook 共用运行库、配置和脚本
├── final_summary.json            # 正式攻击结果摘要
├── dataset.py
├── evaluate.py
├── experiment.py
├── model.py
├── poison.py
├── train.py
└── utils.py
```

Notebook 从章节根目录向上定位 `src/onsite_demo/`，运行库再统一解析 `src/data/`、`src/assets/` 和 `src/onsite_demo/outputs/`，无需修改工作目录或设置额外的项目绝对路径。

## 安装普通依赖

在章节根目录执行：

```bash
python -m pip install -r src/requirements.txt
```

MindSpore、CANN 与 TBE 应由 CANNLab 镜像提供，不建议在普通 CPU 环境中单独安装替代版本。

## 验证

```bash
python src/onsite_demo/scripts/validate_onsite_demo.py
python src/onsite_demo/scripts/final_preupload_audit.py
```

完整命令行彩排说明见 [onsite_demo/README_ONSITE_DEMO.md](./onsite_demo/README_ONSITE_DEMO.md)。

