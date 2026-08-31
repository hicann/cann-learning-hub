# 第五章：多模态机械臂控制系统（香橙派部署）

> ⚠️ 本章为**硬件端综合实验**，部署在香橙派 OrangePi AIPro + JAKA 机械臂上，**不在 CANNLab 上运行**。本章是前 4 章（YOLO 视觉 / 语音 OCR / LLM 微调 / VLA 训练）的**综合应用与真机落地**。
>
> 📖 章节概述详见 [`05.00_chapter_intro.ipynb`](./05.00_chapter_intro.ipynb)

## 与前序章节的关系

| 前序章节 | 本章复用的能力 |
|----------|----------------|
| 第 1 章 YOLO 目标检测 | YOLO 模型转 OM，在昇腾 310 上做水果检测 |
| 第 2 章 语音 OCR | 华为云 SIS 语音识别（ASR）+ 语音合成（TTS） |
| 第 3 章 LLM 微调 | DeepSeek API 做意图解析与多轮对话 |
| 第 4 章 VLA 训练 | VLA 训练的 ACT 模型也可部署到香橙派（见 04.04） |

## 简介

本系统整合了**视觉（YOLO）+ 语音（华为云 SIS）+ 大模型（DeepSeek）**三种模态，在香橙派 + JAKA 机械臂上实现多模态水果分拣任务。

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                 多模态控制中枢 (start.py)             │
├─────────────┬─────────────────┬─────────────────────┤
│  👁️ 视觉    │   🎤 语音        │   🧠 LLM 大模型      │
│  YOLO识别   │   华为云SIS      │   DeepSeek API      │
│  水果检测   │   ASR→文字       │   意图解析+多轮对话   │
│             │   TTS←播报       │                     │
├─────────────┴─────────────────┴─────────────────────┤
│              意图调度（parse_intent）                 │
│   pick / detect / chat / calib / quit / status      │
├─────────────────────────────────────────────────────┤
│        机械臂执行层（不动，原样保留）                 │
│   JAKA SDK (TCP) + 夹爪 PWM (GPIO) + 手眼标定        │
└─────────────────────────────────────────────────────┘
```

## 硬件要求

| 硬件 | 说明 |
|------|------|
| 香橙派 OrangePi AIPro 20T | 昇腾 Ascend 310B4 NPU，主控 |
| JAKA miniCOBO 机械臂 | 6 自由度协作臂，TCP 连接（IP `10.5.5.100`） |
| USB 摄像头 ×2 | 前置 + 腕部 |
| PWM 夹爪 | wiringPi GPIO 控制（Pin 19） |

## 目录结构

```
05_OrangePi-JAKAArm/
├── README.md                 ← 本文件
├── 0-StarterPack/            香橙派开机配置（教师用）
└── 4-FruitSorter/            ★ 综合主项目（多模态水果分拣）
    ├── start.py              ← 主程序入口（多模态控制）
    ├── speech/
    │   ├── huawei_voice.py   华为云 SIS 语音（ASR/TTS）
    │   └── llm_deepseek.py   DeepSeek LLM（意图解析+对话）
    ├── perception/           YOLO 视觉检测
    ├── control/              JAKA 机械臂 + 夹爪控制
    ├── agent/                任务调度
    └── tools/                标定工具
```

### 分步教学脚本（按需下载）

香橙派分步教学脚本（语音+LLM 基础 / YOLO 推理 3 种方式 / JAKA 机械臂 SDK 样例）托管在 ModelScope，按需下载到本章目录：

```bash
pip install modelscope -q
modelscope download --dataset Kumako/orange_pi_teaching_scripts --local_dir ./teaching_scripts
```

下载后得到 `1-Speech&LLMs/`、`2-YOLO/`、`3-JAKAMinicobo/` 三个分步教学目录，配合 05.00 章节概述中的目录说明使用。

## 三种控制方式

### 1. 👁️ 视觉控制（YOLO）
- 说"打开摄像头"或"检测" → YOLO 识别水果并播报
- 说"抓芒果" → 持续检测直到确认目标，自动抓取放置

### 2. 🎤 语音控制（华为云 SIS）
- ASR：录音 → 华为云一句话识别（`chinese_16k_general`）
- TTS：华为云语音合成播报（`chinese_xiaoyan_common`）
- 麦克风不可用时自动退化为键盘输入

### 3. 🧠 大模型控制（DeepSeek）
- 意图解析：把自然语言解析成结构化意图
- 多轮对话：可以问"你能做什么"等，DeepSeek 理解并回复
- 未匹配预设意图时自动交给 DeepSeek 处理

## 配置

在 `4-FruitSorter/` 目录下创建 `.env`（参考 `.env.example`）：

```bash
# 华为云 SIS（语音）
HUAWEI_SIS_AK=你的AccessKey
HUAWEI_SIS_SK=你的SecretKey
HUAWEI_SIS_REGION=cn-east-3
HUAWEI_SIS_PROJECT_ID=你的项目ID

# DeepSeek（大模型）
DEEPSEEK_API_KEY=你的Key    # https://platform.deepseek.com/ 获取
```

> 💡 三个模块都可缺失降级：无华为凭证→跳过语音；无 DeepSeek→退化为关键词规则；无麦克风→键盘输入。

## 运行

```bash
cd 4-FruitSorter
sudo python3 start.py    # sudo 是因为夹爪 GPIO 需要 root
```

启动后系统播报"多模态机械臂控制系统已就绪"，然后进入语音主循环。

## 各模块说明

| 模块 | 文件 | 说明 |
|------|------|------|
| 主控调度 | `4-FruitSorter/start.py` | 多模态主循环，意图分发 |
| 语音 | `4-FruitSorter/speech/huawei_voice.py` | 华为云 SIS 语音识别+合成 |
| LLM | `4-FruitSorter/speech/llm_deepseek.py` | DeepSeek 意图解析+多轮对话 |
| 视觉 | `4-FruitSorter/perception/yolo_detector.py` | YOLO OM 模型推理（Ascend 310） |
| 机械臂 | `4-FruitSorter/control/jaka_arm.py` | JAKA SDK 逆解+移动 |
| 夹爪 | `4-FruitSorter/control/gripper_pwm.py` | GPIO 软 PWM |

详细的各子模块说明见各子目录的 README。
