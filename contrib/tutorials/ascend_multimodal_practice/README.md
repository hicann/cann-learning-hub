![昇腾AI端云协同与多模态机械臂综合实验](./images/READMEImage.png)

------

## 课程简介

本课程围绕华为云昇腾 AI 生态，覆盖**视觉（YOLO 目标检测）**、**语音（ASR/TTS/声音复刻）**、**语言（LLM LoRA 微调）**、**具身智能（VLA 机械臂训练）**、**综合实战（多模态机械臂）** 五大模块，带你从零构建多模态智能应用并在真实硬件上落地。

课程分为 5 个章节，循序渐进：
- **第 1 章 YOLO 目标检测**：从 OpenCV 图像基础到 YOLOv10 推理，再到昇腾云迁移训练；
- **第 2 章 语音与 OCR**：华为云 SIS（ASR/TTS/声音复刻）+ OCR + DeepSeek LLM，构建多模态助手；
- **第 3 章 LLM 微调**：DeepSeek-R1 在昇腾 NPU 上的 LoRA 高效微调（PyTorch + torch_npu）；
- **第 4 章 VLA 机械臂训练**：用 LeRobot 训练 SO-101 机械臂的 ACT 策略（昇腾 NPU）；
- **第 5 章 综合实战**：多模态机械臂水果分拣（香橙派 + JAKA 机械臂，真机部署）。

## 适用学习人群

<table>
<tr><th>类别</th><th>具体要求</th></tr>
<tr><td>编程基础</td><td>熟悉 Python，了解 pip、import、函数定义</td></tr>
<tr><td>深度学习</td><td>了解"模型"、"训练"、"推理"等基本概念</td></tr>
<tr><td>华为云账号</td><td>第 2 章需开通 SIS/OCR 服务；第 1-4 章需 CANNLab 云开发环境（NPU）</td></tr>
<tr><td>本地硬件</td><td>第 2 章部分实验需麦克风/摄像头；第 5 章需香橙派 + JAKA 机械臂</td></tr>
</table>

> 无需多模态开发经验，每章从基础讲起。

## 能力分层设计

<table>
<tr><th>能力层级</th><th>覆盖章节</th><th>核心能力</th><th>典型任务</th></tr>
<tr><td><b>初级（基础应用）</b></td><td>第 1 章 01.02-01.03、第 2 章 02.02-02.05</td><td>掌握 OpenCV 图像处理、YOLO 推理、华为云 API 调用</td><td>用 YOLO 检测图片中的物体；用 ASR/TTS/OCR 完成语音视觉交互</td></tr>
<tr><td><b>中级（模型训练）</b></td><td>第 1 章 01.04、第 3 章</td><td>掌握昇腾 NPU 上的模型训练流程（迁移训练、LoRA 微调）</td><td>在 NPU 上训练 YOLO 识别自定义类别；微调大语言模型学习新角色</td></tr>
<tr><td><b>高级（综合实战）</b></td><td>第 4 章、第 5 章</td><td>掌握 VLA 具身智能和多模态机械臂控制系统</td><td>训练机器人操作策略；部署视觉+语音+LLM 多模态机械臂</td></tr>
</table>

## 课程支持的硬件产品

<table>
<tr><th>硬件型号</th><th>支持情况</th><th>适用章节</th></tr>
<tr><td>昇腾 NPU 910B3</td><td>✅ 训练与推理</td><td>第 1-4 章（CANNLab 云开发环境）</td></tr>
<tr><td>昇腾 NPU 310B4</td><td>✅ 推理（OM 模型）</td><td>第 5 章（香橙派 OrangePi AIPro）</td></tr>
</table>

## 在线体验环境

本课程第 1-4 章统一在 **CANNLab 云开发环境（昇腾 NPU）** 下验证。只需创建**一个**环境即可覆盖实验 1-4：

<table>
<tr><th>项目</th><th>配置</th></tr>
<tr><td>环境类型</td><td>CANNLab 云开发环境（A2 架构）</td></tr>
<tr><td>镜像模板</td><td><code>cann_8.5.2-py3.11-A2-arm</code>（CANN 8.5.2，Python 3.11）</td></tr>
<tr><td>规格</td><td><code>1*NPU 910B3 16vCPUs 32GiB</code></td></tr>
<tr><td>Python 内核</td><td><b>Python 3.11.4 (CANN)</b></td></tr>
</table>

> 💡 **为什么用一个环境**：CANNLab 提供的是 NPU 环境。推理/API 类任务（实验1推理、实验2）虽然不强制需要 NPU，但在同一个 NPU 环境里也能正常运行。统一一个环境省去切换内核的麻烦。

CANNLab 环境创建步骤见 [CANNLab 环境体验指南](https://gitcode.com/cann/cann-learning-hub/blob/master/docs/CANNLab_env_experience_guide.md)。

> ⚠️ **第 5 章环境说明**：第 5 章为硬件端实验，部署在**香橙派 OrangePi AIPro + JAKA 机械臂**上，不在 CANNLab 上运行。详见 [第 5 章 README](./05_OrangePi-JAKAArm/README.md)。

> 📖 详细的验证操作步骤见 [CANNLab 验证方案](./CANNLab验证方案.md)。

## 课程章节目录

### 第一章：YOLO 目标检测

<table>
<tr><th>Notebook</th><th>Link</th></tr>
<tr><td>1.1 章节概述</td><td><a href="./01_yolo_detection/01.01_chapter_intro.ipynb">前往</a></td></tr>
<tr><td>1.2 OpenCV 图像处理基础</td><td><a href="./01_yolo_detection/01.02_opencv_basics.ipynb">前往</a></td></tr>
<tr><td>1.3 YOLOv10 推理</td><td><a href="./01_yolo_detection/01.03_yolo_inference.ipynb">前往</a></td></tr>
<tr><td>1.4 昇腾云迁移训练</td><td><a href="./01_yolo_detection/01.04_transfer_learning.ipynb">前往</a></td></tr>
<tr><td>1.5 章节实践</td><td><a href="./01_yolo_detection/01.05_chapter_practice.ipynb">前往</a></td></tr>
</table>

### 第二章：华为云语音与视觉 SDK

<table>
<tr><th>Notebook</th><th>Link</th></tr>
<tr><td>2.1 章节概述</td><td><a href="./02_speech_ocr/02.01_chapter_intro.ipynb">前往</a></td></tr>
<tr><td>2.2 语音交互基础</td><td><a href="./02_speech_ocr/02.02_speech_basics.ipynb">前往</a></td></tr>
<tr><td>2.3 语音识别与合成（ASR/TTS）</td><td><a href="./02_speech_ocr/02.03_asr_tts.ipynb">前往</a></td></tr>
<tr><td>2.4 声音复刻</td><td><a href="./02_speech_ocr/02.04_voice_cloning.ipynb">前往</a></td></tr>
<tr><td>2.5 OCR 与 LLM 集成</td><td><a href="./02_speech_ocr/02.05_ocr_llm.ipynb">前往</a></td></tr>
<tr><td>2.6 章节实践</td><td><a href="./02_speech_ocr/02.06_chapter_practice.ipynb">前往</a></td></tr>
</table>

### 第三章：大模型 LoRA 微调

<table>
<tr><th>Notebook</th><th>Link</th></tr>
<tr><td>3.1 章节概述</td><td><a href="./03_llm_finetuning/03.01_chapter_intro.ipynb">前往</a></td></tr>
<tr><td>3.2 环境准备、模型下载与数据预处理</td><td><a href="./03_llm_finetuning/03.02_env_dataset.ipynb">前往</a></td></tr>
<tr><td>3.3 LoRA 配置、训练与推理</td><td><a href="./03_llm_finetuning/03.03_lora_train_infer.ipynb">前往</a></td></tr>
<tr><td>3.4 章节实践</td><td><a href="./03_llm_finetuning/03.04_chapter_practice.ipynb">前往</a></td></tr>
</table>

### 第四章：VLA 与 LeRobot 机械臂训练

<table>
<tr><th>Notebook</th><th>Link</th></tr>
<tr><td>4.1 章节概述</td><td><a href="./04_vla_lerobot/04.01_chapter_intro.ipynb">前往</a></td></tr>
<tr><td>4.2 VLA 原理、数据采集理论与数据集探索</td><td><a href="./04_vla_lerobot/04.02_theory_data.ipynb">前往</a></td></tr>
<tr><td>4.3 ACT 训练与过程可视化</td><td><a href="./04_vla_lerobot/04.03_training.ipynb">前往</a></td></tr>
<tr><td>4.4 离线测试、真机推理与章节实践</td><td><a href="./04_vla_lerobot/04.04_eval_practice.ipynb">前往</a></td></tr>
</table>

> ⚠️ **NPU 训练说明**：LeRobot 官方不支持昇腾 NPU。本章已集成 CANN 官方 [ACT 训练样例](https://gitcode.com/cann/cann-recipes-embodied-intelligence/tree/master/manipulation/act/train) 的 NPU 适配补丁（基于真机数据裁剪），放在 `04_vla_lerobot/src/npu_support/`。在 CANNLab NPU 上训练走脚本方式（`setup_lerobot_npu.sh` + `run_train_npu.sh`），详见 [`src/npu_support/SETUP_NPU.md`](./04_vla_lerobot/src/npu_support/SETUP_NPU.md)。

### 第五章：多模态机械臂综合实验（香橙派部署）

<table>
<tr><th>内容</th><th>Link</th></tr>
<tr><td>5.0 章节概述</td><td><a href="./05_OrangePi-JAKAArm/05.00_chapter_intro.ipynb">前往</a></td></tr>
<tr><td>5.0 详细说明</td><td><a href="./05_OrangePi-JAKAArm/README.md">前往</a></td></tr>
</table>

> ⚠️ 第 5 章为**硬件端综合实验**，不在 CANNLab 上运行。

## 华为云凭证配置（重要，第 2 章必需）

第 2 章（语音与 OCR）的 ASR/TTS/声音复刻/OCR 调用华为云服务，需要配置凭证。**请在开始第 2 章前完成以下配置**：

### 需要的凭证

<table>
<tr><th>凭证</th><th>申请方式</th><th>用途</th></tr>
<tr><td><code>HUAWEI_SIS_AK</code></td><td>访问「<a href="https://console.huaweicloud.com/iam/">我的凭证 → 访问密钥</a>」，单击「新增访问密钥」（仅可下载一次，请妥善保存）</td><td>ASR/TTS/声音复刻</td></tr>
<tr><td><code>HUAWEI_SIS_SK</code></td><td>与 AK 同时生成，在创建访问密钥时的 CSV 文件中</td><td>同上</td></tr>
<tr><td><code>HUAWEI_SIS_PROJECT_ID</code></td><td>在「<a href="https://console.huaweicloud.com/iam/">我的凭证</a>」首页的「项目列表」中，按所选区域（如 <code>cn-east-3</code>）复制对应的项目 ID</td><td>同上（**必需**）</td></tr>
<tr><td><code>HUAWEI_SIS_REGION</code></td><td>开通服务时所在的区域</td><td>默认 <code>cn-east-3</code></td></tr>
<tr><td><code>DEEPSEEK_API_KEY</code></td><td><a href="https://platform.deepseek.com/">DeepSeek 开放平台</a> → 注册登录 → 「API Keys」→ 「创建 API Key」（新用户有免费额度）</td><td>第 2 章 02.05 + 第 5 章的 LLM 对话</td></tr>
</table>

> 💡 凭证获取的详细指引见 [02.02 Cell[2]](./02_speech_ocr/02.02_speech_basics.ipynb) 的凭证申请指南表格。

### 配置步骤

1. 运行第 2 章 [02.02](./02_speech_ocr/02.02_speech_basics.ipynb) 的第一个 code cell，会自动生成 `.env` 模板；
2. 编辑 `.env` 文件，填入你的真实凭证；
3. 后续 02.03-02.05 会自动通过 `load_dotenv()` 读取。

> ⚠️ `.env` 含敏感信息，已在 `.gitignore` 中排除，**切勿提交到 git**。

### 需要开通的华为云服务

| 服务 | 开通入口 | 用途 |
| --- | --- | --- |
| 语音交互服务 SIS | [SIS 控制台](https://console.huaweicloud.com/sis/) → 「开通管理」 | ASR/TTS/声音复刻 |
| 文字识别 OCR | [OCR 控制台](https://console.huaweicloud.com/ocr/) → 选择所需 API 开通 | 02.05 OCR |

> ⚠️ **区域一致性**：请确保开通服务、获取 Project ID、以及 `.env` 中的 `HUAWEI_SIS_REGION` 三者使用**同一区域**（默认 `cn-east-3`）。

## 学习路径建议

```text
第 1 章 YOLO 目标检测（5 小节，约 2 小时）
   ├─ 1.2 OpenCV 基础
   ├─ 1.3 YOLO 推理
   └─ 1.4 迁移训练（昇腾 NPU）
   ↓
第 2 章 语音与 OCR（6 小节，约 2 小时）
   ├─ 2.2-2.4 语音（华为云 SIS）
   └─ 2.5 OCR/LLM（华为云 OCR + DeepSeek）
   ↓
第 3 章 LLM 微调（4 小节，约 2 小时含训练）
   ↓
第 4 章 VLA 机械臂（4 小节，约 5 小时含训练）
   ↓
第 5 章 综合实战（约 4 小时，香橙派 + JAKA 机械臂，硬件端）
```

## 学习成果

完成本课程后，你将能够：

1. **视觉**：用 YOLO 做目标检测，完成自定义类别的迁移训练；
2. **语音**：用华为云 SIS 做 ASR/TTS/声音复刻，构建语音交互应用；
3. **多模态**：集成 OCR + DeepSeek LLM，构建"能听能看能想能说"的智能助手；
4. **大模型**：在昇腾 NPU 上完成 LoRA 微调，掌握高效参数微调全流程；
5. **具身智能**：训练 ACT 策略，部署到真实机械臂完成操作任务；
6. **昇腾生态**：熟悉 CANNLab + PyTorch + torch_npu + CANN 的全栈开发。

## 数据集与模型说明

本课程的数据集和模型分两种方式获取：

### 随课程分发（已在仓库中）

| 资源 | 位置 | 说明 |
| --- | --- | --- |
| YOLOv10n 权重 | `01_yolo_detection/src/yolov10n.pt` | 5.6MB |
| 水果数据集 | `01_yolo_detection/src/fruit/` | 74 训练 + 19 验证，5 类水果 |
| 甄嬛数据集 | `03_llm_finetuning/src/huanhuan.json` | 3729 条对话 |
| 华为云 SIS SDK | `02_speech_ocr/src/huaweicloud_sis/` | 35 个文件 |
| 参考音频 | `02_speech_ocr/images/clone_example.wav` | 声音复刻教学用 |

### 运行时自动下载（notebook 内置下载逻辑）

| 资源 | 来源 | 说明 |
| --- | --- | --- |
| DeepSeek-R1-1.5B 模型 | ModelScope（03.02 自动下载） | 3.3GB |
| VLA 数据集 | ModelScope `Kumako/so101_block_vla`（04.02 自动下载） | 449MB |
| ACT 预训练模型 | ModelScope `Kumako/so101_act_pretrained`（04.04 自动下载） | 197MB |

> 💡 第 3、4 章的大模型/数据集体积较大，已托管在 [魔搭 ModelScope](https://modelscope.cn)，运行 notebook 时自动下载（国内高速，无需登录）。

## 反馈与贡献

如在学习过程中发现问题，欢迎在仓库提 Issue 或 PR。
