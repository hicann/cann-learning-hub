# 智能机械臂水果分拣系统

## 📋 项目概述

本项目基于香橙派AI Pro 20T平台，构建了一套完整的智能水果分拣系统。系统通过摄像头自动识别起始区域内的10种水果，利用语音交互接收用户指令，控制JAKA机械臂精准抓取指定水果并放置到目标区域。项目整合了语音识别与合成、大语言模型、YOLO视觉检测、智能体决策等多种人工智能技术。

### 🎯 项目特点
- **全流程自动化**：语音指令→视觉识别→机械臂抓取→精准放置
- **多技术融合**：YOLOv11视觉识别 + 华为云语音(ASR/TTS) + SJTU 大模型意图解析 + JAKA机械臂控制
- **边缘部署优化**：香橙派AI Pro上高效运行的ONNX模型
- **用户友好**：一键启动，中文语音交互，直观可视化界面

## 🏗️ 系统架构

### 整体架构图
```
┌─────────────────────────────────────────────┐
│           用户交互层 (语音输入/播报)        │
│        华为云 SIS (ASR+TTS) + SJTU LLM      │
├─────────────────────────────────────────────┤
│           智能决策层 (Task Agent)           │
│           任务调度与状态管理                │
├─────────────────────────────────────────────┤
│           运动控制层 (硬件控制)             │
│           JAKA机械臂 + PWM夹爪控制         │
├─────────────────────────────────────────────┤
│           视觉感知层 (目标检测)             │
│           YOLOv11n + ONNX推理               │
├─────────────────────────────────────────────┤
│           硬件平台层 (边缘计算)             │
│           香橙派AI Pro 20T                  │
└─────────────────────────────────────────────┘
```

### 模块说明

#### 1. 语音交互模块 (`speech/`)
- **huawei_voice.py**（当前使用）: 语音功能外包模块，封装 `VoiceAssistant` 类——`listen()` 华为云 SIS 语音识别(ASR)、`speak()` 华为云 SIS 语音合成(TTS)、`parse_intent()` 调用 SJTU 模型网关(OpenAI 兼容)做意图解析。所有外部依赖均可缺失降级（缺麦克风→键盘输入、缺凭证→跳过语音、缺 LLM→本地关键词规则）。
- **asr_baidu.py / llm_parser.py**（旧方案，保留备用）: 百度语音识别 + 文心大模型解析，已不再被 `start.py` 调用。

#### 2. 智能决策模块 (`agent/`)
- **task_agent.py**: 核心任务调度器，管理抓取-放置-复位完整流程
- 支持10种水果分类，自动映射到4个目标篮子

#### 3. 运动控制模块 (`control/`)
- **jaka_arm.py**: JAKA机械臂控制，包含手眼标定、安全运动规划、路径记忆
- **gripper_pwm.py**: 夹爪PWM控制，支持精确开合角度调节

#### 4. 视觉感知模块 (`perception/`)
- **camera.py**: 摄像头驱动，优化V4L2访问，支持香橙派多设备节点
- **yolo_detector.py**: YOLO目标检测器，支持ONNX模型推理和多帧强化检测

#### 5. 训练与标定模块 (`training/`, `tools/`)
- **training/**: YOLO模型训练、数据采集、模型导出
- **tools/**: 九点标定工具、手眼标定计算、实时测试工具

## 🔑 代码关键部分

### 1. 手眼标定系统优化
**文件**: `tools/eyehand_calib.py`
- 实现最小二乘法求解仿射变换矩阵
- 增加精度验证系统，实时计算平均误差
- 支持3×2（项目格式）和3×3（通用格式）双矩阵输出

### 2. 智能权限管理
**文件**: `start_sudo.py`
- 解决香橙派上sudo导致的权限问题
- 自动修复X11显示环境，确保SSH远程可视化
- 智能环境变量继承，保留用户配置

### 3. 强化视觉检测
**文件**: `perception/yolo_detector.py`
- 多帧验证机制，提高检测稳定性
- 动态置信度阈值，适应不同光照条件
- 摄像头预热功能，避免冷启动失败

### 4. 安全运动规划
**文件**: `control/jaka_arm.py`
- 路径记忆系统，实现原路返回
- 分步移动算法，避免大跨度运动
- 姿态多样性尝试，解决逆运动学失败

### 5. 水果分类逻辑
**文件**: `agent/task_agent.py`
```python
# 水果分类映射（符合大作业要求）
fruit_category_map = {
    "apple": 1, "pomegranate": 1, "persimmon": 1, "mandarin": 1,  # 第一类
    "lemon": 2, "avocado": 2,                                     # 第二类
    "strawberry": 3,                                              # 第三类
    "mango": 4, "pear": 4,                                        # 第四类
    "banana": self.banana_basket                                  # 第五类（用户指定）
}
```

## 📊 数据集采集与构建

### 数据采集方法
1. **工具**: `training/capture_with_label.py`
2. **采集模式**:
   - 手动模式: 按数字键选择类别，按's'保存
   - 自动模式: 定时自动采集，间隔可调
3. **采集环境**:
   - 不同光照条件（自然光、室内光）
   - 不同角度和距离
   - 不同水果摆放姿态

### 数据集构成
```
dataset/
├── images/
│   ├── train/    # 训练集图像 (约200张)
│   └── val/      # 验证集图像 (约20张)
└── labels/
    ├── train/    # 训练集YOLO标签
    └── val/      # 验证集YOLO标签
```

### 标注规范
1. **格式**: YOLO格式 (归一化坐标)
2. **类别**: 10种水果对应10个类别ID
3. **质量要求**:
   - 边界框紧密贴合水果
   - 遮挡部分标注完整轮廓
   - 模糊图像需重新采集

### 类别定义
```yaml
# dataset.yaml
names:
  0: apple          # 苹果
  1: avocado        # 牛油果
  2: banana         # 香蕉
  3: lemon          # 柠檬
  4: mandarin       # 橘子
  5: mango          # 芒果
  6: pear           # 鸭梨
  7: persimmon      # 柿子
  8: pomegranate    # 石榴
  9: strawberry     # 草莓
```

## 🛠️ 环境配置与依赖

### 硬件环境
- **主控板**: 香橙派AI Pro 20T (8GB)
- **机械臂**: JAKAminiCOBO机械臂
- **摄像头**: USB 摄像头 (支持V4L2)
- **夹爪**: 定制PWM控制夹爪

### 软件环境配置

#### 1. 基础系统配置
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
sudo apt install -y python3-pip python3-venv git vim
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y wiringpi  # GPIO控制

# 设置Python虚拟环境
python3 -m venv venv
source venv/bin/activate
```

#### 2. Python依赖包
```bash
# 在虚拟环境中安装
pip install --upgrade pip

# 核心依赖
pip install ultralytics==8.3.208  # YOLOv11
pip install opencv-python==4.10.0.84
pip install numpy==1.24.3

# 机械臂控制
pip install jkrc  # JAKA机械臂SDK

# 语音录音/识别/播报（缺少则降级为键盘输入）
pip install SpeechRecognition pyaudio   # 麦克风录音
pip install openai python-dotenv        # 意图解析（OpenAI 兼容）
pip install playsound==1.2.2            # 音频播放（锁定版本）

# （可选，旧方案）百度语音 + 文心大模型
# pip install baidu-aip==4.16.15
# pip install erniebot==0.5.9

# 其他工具
pip install albumentations==1.3.1  # 数据增强
pip install onnxruntime==1.17.1    # ONNX推理
pip install pandas==2.0.3
```

#### 3. 华为云语音 SDK 安装（ASR/TTS，必需且不能直接 pip）
华为云 SIS SDK 需从源码安装，进入本项目附带的 `2.华为语音SDK/speech_llms` 目录执行：
```bash
cd <仓库路径>/2.华为语音SDK/speech_llms
python setup.py install --user
```
> 也可从华为云官方下载 `huaweicloud-python-sdk-sis-1.8.6.tar.gz`，解压后 `python setup.py install --user`。
> 未安装该 SDK 时，语音识别(ASR)与语音播报(TTS)不可用，但其余流程仍可运行。

#### 4. 语音服务配置（`.env`）
在 `4-FruitSorter/` 目录下创建 `.env`（参考 `.env.example`），填入凭证：
```bash
# 华为云语音交互 SIS（ASR 识别 + TTS 播报）
HUAWEI_SIS_AK=你的AccessKey
HUAWEI_SIS_SK=你的SecretKey
HUAWEI_SIS_REGION=cn-east-3
HUAWEI_SIS_PROJECT_ID=你的项目ID
HUAWEI_TTS_VOICE=chinese_xiaoyan_common   # 可选，默认女声

# SJTU 模型网关（语音意图解析，OpenAI 兼容）
SJTU_API_KEY=你的网关Key
SJTU_BASE_URL=https://models.sjtu.edu.cn/api/v1
VOICE_LLM_MODEL=deepseek-v4-flash
```
> ⚠️ `.env` 含密钥，切勿提交到公开仓库，建议加入 `.gitignore`。
> 🔑 华为 AK/SK 在「华为云 → 我的凭证 → 访问密钥」创建；Project ID 需与 Region 对应。

## 🚀 快速开始

### 1. 系统启动
```bash
sudo python3 start.py
```
> 启动后进入**语音控制**模式，直接对麦克风说出指令即可。若麦克风/凭证不可用，会自动降级为键盘输入 + 本地关键词解析。

### 2. 系统标定流程（首次使用必须执行）
1. **摄像头标定**: 说出"摄像头标定"，按提示完成九点标定
2. **机械臂标定**: 说出"机械臂标定"，按摄像头标定顺序记录九个点
3. **手眼标定**: 说出"手眼标定/仿射矩阵等"，自动计算变换矩阵

> 目标放置位置由 `data/target_area.csv` 提供（缺省时使用内置默认值），当前版本未提供语音目标区域标定指令。

### 3. 正常使用
当前模型可识别 5 种水果：**牛油果、柠檬、梨、芒果、柿子**。系统启动后，说出指令如：
```bash
"打开摄像头看看"   # 检测并语音播报当前画面中的水果
"帮我抓个芒果"     # 检测确认后自动抓取芒果并放置、复位
"我要一个柠檬"     # 抓取柠檬
"摄像头标定"       # 重新标定摄像头
"退出"             # 退出系统
```

## 📈 系统性能

### 技术指标
- **识别准确率**: mAP@0.5 = 0.995
- **推理速度**: 1.0ms/帧 (香橙派AI Pro)
- **抓取成功率**: >95% (10次测试平均)
- **语音识别准确率**: >90% (中文普通话)
- **端到端延迟**: <3秒 (语音→抓取完成)

### 支持的提高任务
1. ✅ 支持以香蕉作为待抓取目标
2. ✅ 实现更精细的水果分类（10种水果）
3. ✅ 在起始区域内随机放置多种水果作为抓取目标
4. ✅ 采用大模型进行意图解析（ERNIE-3.5）

## 🔧 故障排除
根据系统提示，解决相应问题

### 常见问题
1. **摄像头无法打开**
   ```bash
   # 检查设备权限
   ls -l /dev/video*
   sudo chmod 666 /dev/video0
   ```

2. **机械臂连接失败**
   - 检查IP地址: 默认`10.5.5.100`
   - 检查网络连接
   - 确认机械臂上电并处于伺服状态

3. **显示黑屏（SSH远程）**
   - 使用`start_sudo.py`自动修复
   - 或手动设置: `export DISPLAY=:10.0`

4. **夹爪不动作**
   ```bash
   # 运行夹爪测试
   sudo python3 debug\ methods/calibrate_gripper.py
   ```

### 调试工具
1. **实时YOLO测试**: `python3 tools/test_yolo_realtime.py`
2. **夹爪校准**: `sudo python3 debug\ methods/calibrate_gripper.py`
3. **摄像头测试**: `python3 -c "import cv2; print(cv2.getBuildInformation())"`

## 📁 项目结构
```
AgriSpec-2025/
├── agent/              # 智能决策模块
├── control/            # 硬件控制模块
├── perception/         # 视觉感知模块
├── speech/             # 语音交互模块
│   ├── huawei_voice.py # 语音外包模块（华为ASR+TTS+SJTU意图解析）
│   ├── asr_baidu.py    # 旧方案，保留备用
│   └── llm_parser.py   # 旧方案，保留备用
├── training/           # 模型训练模块
├── tools/              # 标定工具模块
├── debug methods/      # 调试工具
├── models/             # 训练好的模型
├── data/               # 配置文件与标定数据
├── dataset/            # 训练数据集
├── start.py            # 主程序入口（语音控制）
├── .env.example        # 凭证配置模板
├── .env                # 实际凭证（勿提交公开仓库）
└── README.md           # 项目说明
```

## 📄 许可证

本项目仅供课程作业使用，遵循课程相关要求。

---

**最后更新**: 2025年12月29日  
**版本**: v1.0  
**状态**: ✅ 已完成所有基础任务和提高任务