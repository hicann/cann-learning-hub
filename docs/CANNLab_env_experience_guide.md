# CANNLab 云开发平台体验指南

---

## 一、CANNLab简介

CANNLab 是 CANN 社区推出的基于云开发者空间提供 NPU 算力资源的一站式云开发平台。平台面向 CANN 生态的算子开发、模型训练与推理、AI 应用开发等场景，为开发者提供开箱即用的云端开发环境。

开发者无需在本地安装 CANN 工具链，通过 GitCode 上的[ CANN 开源仓库入口](https://gitcode.com/org/cann/cannlab)即可进入平台，按需申请算力，使用 WebIDE 或本地 VSCode 远程连接进行开发。

---

## 二、使用CANNLab环境在线体验cann-learning-hub仓教程

### 2.1 前提条件

**注册 GitCode 账号**

1. 访问[ GitCode 官网](https://gitcode.com)
2. 点击页面右上角 "注册" 按钮
3. 选择注册方式：支持手机号、邮箱或微信注册
4. 按提示完成注册流程

**访问CANNLab环境**

- 如果您的开发习惯是在浏览器 IDE 中在线开发（如 Code Server），建议提前安装Chrome 浏览器。
- 如果您的开发习惯是在本地 IDE 中进行开发（如 VSCode、Jetbrains），建议提前安装 VSCode。

### 2.2 操作步骤

#### 2.2.1 创建环境

1. 进入 CANN 开源组织 https://gitcode.com/cann
2. 点击导航栏中的 **CANNLab**页签后，正式进入 CANNLab 主页。

<img src="./images/CANNLab_env_experience_guide/png1.png">

3. 左侧导航栏切换到**我的环境**，点击**新建环境**可创建 NPU环境，

<img src="./images/CANNLab_env_experience_guide/png2.png">

根据需要选择对应的 A2 或者 A3 环境，选择模板和 NPU 规格，这里以体验ascendc_operator_development课程为例，根据课程所需配置信息，创建 NPU 环境，规格配置如下：

- **开发环境名称**：自行命名。
- **处理器类型**：昇腾 NPU。
- **模板名称**：`cann_9.0.0 py3.11-A2-arm`。
- **规格**：`1*NPU 910B3 16vCPUs 32GiB`。
- **预加载代码仓**
	- **选择仓库**：`CANN/cann-learning-hub`
	- **选择分支**：`master`

<img src="./images/CANNLab_env_experience_guide/png3.png">
<img src="./images/CANNLab_env_experience_guide/png4.png">


> 注意：不同课程的规格配置可能会有所不同，开发者需要根据要体验的课程 `README.md` 中`在线体验环境`说明的规格参数配置并创建对应的 NPU 环境，否则可能会出现未知错误。

环境创建完成后，NPU 环境会自动启动，可以在我的环境页面进行开机/关机/连接/CANN包升级/删除操作。
> 注意：如果开机时提示资源不足，说明当前时间段使用人数较多，可稍后再尝试。

<img src="./images/CANNLab_env_experience_guide/png5.png">

#### 2.2.2 使用环境

<img src="./images/CANNLab_env_experience_guide/png6.png">

环境创建成功后，点击 **"连接"**，选择一种连接方式：

|方式  |适用场景  |说明  |
|--|--|--|
|WebIDE  |轻量开发、跨设备使用  |浏览器直接打开，无需安装任何软件  |
|Visual Studio Code  |重度开发、习惯本地编辑器  |需安装连接云端开发环境的插件  |

##### 2.2.2.1WebIDE 连接

这里以 WebIDE 连接方式为例，展示如何使用环境体验cann-learning-hub仓课程：
1. 点击 **"连接"** -> 选择 **"WebIDE"**
2. 等待页面加载完成，即可进入浏览器内 IDE
3. WebIDE 支持同时开启最多 3 个独立工作窗口，但不推荐开启多个工作窗口
4. 连接成功后，可以看到整体界面类似 VS Code，支持源代码管理、扩展安装等操作，展示如下：

<img src="./images/CANNLab_env_experience_guide/png7.png">

> 说明
> - 目前支持 Chrome 浏览器
> - 如遇到页面卡住或连接断开，可尝试关机再开机恢复实例

### 2.3 体验教程

环境创建并连接成功后，即可打开仓库中的教程文件进行体验。下面以 `cann-learning-hub/tutorials/ascendc_operator_development/02_AscendC_basic/02.02_HelloWorld.ipynb` 为例：

点击窗口左侧菜单栏中的 **资源管理器**，在仓库目录中找到并打开 **02.02_HelloWorld.ipynb** 教程文件。打开后，在 Notebook 右上角点击 **选择内核**，选择 Python Kernel，例如 **Python 3.11.4**：

<img src="./images/CANNLab_env_experience_guide/png8.png">

内核选择完成后，可以看到 Notebook 右上角已显示 **Python 3.11.4**。此时点击 code cell 左侧的运行按钮，即可正常执行单元格代码，code cell 执行成功后，页面会显示对应的运行结果：

<img src="./images/CANNLab_env_experience_guide/png9.png">