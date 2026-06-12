# CANNLab 环境体验指南

## 一、引言

CANNLab 提供两种即开即用的 NPU 体验环境。打开 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub) 仓库后，可在仓库页面看到 **CANNLab** 按钮：

<img src="./images/CANNLab.png">

将鼠标移至 **CANNLab** 图标，会弹出 **云开发** 和 **950 尝鲜体验** 两个选项。

<img src="./images/CANNLab2.png" alt="CANNLab2"  width="250px">

请根据要体验的教程 `README.md` 中说明的支持环境，选择对应的运行环境。

- **云开发环境**：可申请创建 A2/A3 环境。
- **950 尝鲜体验环境**：可申请创建 A5 环境。

点击对应选项后，可使用华为云账号登录并进入开发者空间。当前云开发环境支持体验本仓库中的 `.ipynb` 教程文件。本指南将介绍如何基于 CANNLab 云开发环境体验 `cann-learning-hub` 仓库中的 `.ipynb` 教程。

---

## 二、云开发环境教程体验指南

### 2.1 创建并登录环境

根据上一小节的操作，点击 **云开发** 选项，使用华为云账号登录并进入开发者空间。进入页面后，点击 **创建** 按钮：

<img src="./images/create.png">

云开发环境支持创建 CPU 环境和 NPU 环境。这里以在 A2 环境上体验 Ascend C 系列课程为例，选择创建 NPU 环境，规格配置如下：

- **开发环境名称**：自行命名。
- **处理器类型**：昇腾 NPU。
- **模板名称**：`cann_master-py3.12-A2-arm-20260514`。
- **规格**：`1*NPU 910B3 16vCPUs 32GiB`。

<img src="./images/create_npu.png" alt="create_npu"  width="500px">

点击 **创建** 按钮后，可以看到 NPU 环境已创建。点击 **开机** 按钮启动环境：

<img src="./images/start.png">

> 注意：如果开机时提示资源不足，说明当前时间段使用人数较多，可稍后再尝试。

环境成功开机后，即可通过 WebIDE 或 VS Code 连接使用。这里以 WebIDE 连接方式为例，点击 **WebIDE** 进入环境：

<img src="./images/webIDE.png">

使用 **WebIDE** 连接进入环境后，可以看到整体界面类似 VS Code，支持源代码管理、扩展安装等操作：

<img src="./images/windows.png">

### 2.2 安装 Jupyter 扩展

在 VS Code / CodeArts IDE 中打开 `.ipynb` 文件并执行 Jupyter Notebook 的 code cell，一般需要具备以下扩展：

- **Python 扩展**：用于识别 Python 环境和解释器。
- **Jupyter 扩展**：用于打开和运行 `.ipynb` 文件，并管理 kernel。
- **Jupyter Notebook Renderers 扩展**：用于渲染表格、图片、Markdown、HTML、图表等输出内容。

当前环境已安装 **Python 扩展** 和 **Jupyter Notebook Renderers 扩展**，还需要手动安装 **Jupyter 扩展**。点击左侧菜单栏中的 **扩展**，搜索 **Jupyter**，然后点击 **安装**：

<img src="./images/jupyter_install.png">

### 2.3 体验教程

完成扩展安装后，即可打开仓库中的教程文件进行体验。下面以 `cann-learning-hub/tutorials/ascendc_operator_development/02_AscendC_basic/02.02_HelloWorld.ipynb` 为例：

点击窗口左侧菜单栏中的 **资源管理器**，在仓库目录中找到并打开 **02.02_HelloWorld.ipynb** 教程文件。打开后，在 Notebook 右上角点击 **选择内核**，选择 Python Kernel，例如 **Python 3.12.9**：

<img src="./images/select_kernel.png">

内核选择完成后，可以看到 Notebook 右上角已显示 **Python 3.12.9**。此时点击 code cell 左侧的运行按钮，即可正常执行单元格代码：

<img src="./images/run_code_cell.png" >

code cell 执行成功后，页面会显示对应的运行结果：

<img src="./images/run_code_cell_success.png">