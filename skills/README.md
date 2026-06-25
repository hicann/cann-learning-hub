# Skills - CANNBot 技能集合

本目录包含用于练习和竞赛相关技能模块。

---

## 快速开始

### 第一步：打开云开发环境

1. 访问 [cann-learning-hub](https://www.gitcode.com/cann/cann-learning-hub)，登录 GitCode
2. 点击 **「云开发」** 或 **「Colab」** 按钮，进入在线开发环境
3. 在终端中执行：

```bash
git clone https://gitcode.com/cann/cann-learning-hub.git
cd cann-learning-hub
opencode
```

### 第二步：与 CANNBot 对话

在 opencode 对话界面中输入：

```
帮我生成 erf 算子
```

CANNBot 会引导你完成整个开发流程。

### 两种使用模式

#### 模式 A：半自动（推荐入门）

自己从 CANNJudge 网站下载算子空工程，使用 CANNBot 生成算子代码，自己手工提交到 CANNJudge。

```
1. 从 CANNJudge 网站下载算子空工程模板
2. 在 CANNBot 中对话: "帮我生成 xxx 算子"
3. CANNBot 自动: 设计→开发→编译→测试
4. 自己手工将代码提交到 CANNJudge
```

无需配置任何登录信息，开箱即用。

#### 模式 B：全自动（进阶）

配置 RSA 加密密文后，CANNBot 可自动完成从下载到提交的全流程。

**首次设置 RSA 密文**：

CANNBot 使用 RSA 加密保护你的 CANNJudge 密码，**禁止在对话中输入明文密码**。

##### 1. 在云开发终端生成 RSA 密钥对

```bash
cd skills/cannjudge-submit
python3 generate_key.py
```

输出示例：
```
RSA-2048 密钥对已生成
私钥已保存到: private.pem  (仅保留在服务器，绝不外传)
公钥已保存到: public.pem  (拷贝到个人 PC 用于加密)
```

> `private.pem` 留在云开发服务器上，**不要拷贝出去**。`public.pem` 需要拷贝到你的个人电脑。

##### 2. 将公钥拷贝到个人电脑

**方式 A**：在云开发左侧文件浏览器中，找到 `public.pem`，右键点击 **「下载」**，保存到个人电脑。

**方式 B**：在云开发终端查看公钥内容，手动复制：
```bash
cat public.pem
```
将输出的内容保存为 `public.pem` 文件到你的个人电脑。

##### 3. 在个人电脑上加密密码

确保你的个人电脑已安装 Python 3 和 pycryptodome：

```bash
pip install pycryptodome
```

将 `encrypt_password.py` 也拷贝到个人电脑（或直接下载），然后运行：

```bash
python3 encrypt_password.py --public-key public.pem
```

按提示输入你的 CANNJudge 密码，脚本会输出一长串 **RSA 密文**。

> **密文可以复用**：只要服务器上的 `private.pem` 不变，同一密文可以反复使用，无需每次重新加密。建议保存到安全位置。

##### 4. 使用密文登录

当 CANNBot 需要登录 CANNJudge 时，会询问你的 **邮箱** 和 **RSA 密文**，将它们粘贴给 CANNBot 即可。

> **安全铁律**：CANNBot 只接受密文，解密后的密码仅在内存中用于调用登录 API，**绝不会显示在对话中**。

### 全自动流程示意

```
┌─────────────────────────────────────────────────────┐
│  GitCode 云开发                                      │
│                                                      │
│  1. clone cann-learning-hub                          │
│  2. cd cann-learning-hub && opencode                 │
│  3. 首次: python3 generate_key.py → 生成密钥对        │
│  4. 拷贝 public.pem 到个人 PC                         │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  个人 PC                                      │    │
│  │  5. pip install pycryptodome                  │    │
│  │  6. python3 encrypt_password.py → 得到密文     │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  7. 在 opencode 中对话: "帮我生成 xxx 算子"           │
│  8. 按提示输入邮箱 + 密文                             │
│  9. CANNBot 自动: 登录→下载→设计→开发→编译→提交→查结果 │
└─────────────────────────────────────────────────────┘
```

---

## 技能列表

### 1. cannjudge-submit - CANNJudge 算子提交技能

**功能描述**：帮助用户完成 CANNJudge 平台上的算子开发全流程，包括下载工程、理解题目、实现泛化算子、提交代码、获取结果。

**触发场景**：
- 从 CANNJudge 平台下载算子工程模板
- 提交算子实现到 CANNJudge
- 查看 CANNJudge 提交结果和排行榜

**核心功能**：
- RSA 加密安全登录（禁止明文密码）
- 获取题目信息
- 下载工程模板
- 指导泛化算子设计
- 提交代码实现
- 查询提交结果
- 查看排行榜

**安全铁律**：
- **禁止在对话中直接输入明文密码**
- **禁止将解密后的密码显示在对话中**，仅用于内部调用登录 API
- **私钥 (private.pem) 绝不外传**

**泛化设计核心**：
- **Shape 泛化**：支持任意维度和大小，不硬编码维度数
- **Dtype 泛化**：使用模板类支持多种数据类型
- **属性泛化**：处理所有可能的属性值、默认值、边界值
- **对齐泛化**：处理对齐和非对齐情况，使用 DataCopyPad
- **边界泛化**：处理空输入、单元素、极端值等边界情况

**文件结构**：
```
cannjudge-submit/
├── SKILL.md              # 技能详细说明文档
├── README.md             # 使用说明
├── cannjudge_cli.py      # 命令行工具（支持 RSA 密文登录）
├── generate_key.py       # RSA 密钥对生成脚本（服务器端运行）
├── encrypt_password.py   # RSA 密码加密脚本（PC 端运行）
└── example.py            # 使用示例
```

---

### 2. ascendc-ops-project - Ascend C 算子工程化开发技能

**功能描述**：提供从工程创建、编译打包、安装部署到 aclnn API 测试的完整流程，包含 Tiling 模板编程、属性、TBuf、Workspace 使用。

**触发场景**：
- 从零创建标准算子工程
- 算子编译、打包、安装部署
- 完整开发流程指导（设计→实现→测试→部署）
- 使用 aclnn 二段式接口进行测试验证
- Tiling 模板编程、属性、TBuf、Workspace 使用

**核心功能**：
- 算子原型定义和工程生成
- Host 侧实现（Tiling、InferShape、InferDataType）
- Kernel 侧实现（模板类、多数据类型支持）
- 测试代码编写和验证
- 编译打包安装流程

**开发流程**：
```
需求分析 → 原型定义 → 工程生成 → Tiling设计 → Host实现 → Kernel实现 → 编译安装 → 测试验证
```

**重要铁律**：
- **严格按照算子原型开发**：输入/输出/属性必须与原型完全一致
- **输入不能当属性**：输入是运行时数据，属性是编译时常量
- **属性不能当输入**：属性值在编译时确定，不能动态变化
- **不能增删输入输出**：必须与原型完全一致
- **不能修改数据类型**：dtype 必须与原型一致

**文件结构**：
```
ascendc-ops-project/
├── SKILL.md                          # 技能详细说明文档
├── references/                       # 参考文档
│   ├── api_best_practices.md        # API 最佳实践
│   ├── tiling_design.md             # Tiling 设计指南
│   ├── precision_standard.md        # 精度验证标准
│   └── cmake_guide.md               # CMake 配置指南
├── templates/                        # 代码模板
│   ├── host_template.cpp            # Host 侧实现模板
│   ├── kernel_template.cpp          # Kernel 侧实现模板
│   ├── op_host_template.cpp         # Host 侧实现模板
│   ├── op_kernel_template.cpp       # Kernel 侧实现模板
│   ├── tiling_header_template.h     # Tiling 数据结构模板
│   ├── tiling_key_template.h        # TilingKey 模板
│   ├── tiling_template.h            # Tiling 模板
│   ├── test_main.cpp.template       # 测试主函数模板
│   ├── test_template.cpp            # 测试代码模板
│   ├── build.sh.template            # 编译脚本模板
│   ├── CMakeLists.txt.template      # CMakeLists.txt 模板
│   └── CMakePresets.json.template   # CMakePresets 模板
└── examples/                         # 完整示例
    ├── add_custom_example.md        # AddCustom 示例
    └── clamp_example.md             # Clamp 示例
```

---

## 技能依赖关系

`cannjudge-submit` 依赖 `ascendc-ops-project`：

1. **cannjudge-submit**：处理 CANNJudge 平台交互（登录、下载、提交、查询）
2. **ascendc-ops-project**：提供算子实现的技术细节（Tiling 设计、Host/Kernel 实现）

使用流程：
1. **模式 A**：自己从 CANNJudge 下载工程 → 使用 `ascendc-ops-project` 指导算子实现 → 自己手工提交
2. **模式 B**：使用 `cannjudge-submit` 下载题目和工程模板 → 使用 `ascendc-ops-project` 指导算子实现 → 使用 `cannjudge-submit` 提交代码和查看结果

---

## 前置要求

### 通用要求
- 已安装 CANN 开发环境
- 已配置编译工具链

### cannjudge-submit 额外要求
- 已注册 CANNJudge 账号（https://cannjudge.cn）
- 已安装 Python 3.x、requests
- 全自动模式（模式 B）额外需要：pycryptodome

### ascendc-ops-project 额外要求
- 已安装 msopgen 工具
- 已配置 NPU 设备驱动

---

## 常见问题

### Q1: 密文太长，每次都要重新输入吗？（模式 B）
A: 不需要。只要服务器上的 `private.pem` 没有重新生成，同一密文可以反复使用。建议将密文保存到安全位置（如密码管理器）。

### Q2: 密钥对丢失了怎么办？（模式 B）
A: 在服务器上重新运行 `python3 generate_key.py`（先删除旧的 `private.pem` 和 `public.pem`），然后在 PC 上重新加密密码。旧密文将失效。

### Q3: 为什么平台不提供测试用例？
A: CANNJudge 平台不开放测试用例 API，目的是让开发者设计泛化算子，而不是针对特定测试用例优化。泛化算子可以处理所有可能的输入情况，具有更好的通用性和鲁棒性。

### Q4: 如何设计泛化算子？
A: 参考 `cannjudge-submit/SKILL.md` 中的"泛化设计核心要素"和检查清单，以及 `ascendc-ops-project/SKILL.md` 中的模板实现方法。核心是：动态处理 shape、使用模板支持多 dtype、处理所有属性值和边界情况。

### Q5: 如何获取 CANNJudge 题目信息？
A: 使用 `cannjudge-submit` skill，所有题目信息必须从 CANNJudge 网站获取，禁止使用本地信息文件。

### Q6: 算子原型定义有什么要求？
A: 所有输入/输出/属性必须与原型完全一致，详见 `ascendc-ops-project/SKILL.md` 中的"铁律"部分。

### Q7: Float16 精度问题如何处理？
A: 将 Float16 转换为 Float32 进行中间计算，再转换回 Float16，详见 `ascendc-ops-project/SKILL.md` 的 2.3 节。

### Q8: 为什么我的算子在部分测试用例上失败？
A: 可能是算子不够泛化。检查是否处理了所有泛化场景：任意维度、所有 dtype、边界值、非对齐情况等。参考 `cannjudge-submit/SKILL.md` 的泛化设计检查清单。

---

## 更新日志

- 2026-05-01: 新增 RSA 加密安全登录机制，替换明文密码登录
- 2024-04-24: 更新 skill，强调算子泛化性设计，移除测试用例相关内容
- 2024-04-24: 初始提交，包含 cannjudge-submit 和 ascendc-ops-project 两个技能

---

## 许可证

本技能集合遵循 CANN 社区开源协议。

## 联系方式

如有问题或建议，请在 GitCode 仓库提交 Issue。
