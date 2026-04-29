# Skills - CANNBot 技能集合

本目录包含用于 CANN 算子开发和 CANNJudge 竞赛的技能模块。

## 技能列表

### 1. cannjudge-submit - CANNJudge 算子提交技能

**功能描述**：帮助用户完成 CANNJudge 算子竞赛的完整流程，包括下载工程、理解题目、实现泛化算子、提交代码、获取结果。

**触发场景**：
- 参加 CANNJudge 算子竞赛
- 从 CANNJudge 下载算子工程模板
- 提交算子实现到 CANNJudge
- 查看 CANNJudge 排行榜

**核心功能**：
- 登录 CANNJudge 平台
- 获取题目信息
- 下载工程模板
- 指导泛化算子设计
- 提交代码实现
- 查询提交结果
- 查看排行榜

**使用示例**：
```
用户: 帮我从 CANNJudge 下载 DepthToSpace 题目并实现泛化算子

助手执行流程:
1. 登录 CANNJudge 平台
2. 获取题目信息和算子原型
3. 下载工程模板
4. 分析泛化需求（支持的 dtype、维度、属性等）
5. 指导用户实现泛化算子逻辑
6. 编译并提交
7. 查看结果和排行榜
```

**重要说明**：
- **平台不开放测试用例 API，必须设计泛化算子**
- 所有题目信息必须从 CANNJudge 网站获取，禁止使用本地信息文件
- 算子必须具备泛化性，能够处理各种 shape、dtype、属性组合
- 算子实现的技术细节参考 `ascendc-ops-project` skill
- 支持的提交状态：Running、Accepted、Wrong Answer、Compile Error、Runtime Error

**泛化设计核心**：
- **Shape 泛化**：支持任意维度和大小，不硬编码维度数
- **Dtype 泛化**：使用模板类支持多种数据类型
- **属性泛化**：处理所有可能的属性值、默认值、边界值
- **对齐泛化**：处理对齐和非对齐情况，使用 DataCopyPad
- **边界泛化**：处理空输入、单元素、极端值等边界情况

**文件结构**：
```
cannjudge-submit/
├── SKILL.md           # 技能详细说明文档
├── README.md          # 使用说明
├── cannjudge_cli.py   # 命令行工具
└── example.py         # 使用示例
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
│   ├── op_host_template.cpp         # Host 侧实现模板
│   ├── op_kernel_template.cpp       # Kernel 侧实现模板
│   ├── tiling_header_template.h     # Tiling 数据结构模板
│   ├── test_template.cpp            # 测试代码模板
│   ├── build.sh.template            # 编译脚本模板
│   └── CMakeLists.txt.template      # CMakeLists.txt 模板
└── examples/                         # 完整示例
    ├── add_custom_example.md        # AddCustom 示例
    └── clamp_example.md             # Clamp 示例
```

---

## 使用方法

### 方法一：直接使用技能文档

每个技能目录下的 `SKILL.md` 文件包含详细的使用说明和示例代码，可以直接阅读参考。

### 方法二：在 CANNBot 中使用

如果您使用 CANNBot 智能助手，这些技能会被自动加载，您可以直接向助手提问：

```
用户: 帮我创建一个 AddCustom 算子工程
助手: [自动使用 ascendc-ops-project skill 提供指导]

用户: 帮我提交算子到 CANNJudge
助手: [自动使用 cannjudge-submit skill 处理提交流程]
```

---

## 技能依赖关系

`cannjudge-submit` 依赖 `ascendc-ops-project`：

1. **cannjudge-submit**：处理 CANNJudge 平台交互（登录、下载、提交、查询）
2. **ascendc-ops-project**：提供算子实现的技术细节（Tiling 设计、Host/Kernel 实现）

使用流程：
1. 使用 `cannjudge-submit` 下载题目和工程模板
2. 使用 `ascendc-ops-project` 指导算子实现
3. 使用 `cannjudge-submit` 提交代码和查看结果

---

## 前置要求

### 通用要求
- 已安装 CANN 开发环境
- 已配置编译工具链

### cannjudge-submit 额外要求
- 已注册 CANNJudge 账号（https://cannjudge.cn）
- 已安装 Python 3.x 和 requests 库

### ascendc-ops-project 额外要求
- 已安装 msopgen 工具
- 已配置 NPU 设备驱动

---

## 常见问题

### Q1: 为什么平台不提供测试用例？
A: CANNJudge 平台不开放测试用例 API，目的是让开发者设计泛化算子，而不是针对特定测试用例优化。泛化算子可以处理所有可能的输入情况，具有更好的通用性和鲁棒性。

### Q2: 如何设计泛化算子？
A: 参考 `cannjudge-submit/SKILL.md` 中的"泛化设计核心要素"和检查清单，以及 `ascendc-ops-project/SKILL.md` 中的模板实现方法。核心是：动态处理 shape、使用模板支持多 dtype、处理所有属性值和边界情况。

### Q3: 如何获取 CANNJudge 题目信息？
A: 使用 `cannjudge-submit` skill，所有题目信息必须从 CANNJudge 网站获取，禁止使用本地信息文件。

### Q4: 算子原型定义有什么要求？
A: 所有输入/输出/属性必须与原型完全一致，详见 `ascendc-ops-project/SKILL.md` 中的"铁律"部分。

### Q5: 如何处理多数据类型支持？
A: 使用模板类实现，在 Kernel 入口函数中根据 dtype 参数选择对应的模板实例，详见 `ascendc-ops-project/SKILL.md` 的 Part 5。

### Q6: Float16 精度问题如何处理？
A: 将 Float16 转换为 Float32 进行中间计算，再转换回 Float16，详见 `ascendc-ops-project/SKILL.md` 的 2.3 节。

### Q7: 为什么我的算子在部分测试用例上失败？
A: 可能是算子不够泛化。检查是否处理了所有泛化场景：任意维度、所有 dtype、边界值、非对齐情况等。参考 `cannjudge-submit/SKILL.md` 的泛化设计检查清单。

---

## 更新日志

- 2024-04-24: 更新 skill，强调算子泛化性设计，移除测试用例相关内容
- 2024-04-24: 初始提交，包含 cannjudge-submit 和 ascendc-ops-project 两个技能

---

## 许可证

本技能集合遵循 CANN 社区开源协议。

## 联系方式

如有问题或建议，请在 GitCode 仓库提交 Issue。
