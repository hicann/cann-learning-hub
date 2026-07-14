# Vector 算子开发示例代码

本目录下的代码由对应的 Notebook（`03.02_simd_operator_development.ipynb`）通过 `%%writefile` 动态生成到 `Sources/` 目录。

## 编译运行

打开 `03.02_simd_operator_development.ipynb`，按顺序点击 ▶ 执行各 Cell：

1. 环境初始化
2. 写入 membase 代码 → 编译 → 运行
3. 写入 regbase 代码 → 编译 → 运行

## FastGelu 算子

计算公式：`y = x / (1 + exp(-1.702 * x))`

**SIMD 指令分解**（5 条指令）：
1. `Muls` — 标量乘：`1.702 * x`
2. `Neg` — 取反：`-1.702 * x`
3. `Exp` — 指数：`exp(-1.702 * x)`
4. `Adds` — 加一：`1 + exp(-1.702 * x)`
5. `Div` — 除法：`x / (1 + exp(-1.702 * x))`
