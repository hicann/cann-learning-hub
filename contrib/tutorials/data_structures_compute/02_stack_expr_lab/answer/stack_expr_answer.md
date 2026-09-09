# 栈表达式求值实验 — 参考答案

## 实践 1：补全 BracketMatchLite 的配对检查逻辑

### 任务要求
补全括号匹配算子中右括号与栈顶左括号的配对检查。

### 参考思路
1. 右括号到來时，检查栈是否为空（失配①）
2. Pop 栈顶左括号，检查类型是否匹配（失配②）
3. 扫描结束后检查栈是否为空（失配③）

### 关键代码
```bash
cat answer/bracket_match_answer.cpp
```

---

## 实践 2：实现 SuffixEvalLite 的运算逻辑

### 任务要求
实现后缀表达式求值中运算符的处理：Pop 两个操作数，执行运算，Push 结果。

### 参考思路
1. token >= 0 → 操作数，Push 入栈
2. token < 0 → 运算符，Pop b、Pop a、计算 a op b、Push 结果
3. 最终栈底元素为结果

### 关键代码
```bash
cat answer/suffix_eval_answer.cpp
```

---

## 实践 3：验证 InfixToPostfixLite 的转换结果

### 任务要求
运行 benchmark，确认中缀转后缀结果正确。

### 验证方法
```bash
cat answer/expected_output.txt
```

---

## 课后测试答案

选择题与填空题答案见 [`chapter_test_answer.md`](./chapter_test_answer.md)。

---

## 关键技术要点

### 栈空间声明

在 Ascend C Kernel 中，栈数组声明为 Kernel 类的成员变量：

```cpp
class KernelBracketMatchLite {
private:
    char ubStack[MAX_STACK_SIZE];  // 编译器自动分配到 Local Memory
    uint32_t top = 0;
};
```

> **注意**：`__ubuf__` 关键字在 CANN 9.0+ 中仅用于指针类型转换，不能用于数组声明。

### Tiling 参数传递

TilingFunc 中通过输入 shape 计算每个核的处理长度：

```cpp
uint32_t blockLength = totalLen / blockDim;
tiling->exprLength = blockLength;
```

### 多算子工程生成

JSON 中包含多个算子时，需使用 `-op` 参数逐个生成：

```bash
msopgen gen -i ops.json -c ai_core-ascend910_93 -op BracketMatchLite -m 0
msopgen gen -i ops.json -c ai_core-ascend910_93 -op SuffixEvalLite -m 1
msopgen gen -i ops.json -c ai_core-ascend910_93 -op InfixToPostfixLite -m 1
```
