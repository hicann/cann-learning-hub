# npu_kernel_dev 核函数工程

已按 CANNJudge 2026-09-10 的页面代码和 Add 题实际下载、提交、评测验证。
示例：https://cannjudge.cn/hc_cann_codelabs/cann/add 。具体文件与签名每次重新读取模板。

## 下载与识别

- `GET /api/problems/{id}`：`code_template == "npu_kernel_dev"`。
- `GET /api/problems/{id}/template?userId=...`：`data.files`，每项有 path、content、editable。
- `GET /api/problems/{id}/package?userId=...`：ZIP。下载全部工程，保留构建和本地测试文件。

Add 示例根目录：

```text
CMakeLists.txt         只读构建配置
kernel.asc            可编辑核函数及run_kernel入口
main.asc              只读本地调用程序
data_utils.h          只读I/O辅助代码
run.sh                只读运行脚本
scripts/              只读数据生成、golden与验证程序
```

`kernel.asc` 被平台主程序包含，不新增 main()、include guard 或 pragma once。
TensorInfo/TensorGroupInfo 由框架预定义。Add 的接口为：

```cpp
extern "C" void run_kernel(
    GM_ADDR x1, const TensorGroupInfo& info_x1,
    GM_ADDR x2, const TensorGroupInfo& info_x2,
    GM_ADDR y, const TensorGroupInfo& info_y,
    int64_t availableCoreNum, aclrtStream stream);
```

这只是 Add 的接口示例，其他题目的输入/输出/属性可能不同。
dtype 编码来自模板，例如 0=float32、1=float16、5=int32；不能套用 ge::DataType 数值。
Host 使用 shape/dtype/availableCoreNum 做切分，在传入 stream 上启动实际NPU核函数；不从CPU读取张量数据代算。

## 提交文件规则

提交 `files` 数组，每项仅包含相对 `path` 和 UTF-8 `content`。
读取当次在线模板的所有 editable 文件，可额外加入根目录合法 `.asc`、`.h` 辅助文件。
文件名1～128字符，字母/数字/下划线开头，其余可含字母、数字、下划线、连字符和点。
禁止额外提交 judge.asc、main.asc、data_utils.h；排除构建、测试、日志、二进制和会话凭据。
已有只读文件若修改，CLI停止而非偷偷忽略；保存原始模板并恢复相应文件后再试。
工程元数据中的problem_id与目标不同时停止，防止误投同名题。

```json
{
  "problemId": "题目ID",
  "userId": "当前会话用户ID",
  "files": [
    {"path": "kernel.asc", "content": "源码内容"},
    {"path": "add_tiling.h", "content": "辅助头文件内容"}
  ]
}
```

发送到 `POST /api/submissions/submit`，响应 `data.submissionId`。
核函数工程不需要伪造 op_host/op_kernel 目录，也不应塞进旧 kernel_cpp 字段。

## SDK 示例

```python
from cannjudge_cli import CANNJudgeClient

client = CANNJudgeClient()
client.load_session()
problem = client.resolve_problem("https://cannjudge.cn/hc_cann_codelabs/cann/add")
directory = client.download_package(problem['_id'], './fresh-output', include_metadata=True)
# 完成可编辑源码，随后：
payload = client.prepare_submission(problem['_id'], directory)
submission_id = client.submit_payload(payload)
result = client.wait_for_result(submission_id)
client.save_session()
```

只有真正返回 Pass/Accepted 且所有测试点通过才报告完成。保存提交ID可断点续查，未知结果不重复POST。
Add 模板本地 main.asc 只运行首个公开样例；平台15点通过与额外边界验证须分别说明。
