# CANNJudge 算子提交 Skill

帮助用户完成 CANNJudge 算子竞赛的完整流程。

## 功能

- ✅ 登录 CANNJudge
- ✅ 获取题目信息
- ✅ 下载工程模板
- ✅ 提交算子实现
- ✅ 查询提交结果
- ✅ 查看排行榜

## 使用方法

### 方式 1: 使用命令行工具

```bash
# 设置登录信息（可选，也可在命令中指定）
export CANNJUDGE_EMAIL="your_email@example.com"
export CANNJUDGE_PASSWORD="your_password"

# 登录
python cannjudge_cli.py login --email "your_email" --password "your_password"

# 下载工程
python cannjudge_cli.py download --problem-id "题目ID" --output "./output"

# 提交代码
python cannjudge_cli.py submit --problem-id "题目ID" --project-dir "./output/project"

# 查询结果
python cannjudge_cli.py query --submission-id "提交ID"

# 查看排行榜
python cannjudge_cli.py rank --problem-id "题目ID"
```

### 方式 2: 使用示例脚本

```bash
python example.py
```

按照提示逐步完成整个流程。

### 方式 3: 在代码中直接使用

```python
from cannjudge_cli import CANNJudgeClient

# 创建客户端
client = CANNJudgeClient()

# 登录
user_info = client.login("email@example.com", "password")

# 获取题目信息
problem = client.get_problem("题目ID")

# 下载工程
extract_dir = client.download_package("题目ID", "./output")

# 提交代码
submission_id = client.submit(
    problem_id="题目ID",
    tiling_h="...",
    tiling_key_h="...",
    host_cpp="...",
    kernel_cpp="..."
)

# 等待结果
result = client.wait_for_result(submission_id)

# 查看排行榜
rankings = client.get_rankings("题目ID")
```

## API 接口说明

### 登录
```
POST /api/users/login
Body: {"email": "...", "password": "..."}
返回: {"_id": "用户ID", "nickname": "...", ...}
```

### 获取题目
```
GET /api/problems/{problemId}
返回: {"_id": "...", "name": "...", "desc": "...", ...}
```

### 下载工程
```
GET /api/problems/{problemId}/package?userId={userId}
返回: Zip 文件
```

### 提交代码
```
POST /api/submissions/submit
Body: {
  "problemId": "...",
  "userId": "...",
  "tiling_h": "...",
  "tiling_key_h": "...",
  "host_cpp": "...",
  "kernel_cpp": "..."
}
返回: {"code": 0, "data": {"submissionId": "..."}}
```

### 查询结果
```
GET /api/submissions/{submissionId}
返回: {"status": "...", "result": [...], ...}
```

### 排行榜
```
GET /api/submissions/problem/{problemId}/latest
返回: [用户提交列表]
```

## 提交结果说明

### 状态
- `Running`: 正在执行
- `Accepted`: 全部通过
- `Wrong Answer`: 部分失败
- `Compile Error`: 编译失败
- `Runtime Error`: 运行时错误

### 测试用例结果
- `testcase_status`: 测试用例状态
- `precision_ratio`: 精度比例 (1.0 = 完全匹配)
- `time`: 执行时间 (毫秒)

## 注意事项

1. **敏感信息**: 不要在代码中硬编码账号密码
2. **Cookie**: 登录后会自动管理 Cookie
3. **轮询**: 查询结果时默认 3 秒间隔
4. **超时**: 默认最长等待 120 秒

## 工程模板结构

```
code/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── {op_name}.cpp
└── op_kernel/
    ├── CMakeLists.txt
    ├── {op_name}_tiling.h
    ├── tiling_key_{op_name}.h
    └── {op_name}.cpp
```

## 相关资源

- CANNJudge 网站: https://cannjudge.cn
- CANN 文档: https://www.hiascend.com/document
