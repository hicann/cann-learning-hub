# CANNJudge 算子提交 Skill

支持邮箱/手机号与密码登录（含图形验证码）、会话复用，以及传统算子和 `npu_kernel_dev` 核函数工程的下载、提交、查询。

## 快速开始

新登录前，Agent 先询问登录方式：**Agent 对话输入账号密码、Linux/本地终端隐藏输入、RSA 密文**。已有有效会话直接复用，已选择方式不重复询问。

选择 Agent 对话方式后提供账号密码，Agent 用运行时变量调用 `client.login(account, password)` 并 `client.save_session()`；不回显密码、不写入脚本或日志。对话输入不隐藏，可改选终端方式。

**Linux / SSH终端**：使用 `bash /path/to/cannjudge-submit/login.sh`，账号和密码在终端输入，密码隐藏。默认直接调用登录接口，不需要图片或二维码。

```bash
python3 -m pip install requests
python3 cannjudge_cli.py login
python3 cannjudge_cli.py login --account "your@email.com" --login-type email
```

`--email` 是 `--account` 的兼容别名。仅在明确需要验证码时添加 `--captcha`；SVG 原样保存在 Agent 当前工作目录。
详细步骤见 [Linux登录说明](references/linux-login.md)。Windows会话不能复制到Linux使用。
会话保存在 `~/.cannjudge/session.json`，Windows DPAPI加密，POSIX权限600，仅保存用户ID/Cookie/登录时间。
不保存密码，不从密码环境变量自动登录。`logout`清除本机会话；`--session-file PATH`（子命令前）可隔离账号。

## 下载、开发与提交

```bash
python3 cannjudge_cli.py info --problem-url "https://cannjudge.cn/hc_cann_codelabs/cann/add"
python3 cannjudge_cli.py download --problem-url "https://cannjudge.cn/hc_cann_codelabs/cann/add" --output ./add-work
# 完成可编辑源码后预检
python3 cannjudge_cli.py submit --problem-url "https://cannjudge.cn/hc_cann_codelabs/cann/add" --project-dir ./add-work/project --dry-run
# 提交，不重复确认已授权的操作
python3 cannjudge_cli.py submit --problem-url "https://cannjudge.cn/hc_cann_codelabs/cann/add" --project-dir ./add-work/project --no-wait
python3 cannjudge_cli.py query --submission-id ID
python3 cannjudge_cli.py rank --problem-id ID
```

所有题目选项也支持 `--problem-id ID`。完整链接会保留小组/比赛作用域，避免同名题冲突。
默认 `--project-type auto` 依据网站的 `code_template` 分流；可显式指定 `npu_kernel_dev` 或 `registry`。
下载器保留完整ZIP和工程文件，并生成不含凭据的模板清单；现有工程不会被覆盖。

- 核函数工程：读取在线模板的 editable 文件，以及根目录新增的合法 `.asc`/`.h`，以 `files` 数组提交。
- 传统工程：读取 `op_kernel/*_tiling.h`、可选 `tiling_key_*.h`、`op_host/*.cpp`、`op_kernel/*.cpp`，使用原有四字段提交。匹配多个文件会停止，避免误选。
- `--dry-run` 显示拟提交文件大小与SHA-256，不发送源码到提交接口。
- 只读模板被修改、源码缺失、题目绑定不一致或路径越界会停止。

详细核函数接口约束见 [kernel-project.md](references/kernel-project.md)。

## SDK

```python
from cannjudge_cli import CANNJudgeClient, print_submission_result

client = CANNJudgeClient()
if not client.load_session():
    raise RuntimeError("请先在自己的终端运行 cannjudge_cli.py login")
problem = client.resolve_problem("https://cannjudge.cn/hc_cann_codelabs/cann/add")
folder = client.download_package(problem['_id'], './fresh-output', include_metadata=True)
# 完成 folder 中可编辑源码，再执行：
payload = client.prepare_submission(problem['_id'], folder)
submission_id = client.submit_payload(payload)
print_submission_result(client.wait_for_result(submission_id))
client.save_session()
```

旧SDK `submit(problem_id, tiling_h, tiling_key_h, host_cpp, kernel_cpp)` 保留。
`login_interactive(account)` 用于用户自己的交互终端；登录后SDK显式 `save_session()`。
现有系统可调用 `login(account, password)`，不要输出密码或完整响应。

## API

| 操作 | 接口 |
|---|---|
| 验证码 | `GET /api/users/captcha` → captchaId、image（服务端原始SVG，本地原样保存） |
| 登录 | `POST /api/users/login`，account/loginType/password（captchaId/captchaCode 为可选配套字段） |
| 题目 | `GET /api/problems/{id}` |
| 按名查题 | `GET /api/problems/name/{name}?contestId=...` |
| 模板清单 | `GET /api/problems/{id}/template?userId=...` |
| 下载ZIP | `GET /api/problems/{id}/package?userId=...` |
| 提交 | `POST /api/submissions/submit`，problemId/userId/files 或原四字段 |
| 查询 | `GET /api/submissions/{id}` |
| 最新排行 | `GET /api/submissions/problem/{id}/latest` |

成功响应含 `data.submissionId`。当前平台成功终态 `Pass`，兼容 `Accepted`，每个测试点读取 testcase_status、precision_ratio、time、msg。
时间展示平台原值，不臆测单位。默认等待120秒；`--max-wait` 可调整，`--no-wait` 后使用提交ID继续查询。
网络超时或没有提交ID时，先查询已有提交记录，不自动重复POST。401清理会话；403先检查题目/小组权限。

## RSA 兼容方式

见 [rsa-login.md](references/rsa-login.md)。`login_with_ciphertext` 接受与 `login` 相同的关键字登录选项。
CLI已有验证码可使用 `--captcha-id` 和 `--captcha-code`，由用户读取验证码；普通密码登录默认不带验证码。
`--password` 仅保留兼容，会提醒命令历史暴露风险；不建议用户使用。

## 验证

```bash
python3 -m pytest tests -q
```

覆盖密码隐藏输入、会话持久化/失效、邮箱/手机号验证码协议、传统工程兼容、核函数多文件提交、只读保护、路径安全、题目作用域和Pass终态。
真实端到端验证：上述Add题核函数工程，提交 `6aa27abe2d3dd2c5ae243b01`，平台 `Pass`，15/15 测试点通过（2026-09-10）。
