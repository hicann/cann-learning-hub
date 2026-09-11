---
name: cannjudge-submit
description: 登录 CANNJudge，按题目链接下载传统算子或 npu_kernel_dev 核函数工程，提交可编辑源码，查询评测结果和排行榜；支持邮箱/手机号密码、图形验证码、RSA 及会话复用。
---

# CANNJudge 下载、开发与提交

使用本目录的 `cannjudge_cli.py` 与 `CANNJudgeClient`，依赖 `requests`；只有 RSA 方式需要 `pycryptodome`。
首先从网站读取题目和模板，以当次题目的签名、dtype、广播、精度、CANN版本及只读文件为准。
不要用通用 Add 示例或其他题目规格替代当前题目。题目下载包中的公开数据可用于本地验证，不尝试访问隐藏用例。

## 登录选择

需要新登录时，**先询问登录方式**，等待用户选择后再收集对应凭据。提供：
1. Agent 对话输入账号密码：用户在当前对话提供邮箱/手机号和密码，Agent 直接调用 `client.login(account, password)`。
2. Linux/本地终端输入：用户在自己的交互终端输入账号及隐藏密码，运行 `bash /path/to/cannjudge-submit/login.sh`。
3. RSA 密文登录：使用用户提供的密文和私钥路径，详见 [references/rsa-login.md](references/rsa-login.md)。

询问示例：“请选择登录方式：在 Agent 对话中输入账号密码、在 Linux/本地终端隐藏输入，或使用 RSA 密文。”
用户已明确选择方式或主动提供账号密码时，视为选择完成，不重复询问。
已有有效会话时直接复用，不为继续下载或提交再次询问登录方式；CLI 自动加载，SDK 调用 `client.load_session()`。

### Agent 对话登录

选定此方式后再请求账号密码；说明对话输入不隐藏，终端方式可隐藏输入。不要回显密码，也不要将其写入源码、临时脚本、命令行参数、日志或交付包。
将用户提供的账号密码作为运行时值传给 SDK（使用执行工具的标准输入或已有内存变量传递，不能拼接到 shell 命令），不要打印完整登录响应。
仅向 CANNJudge 官方 HTTPS 登录接口发送凭据。无可用的运行时传递通道时说明限制，让用户改用终端方式，不能暗中写入文件。

```python
client = CANNJudgeClient()
client.logout()  # 显式重新登录时清除旧会话，失败不回退旧账号
client.login(account, password)
client.save_session()
```

`account`、`password` 是用户本次提供的运行时变量，示例不能替换成持久化的真实密码。
登录失败时报告不含凭据的错误，不自动重复尝试或切换登录方式。

### 终端与验证码选项

Linux/SSH 详细步骤见 [references/linux-login.md](references/linux-login.md)。默认直接请求 `/api/users/login`，不获取图片或二维码。

```bash
python3 cannjudge_cli.py login
python3 cannjudge_cli.py login --account "your@email.com"
```

`--email` 是 `--account` 的兼容别名，`--login-type email|phone` 可覆盖自动判断。
SDK `login(account, password)` 默认不带验证码；请求含 `account`、`loginType`、`password`，邮箱兼容字段 `email`，响应含 `_id`。
保留 `--captcha` 作为显式可选功能，仅在用户选择或服务端明确要求时使用，不默认推断网站必需验证码。
该选项获取原始 SVG，保存于 Agent 当前工作目录 `Path.cwd()`，名称为 `cannjudge-captcha-随机值.svg`，登录后保留；不做格式转换。
`--captcha-display file` 适用于SSH；`auto` 在无桌面环境只保存文件。用户自行读取验证码，助手不代读。
SDK 的 `captcha_id`、`captcha_code` 为配套可选参数。鉴权失败不自动切换验证码或绕过校验。
CLI `--password` 仅保留兼容，终端优先隐藏输入；不从 `CANNJUDGE_PASSWORD` 自动读取密码。

登录后仅保存用户 ID、Cookie、登录时间；不保存密码、密文或完整用户响应。
默认 `~/.cannjudge/session.json`，Windows 使用当前账号 DPAPI 加密，Linux/macOS 权限600。
同一机器和系统账号可跨目录复用，最长7天，服务端可能更早失效。Cookie也是凭据，不输出或随工程交付。
Windows DPAPI会话不能搬到Linux复用；在执行提交的Linux主机和系统账号下重新登录。
401清除本地会话；403区分题目/小组权限，不盲目重新提交。`logout` 只清除本机缓存。
多账号隔离使用位于子命令之前的 `--session-file PATH`。

## 获取题目与下载

```bash
python3 cannjudge_cli.py info --problem-url "https://cannjudge.cn/hc_cann_codelabs/cann/add"
python3 cannjudge_cli.py download --problem-url "https://cannjudge.cn/hc_cann_codelabs/cann/add" --output ./add-work
```

也支持 `--problem-id ID`。完整链接按小组、比赛、题目逐层解析，名称接口携带 contestId，避免同名题误选。
`info`、`download`、`submit --dry-run` 可访问公开资源而不强制本地登录，私有资源仍遵循服务端鉴权。
下载真实 `/api/problems/{id}/package` ZIP，同时读取 `/template` 保存 `.cannjudge-project.json` 元数据。
下载器拒绝路径越界、符号链接和覆盖现有工程；如需重新下载，选择新目录。

## 按工程类型开发和提交

以网站 `problem.code_template` 为分流依据；`--project-type auto` 为默认，不凭目录名推测题目类型。

| 类型 | 主要代码形式 | 提交内容 |
|---|---|---|
| `npu_kernel_dev` | 根目录 `kernel.asc`，直接调用 `run_kernel(...)` | `files: [{path, content}]` |
| 传统算子工程 | `code/op_host`、`code/op_kernel`（也可直接指向code） | tiling_h、tiling_key_h、host_cpp、kernel_cpp |

核函数题先读 [references/kernel-project.md](references/kernel-project.md)。使用 `ops-direct-invoke-flash` 指导核函数设计，保留题目下载的调用接口和构建框架，不能改成默认的 PyTorch 注册工程。
传统工程可使用已有 Ascend C 开发技能。只支持题目规定的 dtype、shape 和属性，不无条件增加无关类型或转换。

```bash
# 先检查文件、只读模板、题目绑定，显示文件列表和SHA-256；不会提交
python3 cannjudge_cli.py submit --problem-url "https://cannjudge.cn/hc_cann_codelabs/cann/add" --project-dir ./add-work/project --dry-run
# 提交并返回 ID；随后查询该 ID
python3 cannjudge_cli.py submit --problem-url "https://cannjudge.cn/hc_cann_codelabs/cann/add" --project-dir ./add-work/project --no-wait
python3 cannjudge_cli.py query --submission-id ID
```

用户已经要求提交时，在代码和检查完成后直接提交，无需重复确认。仅请求缺失的账号登录或目标信息。
对修改代码的提交不自动重试。网络超时或缺少 submissionId 时先查询该账号在该题目的提交记录，核实是否已受理。
平台报错后依据实际编译/用例信息修复，再提交新版本；不把 Running 当作完成。
默认提交后等待120秒，也可用 `--max-wait` 调整或 `--no-wait` 后分次查询，保持用户知悉进展。

## 结果与验证

当前平台成功状态是 `Pass`，兼容旧状态 `Accepted`。同时检查 result 中所有可见测试点的状态。
失败状态可能包括 `Fail`、`Wrong Answer`、`Compile Error`、`Runtime Error`、超时等；保留平台 msg 便于定位。
`precision_ratio` 原样报告，不将所有新字段强行解释成旧版含义；time单位以页面/评测说明为准。
没有NPU环境时，可用在线评测承担编译和真机验证，但不能把静态/CPU检查表述为NPU通过。
交付完整源码工程、提交ID/可访问链接、测试点结果和限制。平台公开测试通过不等于所有规格边界均已实测。

```bash
python3 cannjudge_cli.py rank --problem-id ID
python3 cannjudge_cli.py logout
python3 -m pytest tests -q
```

SDK、API 和传统工程用法见 [README.md](README.md)。
