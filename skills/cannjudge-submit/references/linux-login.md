# Linux / SSH 终端登录

在实际执行下载、提交的 Linux 主机、同一个系统账号下登录。
新登录前先选择方式：Agent 对话输入、终端隐藏输入或 RSA。本文说明终端方式；默认直接调用登录接口，不需要图片或二维码。

## 在当前 Agent 工作目录登录

保持当前目录为 Agent 的项目/工作目录。使用技能绝对路径执行，不要为了登录切换到技能安装目录。
在当前 Python 环境安装依赖，然后运行（替换技能路径和账号）：

```bash
python3 -m pip install requests
bash /path/to/cannjudge-submit/login.sh --account 'your@email.com'
```

也可直接运行：

```bash
python3 /path/to/cannjudge-submit/cannjudge_cli.py login --account 'your@email.com'
```

如果系统 Python 不允许安装包，可先在工作目录创建并激活虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install requests
```

省略 --account 可在终端输入账号；支持邮箱或手机号。
自定义会话路径时使用完整CLI，将 --session-file PATH 放在 login 子命令之前。
login.sh 不改变调用者的工作目录。

## 可选：查看 SVG 验证码

仅当用户选择或平台明确要求验证码时，给登录命令添加 `--captcha --captcha-display file`。默认流程不执行以下步骤。

1. 服务端返回的 SVG **原样保存**，不进行栅格化或图片格式转换。
2. SVG 保存于 **Agent 当前工作目录**，例如 `/work/my-agent/cannjudge-captcha-随机值.svg`；命令打印完整路径并等待输入。
3. 直接在工作区打开 SVG；若远程工作区无法显示图片，用 SFTP/SCP 下载到自己的电脑查看。
4. 回到 Linux 终端输入验证码，再隐藏输入密码。

SCP 示例（替换为实际用户名、服务器和打印的SVG路径）：

```bash
scp 'user@server:/work/my-agent/cannjudge-captcha-实际随机值.svg' ./cannjudge-captcha.svg
```

Linux下SVG权限为600。每次获取使用不同文件名，避免覆盖；登录成功、失败或取消后图片保留，用户可自行删除。
这里的当前目录指执行命令时的工作目录（pwd / Path.cwd()），不是系统 /tmp 或技能安装目录。
验证码过期时重新运行命令，不复用旧图片。不得把会话文件随SVG一起分享。

`--captcha-display auto` 在SSH或无DISPLAY/WAYLAND_DISPLAY的Linux中使用文件方式；
`file` 始终不打开浏览器，`browser` 适用于确实有桌面的机器。三种方式均保存同一规则的SVG文件。
需要交互TTY，密码不通过管道、命令行或环境变量传入。

## 后续会话复用

成功后会话仍保存于 Linux 主机的 `~/.cannjudge/session.json`（权限600），后续同账号自动复用。
本次工作目录调整仅针对验证码图片。Windows DPAPI会话不能搬到Linux使用，须在Linux重新登录。
不要在登录与提交之间切换为sudo/root。
