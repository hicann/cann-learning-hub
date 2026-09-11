#!/usr/bin/env python3
"""
CANNJudge 算子提交辅助工具

提供命令行接口完成 CANNJudge 算子竞赛的完整流程。
"""

import argparse
import requests
import json
import time
import os
import zipfile
import tempfile
import base64
import getpass
import sys
import warnings
import re
import webbrowser
from urllib.parse import quote, urlparse, unquote
from http.cookiejar import Cookie
from pathlib import Path

from session_store import DEFAULT_SESSION_FILE, load_session, save_session
from project_files import extract_package, collect_kernel_files, collect_registry_files

try:
    from Crypto.PublicKey import RSA
    from Crypto.Cipher import PKCS1_v1_5
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


class CANNJudgeError(ValueError):
    """可直接显示的操作提示，不包含凭据或服务端原始响应。"""


def save_captcha_svg(svg: str) -> Path:
    """Preserve the server's SVG verbatim in the agent's current working directory."""
    if not isinstance(svg, str) or not svg.strip():
        raise CANNJudgeError('验证码图片为空，请重新获取')
    fd, filename = tempfile.mkstemp(prefix='cannjudge-captcha-', suffix='.svg', dir=Path.cwd())
    path = Path(filename)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(svg.encode('utf-8'))
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def decrypt_password(ciphertext_b64: str, private_key_path: str = "private.pem") -> str:
    """用 RSA 私钥解密密码密文。返回的明文密码仅供内部调用登录 API，禁止输出到对话/日志/文件。"""
    if not HAS_CRYPTO:
        raise ImportError("需要 pycryptodome 库: pip install pycryptodome")
    with open(private_key_path, "r") as f:
        private_key_pem = f.read()
    rsa_key = RSA.importKey(private_key_pem)
    cipher = PKCS1_v1_5.new(rsa_key)
    ciphertext = base64.b64decode(ciphertext_b64, validate=True)
    sentinel = os.urandom(32)
    plaintext = cipher.decrypt(ciphertext, sentinel=sentinel)
    if plaintext == sentinel:
        raise CANNJudgeError("密文或私钥不匹配，请检查后重试")
    return plaintext.decode()


class CANNJudgeClient:
    """CANNJudge API 客户端"""
    
    BASE_URL = "https://cannjudge.cn"
    
    SESSION_MAX_AGE = 7 * 24 * 60 * 60

    def __init__(self, session_file=None):
        self.session = requests.Session()
        self.user_id = None
        self.user_info = None
        self.session_file = Path(session_file or DEFAULT_SESSION_FILE).expanduser()
        self.logged_in_at = None

    def save_session(self):
        """只保存用户 ID 和 Cookie，不保存密码、密文或完整登录响应。"""
        self.session.cookies.clear_expired_cookies()
        if not self.user_id or not self.session.cookies:
            raise CANNJudgeError("登录未返回可复用 Cookie，请检查平台登录接口")
        cookies = []
        for cookie in self.session.cookies:
            record = vars(cookie).copy()
            record["rest"] = record.pop("_rest")
            cookies.append(record)
        save_session(self.session_file, {
            "version": 1, "base_url": self.BASE_URL,
            "user_id": self.user_id, "logged_in_at": self.logged_in_at,
            "cookies": cookies,
        })

    def load_session(self) -> bool:
        """恢复本机会话；服务端仍可能提前使 Cookie 失效。"""
        try:
            data = load_session(self.session_file)
        except FileNotFoundError:
            return False
        except (OSError, ValueError):
            raise CANNJudgeError("无法读取登录会话，请重新运行 login") from None
        try:
            if data["version"] != 1 or data["base_url"] != self.BASE_URL:
                raise ValueError
            user_id = data["user_id"]
            logged_in_at = data["logged_in_at"]
            if not isinstance(user_id, str) or not user_id:
                raise ValueError
            if not 0 <= time.time() - logged_in_at < self.SESSION_MAX_AGE:
                raise ValueError
            cookies = requests.cookies.RequestsCookieJar()
            for record in data["cookies"]:
                cookies.set_cookie(Cookie(**record))
            cookies.clear_expired_cookies()
            if not cookies:
                raise ValueError
        except (KeyError, TypeError, ValueError, AttributeError):
            raise CANNJudgeError("登录会话已过期或损坏，请重新运行 login") from None
        self.user_id = user_id
        self.user_info = {"_id": user_id}
        self.logged_in_at = logged_in_at
        self.session.cookies = cookies
        return True

    def logout(self):
        """清除本机会话，不承诺撤销服务端或其他设备的登录。"""
        try:
            self.session_file.unlink()
        except FileNotFoundError:
            pass
        self.session.cookies.clear()
        self.user_id = self.user_info = self.logged_in_at = None

    def login_interactive(self, email: str, *, captcha=False, login_type=None,
                          captcha_display='auto') -> dict:
        """用户在真实终端中输入密码，禁止退回回显输入。"""
        if not sys.stdin.isatty():
            raise CANNJudgeError("请在你自己的交互终端运行 login；Agent 对话登录请调用 login(account, password)")
        if captcha_display not in ('auto', 'file', 'browser'):
            raise CANNJudgeError('验证码展示方式应为 auto、file 或 browser')
        captcha_id = captcha_code = None
        if captcha:
            challenge = self.get_captcha()
            captcha_id = challenge['captchaId']
            image_path = save_captcha_svg(challenge['image'])
            print(f'验证码SVG（Agent当前工作目录）: {image_path}')
            remote = bool(os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_TTY'))
            headless_linux = sys.platform.startswith('linux') and not (
                os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
            open_browser = captcha_display == 'browser' or (
                captcha_display == 'auto' and not remote and not headless_linux)
            opened = False
            if open_browser:
                try:
                    opened = webbrowser.open(image_path.as_uri())
                except (webbrowser.Error, OSError):
                    pass
            if not opened:
                print('请保持此终端等待，直接查看上述SVG，或用SFTP/SCP下载到自己的电脑查看。')
            print('服务端原始SVG会保留在当前工作目录，不做格式转换。')
            captcha_code = input('请输入图片中的验证码: ').strip()
            if not captcha_code:
                raise CANNJudgeError('验证码不能为空')
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            try:
                password = getpass.getpass("请输入 CANNJudge 密码（输入不显示）: ")
            except (getpass.GetPassWarning, EOFError):
                raise CANNJudgeError("当前终端不支持安全输入密码，请使用本机终端或 RSA 密文登录") from None
        if not password:
            raise CANNJudgeError("密码不能为空")
        try:
            return self.login(email, password, login_type=login_type,
                              captcha_id=captcha_id, captcha_code=captcha_code)
        finally:
            del password  # 释放引用；Python 不保证字符串内存被立即擦除。

    def _check_response(self, resp):
        if resp.status_code == 401:
            self.logout()
            raise CANNJudgeError("登录会话已失效，请重新运行 login；本次操作未自动重试")
        resp.raise_for_status()
    
    def get_captcha(self) -> dict:
        resp = self.session.get(f'{self.BASE_URL}/api/users/captcha', timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if not result.get('captchaId') or not isinstance(result.get('image'), str):
            raise CANNJudgeError('平台未返回有效验证码')
        return result

    def login(self, email: str, password: str, *, login_type=None,
              captcha_id=None, captcha_code=None) -> dict:
        """登录 CANNJudge（password 为明文密码）"""
        self.session.cookies.clear()
        self.user_id = self.user_info = self.logged_in_at = None
        account_type = login_type or ('phone' if re.fullmatch(r'\d{6,15}', email) else 'email')
        payload = {'account': email, 'loginType': account_type, 'password': password}
        if account_type == 'email':
            payload['email'] = email  # compatibility with older deployments
        if bool(captcha_id) != bool(captcha_code):
            raise CANNJudgeError('captchaId 与 captchaCode 必须同时提供')
        if captcha_id:
            payload.update(captchaId=captcha_id, captchaCode=captcha_code)
        resp = self.session.post(
            f"{self.BASE_URL}/api/users/login",
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        
        user_info = resp.json()
        if not isinstance(user_info, dict) or not isinstance(user_info.get("_id"), str) or not user_info["_id"]:
            raise CANNJudgeError("登录响应缺少用户 ID，请检查账号或平台接口")
        self.user_info = user_info
        self.user_id = user_info["_id"]
        self.logged_in_at = time.time()
        
        return self.user_info
    
    def login_with_ciphertext(self, email: str, ciphertext: str,
                              private_key_path: str = "private.pem", **login_options) -> dict:
        """登录 CANNJudge（ciphertext 为 RSA 加密后的密码密文，服务器用私钥解密）。
        解密后的明文密码仅供内部调用登录 API，禁止输出到对话/日志/文件。"""
        password = decrypt_password(ciphertext, private_key_path)
        try:
            return self.login(email, password, **login_options)
        finally:
            del password
    
    def get_problem(self, problem_id: str) -> dict:
        """获取题目信息"""
        resp = self.session.get(
            f"{self.BASE_URL}/api/problems/{problem_id}",
            timeout=30
        )
        self._check_response(resp)
        return resp.json()

    def _get_json(self, path, params=None):
        resp = self.session.get(self.BASE_URL + path, params=params, timeout=30)
        self._check_response(resp)
        return resp.json()

    def resolve_problem(self, value: str) -> dict:
        """Resolve the full group/contest/problem URL without confusing names across contests."""
        if not value.startswith(('http://', 'https://')):
            return self.get_problem(value)
        parsed = urlparse(value)
        if parsed.scheme != 'https' or parsed.netloc != 'cannjudge.cn':
            raise CANNJudgeError('题目链接必须来自 https://cannjudge.cn')
        parts = [unquote(p) for p in (parsed.fragment.split('?')[0] if parsed.fragment.startswith('/') else parsed.path).strip('/').split('/')]
        if len(parts) < 3:
            raise CANNJudgeError('请提供包含小组、比赛和题目名称的完整题目链接')
        group_name, contest_name, problem_name = parts[:3]
        if group_name == 'public':
            group = self._get_json('/api/groups/public')
        else:
            group = self._get_json('/api/groups/name/' + quote(group_name, safe=''))
        contests = self._get_json('/api/contests/group/' + group['_id'])
        matches = [c for c in contests if c.get('name') == contest_name or c.get('_id') == contest_name]
        if len(matches) != 1:
            raise CANNJudgeError('无法在指定小组唯一定位比赛')
        contest_id = matches[0]['_id']
        params = {'contestId': contest_id}
        if self.user_id:
            params['userId'] = self.user_id
        problem = self._get_json('/api/problems/name/' + quote(problem_name, safe=''), params)
        if problem.get('contest_id') != contest_id:
            raise CANNJudgeError('题目返回的比赛与链接不一致')
        return problem

    def get_template(self, problem_id):
        return self._get_json(f'/api/problems/{problem_id}/template',
                              {'userId': self.user_id} if self.user_id else None)
    
    def download_package(self, problem_id: str, output_dir: str = None, *,
                         include_metadata=False) -> str:
        """下载工程模板"""
        resp = self.session.get(
            f"{self.BASE_URL}/api/problems/{problem_id}/package",
            params={"userId": self.user_id},
            timeout=60
        )
        self._check_response(resp)
        
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        problem = self.get_problem(problem_id) if include_metadata else None
        template = self.get_template(problem_id) if include_metadata else None
        return extract_package(resp.content, output_dir, problem, template)

    def prepare_submission(self, problem_id, project_dir, project_type='auto'):
        if project_type == 'auto':
            problem = self.get_problem(problem_id)
            project_type = problem.get('code_template') or 'registry'
        if project_type == 'npu_kernel_dev':
            files = collect_kernel_files(project_dir, self.get_template(problem_id), problem_id)
            return {'problemId': problem_id, 'userId': self.user_id, 'files': files}
        return {'problemId': problem_id, 'userId': self.user_id,
                **collect_registry_files(project_dir)}

    def submit_payload(self, payload):
        if not self.user_id or payload.get('userId') != self.user_id:
            raise CANNJudgeError('请先登录，并确保提交账号一致')
        resp = self.session.post(f'{self.BASE_URL}/api/submissions/submit', json=payload, timeout=30)
        self._check_response(resp)
        result = resp.json()
        submission_id = result.get('data', {}).get('submissionId')
        if not submission_id:
            raise CANNJudgeError('提交响应没有 submissionId；请先查询提交记录，避免重复提交')
        return submission_id
    
    def submit(self, problem_id: str, 
               tiling_h: str, tiling_key_h: str,
               host_cpp: str, kernel_cpp: str) -> str:
        """提交代码"""
        payload = {
            "problemId": problem_id,
            "userId": self.user_id,
            "tiling_h": tiling_h,
            "tiling_key_h": tiling_key_h,
            "host_cpp": host_cpp,
            "kernel_cpp": kernel_cpp
        }
        
        return self.submit_payload(payload)
    
    def get_submission(self, submission_id: str) -> dict:
        """查询提交结果"""
        resp = self.session.get(
            f"{self.BASE_URL}/api/submissions/{submission_id}",
            timeout=30
        )
        self._check_response(resp)
        return resp.json()
    
    def wait_for_result(self, submission_id: str, 
                        max_wait: int = 120, interval: int = 3) -> dict:
        """等待提交完成并返回结果"""
        if max_wait < 0 or interval <= 0:
            raise CANNJudgeError('等待秒数不能为负，轮询间隔必须大于0')
        deadline = time.monotonic() + max_wait
        while True:
            result = self.get_submission(submission_id)
            status = result.get("status")
            if status in ["Pass", "Fail", "Accepted", "Wrong Answer",
                          "Compile Error", "Runtime Error",
                          "Time Limit Exceeded", "Memory Limit Exceeded", "System Error"]:
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return result
            time.sleep(min(interval, remaining))
    
    def get_rankings(self, problem_id: str) -> list:
        """获取排行榜"""
        resp = self.session.get(
            f"{self.BASE_URL}/api/submissions/problem/{problem_id}/latest",
            timeout=30
        )
        self._check_response(resp)
        return resp.json()


def print_submission_result(result: dict):
    """打印提交结果"""
    status = result.get("status", "Unknown")
    print(f"\n{'='*60}")
    print(f"状态: {status}")
    print(f"{'='*60}")
    
    if result.get('msg'):
        print(result['msg'])
    if "result" in result:
        print("\n测试用例结果:")
        for idx, tc in enumerate(result["result"]):
            tc_status = tc.get("testcase_status", "Unknown")
            precision = tc.get("precision_ratio", "N/A")
            elapsed = tc.get("time", "N/A")
            print(f"  Case {idx+1}: {tc_status}")
            print(f"           precision_ratio: {precision}, time（平台原值）: {elapsed}")
            if tc.get('msg'):
                print(f"           {tc['msg']}")
        passed = sum(tc.get('testcase_status') in ('Pass', 'Accepted') for tc in result['result'])
        print(f"通过: {passed}/{len(result['result'])}")


def main():
    parser = argparse.ArgumentParser(
        description="CANNJudge 算子提交辅助工具"
    )
    parser.add_argument("--session-file", help="会话文件路径（默认 ~/.cannjudge/session.json）")
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    def problem_selector(subparser):
        group = subparser.add_mutually_exclusive_group(required=True)
        group.add_argument('--problem-id', help='题目 ID')
        group.add_argument('--problem-url', help='包含小组/比赛/题目名称的完整链接')

    info_parser = subparsers.add_parser('info', help='读取网站上的题目描述及工程类型')
    problem_selector(info_parser)
    
    # 登录命令
    login_parser = subparsers.add_parser("login", help="登录 CANNJudge")
    login_parser.add_argument("--account", "--email", dest="email", help="邮箱或手机号；省略时在终端输入")
    login_parser.add_argument("--login-type", choices=['email', 'phone'], help="默认按账号格式判断")
    login_parser.add_argument("--captcha", action="store_true", help="获取验证码图片并由用户在终端输入（可选，仅在平台明确要求时使用）")
    login_parser.add_argument('--captcha-display', choices=['auto', 'file', 'browser'], default='auto',
                              help='验证码展示：auto在SSH/无桌面Linux下仅保存文件；file不打开浏览器')
    login_parser.add_argument("--captcha-id", help="已有验证码 ID，与 --captcha-code 配套")
    login_parser.add_argument("--captcha-code", help="用户读取的验证码，与 --captcha-id 配套")
    credentials = login_parser.add_mutually_exclusive_group()
    credentials.add_argument("--password", help="兼容旧版；建议省略此参数，在终端隐藏输入")
    credentials.add_argument("--ciphertext", help="RSA 加密后的密码密文（兼容方式）")
    login_parser.add_argument("--private-key", default="private.pem", help="RSA 私钥路径")
    subparsers.add_parser("logout", help="清除本机保存的登录会话")
    
    # 下载命令
    download_parser = subparsers.add_parser("download", help="下载工程模板")
    problem_selector(download_parser)
    download_parser.add_argument("--output", help="输出目录")
    
    # 提交命令
    submit_parser = subparsers.add_parser("submit", help="提交代码")
    problem_selector(submit_parser)
    submit_parser.add_argument("--project-dir", required=True, help="工程目录")
    submit_parser.add_argument('--project-type', choices=['auto', 'npu_kernel_dev', 'registry'], default='auto', help='默认依据网站 code_template 自动判断')
    submit_parser.add_argument('--dry-run', action='store_true', help='核对模板并显示提交文件，不创建提交')
    submit_parser.add_argument('--no-wait', action='store_true', help='获得提交 ID 后立即返回')
    submit_parser.add_argument('--max-wait', type=int, default=120, help='结果等待秒数')
    
    # 查询命令
    query_parser = subparsers.add_parser("query", help="查询提交结果")
    query_parser.add_argument("--submission-id", required=True, help="提交ID")
    
    # 排行榜命令
    rank_parser = subparsers.add_parser("rank", help="查看排行榜")
    rank_parser.add_argument("--problem-id", required=True, help="题目ID")
    
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0
    try:
        return run_command(args)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "未知"
        if args.command == 'login' and status in (400, 401):
            print('错误: 登录失败，请检查账号和密码；仅在平台明确要求验证码时使用 --captcha', file=sys.stderr)
        elif status == 401:
            print("错误: 登录失败，请检查邮箱和密码", file=sys.stderr)
        elif status == 403:
            print("错误: 平台拒绝访问，请检查账号权限或重新登录", file=sys.stderr)
        else:
            print(f"错误: 平台请求失败（HTTP {status}）", file=sys.stderr)
    except requests.RequestException:
        print("错误: 无法连接 CANNJudge，请检查网络后重试", file=sys.stderr)
    except CANNJudgeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
    except (ValueError, OSError, ImportError) as exc:
        # 不输出响应正文或 traceback，避免暴露登录请求内容。
        if args.command == "login":
            print("错误: 登录或保存会话失败，请检查终端、凭据、密钥和会话文件权限", file=sys.stderr)
        else:
            print(f"错误: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n操作已取消", file=sys.stderr)
    return 1


def run_command(args):
    
    # 创建客户端
    client = CANNJudgeClient(session_file=args.session_file)
    if args.command == "logout":
        client.logout()
        print("已清除本机登录会话")
        return 0
    
    if args.command == "login":
        if not args.email:
            if not sys.stdin.isatty():
                raise CANNJudgeError('请在自己的交互终端运行 login')
            args.email = input('CANNJudge 账号（邮箱或手机号）: ').strip()
        if not args.email:
            raise CANNJudgeError('账号不能为空')
        # 切换账号或重新登录失败时，不再悄悄复用之前的账号。
        client.logout()
        if args.ciphertext:
            if args.captcha:
                raise CANNJudgeError('RSA 方式请使用 --captcha-id 与 --captcha-code；普通账号密码方式使用 --captcha')
            result = client.login_with_ciphertext(args.email, args.ciphertext, args.private_key,
                                                 login_type=args.login_type,
                                                 captcha_id=args.captcha_id, captcha_code=args.captcha_code)
        elif args.password:
            print("提示: 建议省略 --password，在终端隐藏输入，避免密码进入命令历史", file=sys.stderr)
            result = client.login(args.email, args.password, login_type=args.login_type,
                                  captcha_id=args.captcha_id, captcha_code=args.captcha_code)
        else:
            if args.captcha_id or args.captcha_code:
                raise CANNJudgeError('隐藏密码登录请使用 --captcha 自动获取验证码')
            result = client.login_interactive(args.email, captcha=args.captcha, login_type=args.login_type,
                                              captcha_display=args.captcha_display)
        client.save_session()
        print(f"登录成功!")
        print(f"用户ID: {result['_id']}")
        print("登录会话已保存，后续命令会自动复用；过期后重新登录即可")
        return 0

    loaded = client.load_session()
    if not loaded and args.command not in ('info', 'download') and not (args.command == 'submit' and args.dry_run):
        print("错误: 请先在自己的终端运行 login", file=sys.stderr)
        return 1
        
    if args.command in ('info', 'download', 'submit'):
        if args.problem_url:
            args.problem_id = client.resolve_problem(args.problem_url)['_id']

    if args.command == 'info':
        print(json.dumps(client.get_problem(args.problem_id), ensure_ascii=False, indent=2))
        return 0

    if args.command == "download":
        
        print(f"下载题目 {args.problem_id} 的工程模板...")
        extract_dir = client.download_package(args.problem_id, args.output, include_metadata=True)
        print(f"工程已解压到: {extract_dir}")
        
    elif args.command == "submit":
        payload = client.prepare_submission(args.problem_id, args.project_dir, args.project_type)
        if args.dry_run:
            files = payload.get('files') or [{'path': k, 'content': v} for k, v in payload.items() if k not in ('problemId', 'userId')]
            import hashlib
            print(json.dumps({'problemId': args.problem_id, 'files': [
                {'path': f['path'], 'bytes': len(f['content'].encode('utf-8')),
                 'sha256': hashlib.sha256(f['content'].encode('utf-8')).hexdigest()} for f in files
            ]}, ensure_ascii=False, indent=2))
            return 0
        
        print(f"提交代码到题目 {args.problem_id}...")
        submission_id = client.submit_payload(payload)
        print(f"提交成功! Submission ID: {submission_id}")
        
        if not args.no_wait:
            print("\n等待结果...")
            result = client.wait_for_result(submission_id, max_wait=args.max_wait)
            print_submission_result(result)
        
    elif args.command == "query":
        if not client.user_id:
            print("错误: 请先登录")
            return 1
        
        result = client.get_submission(args.submission_id)
        print_submission_result(result)
        
    elif args.command == "rank":
        if not client.user_id:
            print("错误: 请先登录")
            return 1
        
        rankings = client.get_rankings(args.problem_id)
        
        print(f"\n题目 {args.problem_id} 排行榜:")
        print("=" * 60)
        
        # 按状态和分数排序
        sorted_rankings = sorted(
            rankings,
            key=lambda x: (x.get("status") in ('Pass', 'Accepted'), x.get("score", 0)),
            reverse=True
        )
        
        for i, item in enumerate(sorted_rankings[:20]):
            user_id = item.get("user_id", "unknown")
            status = item.get("status", "unknown")
            score = item.get("score", "N/A")
            print(f"{i+1:3d}. {user_id:24s} | {status:15s} | score: {score}")
    
    if loaded:
        try:
            client.save_session()
        except (OSError, ValueError):
            print("提示: 本次操作已完成，但无法更新本机会话；下次使用前可能需要重新登录", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
