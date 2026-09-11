"""Offline authentication/session regression tests; no real credentials or API calls."""

import base64
import contextlib
import getpass
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
import zipfile
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cannjudge_cli as cli
import session_store

REAL_SESSION = cli.requests.Session
TEST_PASSWORD = "  fake-password-with-spaces  "
TEST_COOKIE = "fake-session-token"


class FakeAdapter(cli.requests.adapters.BaseAdapter):
    def __init__(self):
        self.requests = []
        self.status = 200
        self.issue_cookie = True

    def send(self, request, **kwargs):
        self.requests.append(request)
        response = cli.requests.Response()
        response.request = request
        response.url = request.url
        response.status_code = self.status
        headers = Message()
        if request.url.endswith("/api/users/login"):
            payload = json.loads(request.body)
            if payload["password"] != TEST_PASSWORD:
                raise AssertionError("password changed before transport")
            if self.issue_cookie:
                headers.add_header("Set-Cookie", f"sid={TEST_COOKIE}; Path=/api; Secure; HttpOnly")
            body = {"_id": "user-123", "nickname": "test", "password": TEST_PASSWORD}
        else:
            if f"sid={TEST_COOKIE}" not in request.headers.get("Cookie", ""):
                raise AssertionError("saved cookie was not sent")
            if "/package?" in request.url:
                archive = io.BytesIO()
                with zipfile.ZipFile(archive, "w") as stream:
                    stream.writestr("code/op_kernel/test.cpp", "// test fixture")
                response._content = archive.getvalue()
                body = None
            elif request.url.endswith("/api/submissions/submit"):
                if json.loads(request.body)["userId"] != "user-123":
                    raise AssertionError("saved user ID missing")
                body = {"data": {"submissionId": "submission-1"}}
            elif request.url.endswith("/latest"):
                body = []
            elif request.url.split('?')[0].endswith('/template'):
                body = {'data': {'files': []}}
            elif request.url.endswith('/api/problems/problem-1'):
                body = {'_id': 'problem-1', 'code_template': 'registry'}
            else:
                body = {"status": "Accepted", "result": []}
        if body is not None:
            response._content = json.dumps(body).encode()
        response._content_consumed = True
        response.raw = SimpleNamespace(_original_response=SimpleNamespace(msg=headers))
        return response

    def close(self):
        pass


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "account" / "session.json"
        self.adapter = FakeAdapter()
        self.session_patch = patch.object(cli.requests, "Session", self.new_session)
        self.session_patch.start()
        self.addCleanup(self.session_patch.stop)

    def new_session(self):
        session = REAL_SESSION()
        session.mount("https://", self.adapter)
        session.mount("http://", self.adapter)
        return session

    def client(self):
        return cli.CANNJudgeClient(session_file=self.path)

    def seed(self):
        client = self.client()
        client.login("test@example.com", TEST_PASSWORD)
        client.save_session()
        return client

    def command(self, *arguments):
        output = io.StringIO()
        with patch.object(sys, "argv", ["cannjudge_cli.py", "--session-file", str(self.path), *arguments]), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = cli.main()
        self.assertNotIn(TEST_PASSWORD, output.getvalue())
        self.assertNotIn(TEST_COOKIE, output.getvalue())
        return result, output.getvalue()

    def test_direct_sdk_login_needs_no_captcha_or_terminal(self):
        for account, login_type in [('test@example.com', 'email'), ('13800000000', 'phone')]:
            with self.subTest(login_type=login_type):
                self.adapter.requests.clear()
                client = self.client()
                with patch.object(sys.stdin, 'isatty', return_value=False), \
                        patch.object(cli.getpass, 'getpass', side_effect=AssertionError('must not prompt')):
                    client.login(account, TEST_PASSWORD)
                client.save_session()
                self.assertEqual(len(self.adapter.requests), 1)
                request = self.adapter.requests[0]
                self.assertEqual(request.method, 'POST')
                self.assertTrue(request.url.endswith('/api/users/login'))
                payload = json.loads(request.body)
                self.assertEqual(payload['account'], account)
                self.assertEqual(payload['loginType'], login_type)
                self.assertNotIn('captchaId', payload)
                self.assertNotIn('captchaCode', payload)
                self.assertNotIn(TEST_PASSWORD, json.dumps(session_store.load_session(self.path)))

    def test_interactive_login_preserves_spaces_and_restores_cookie_attributes(self):
        client = self.client()
        with patch.object(sys.stdin, "isatty", return_value=True), \
                patch.object(cli.getpass, "getpass", return_value=TEST_PASSWORD):
            client.login_interactive("test@example.com")
        client.save_session()
        restored = self.client()
        self.assertTrue(restored.load_session())
        self.assertEqual(restored.user_id, "user-123")
        original_cookie = next(iter(client.session.cookies))
        self.assertEqual(vars(next(iter(restored.session.cookies))), vars(original_cookie))
        self.assertEqual(original_cookie.path, "/api")
        self.assertTrue(original_cookie.secure)
        restored.get_rankings("problem-1")
        decoded = session_store.load_session(self.path)
        self.assertEqual(set(decoded), {"version", "base_url", "user_id", "logged_in_at", "cookies"})
        self.assertNotIn(TEST_PASSWORD, json.dumps(decoded))
        self.assertNotIn("nickname", json.dumps(decoded))

    def test_login_then_download_submit_query_rank_in_separate_cli_invocations(self):
        with patch.object(sys.stdin, "isatty", return_value=True), \
                patch.object(cli.getpass, "getpass", return_value=TEST_PASSWORD):
            self.assertEqual(self.command("login", "--email", "test@example.com")[0], 0)
        project = Path(self.temp.name) / "project"
        for name in ["op_kernel/test_tiling.h", "op_kernel/test.cpp", "op_host/test.cpp"]:
            file = project / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("// fixture", encoding="utf-8")
        download = Path(self.temp.name) / "download"
        self.assertEqual(self.command("download", "--problem-id", "problem-1", "--output", str(download))[0], 0)
        self.assertTrue((download / "project/code/op_kernel/test.cpp").exists())
        with patch.object(cli.time, "sleep"):
            self.assertEqual(self.command("submit", "--problem-id", "problem-1", "--project-dir", str(project))[0], 0)
        self.assertEqual(self.command("query", "--submission-id", "submission-1")[0], 0)
        self.assertEqual(self.command("rank", "--problem-id", "problem-1")[0], 0)
        self.assertEqual(sum(r.url.endswith("/login") for r in self.adapter.requests), 1)

    def test_session_loads_in_new_process_from_different_working_directory(self):
        self.seed()
        script = """import sys
sys.path.insert(0, sys.argv[1])
from cannjudge_cli import CANNJudgeClient
c = CANNJudgeClient(session_file=sys.argv[2])
assert c.load_session()
assert c.user_id == 'user-123'
assert len(c.session.cookies) == 1
print('restored')
"""
        result = subprocess.run([sys.executable, "-c", script, str(Path(cli.__file__).parent), str(self.path)],
                                cwd=self.temp.name, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "restored")

    def test_no_session_stops_before_network_request(self):
        self.assertEqual(self.command("rank", "--problem-id", "p")[0], 1)
        self.assertEqual(self.adapter.requests, [])

    def test_noninteractive_login_does_not_read_password(self):
        with patch.object(sys.stdin, "isatty", return_value=False), patch.object(cli.getpass, "getpass") as prompt:
            code, output = self.command("login", "--email", "test@example.com")
        self.assertEqual(code, 1)
        self.assertIn("交互终端", output)
        prompt.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_getpass_never_falls_back_to_echo(self):
        def unavailable(*args):
            warnings.warn("echo unavailable", getpass.GetPassWarning)
            self.fail("continued to echoed input")
        with patch.object(sys.stdin, "isatty", return_value=True), \
                patch.object(cli.getpass, "getpass", side_effect=unavailable):
            with self.assertRaises(cli.CANNJudgeError):
                self.client().login_interactive("test@example.com")

    def test_expired_session_or_cookies_stop_before_network(self):
        for mode in ["timestamp", "cookie"]:
            with self.subTest(mode=mode):
                self.seed()
                data = session_store.load_session(self.path)
                if mode == "timestamp":
                    data["logged_in_at"] = time.time() - cli.CANNJudgeClient.SESSION_MAX_AGE - 1
                else:
                    data["cookies"][0]["expires"] = int(time.time()) - 1
                session_store.save_session(self.path, data)
                count = len(self.adapter.requests)
                self.assertEqual(self.command("rank", "--problem-id", "p")[0], 1)
                self.assertEqual(len(self.adapter.requests), count)

    def test_malformed_or_wrong_origin_session_is_rejected(self):
        self.seed()
        data = session_store.load_session(self.path)
        for invalid in [[], {}, {**data, "base_url": "https://example.com"}, {**data, "cookies": [None]}]:
            session_store.save_session(self.path, invalid)
            with self.assertRaises(cli.CANNJudgeError):
                self.client().load_session()

    def test_401_clears_session_and_does_not_retry_submission(self):
        client = self.seed()
        self.adapter.status = 401
        count = len(self.adapter.requests)
        with self.assertRaises(cli.CANNJudgeError):
            client.submit("p", "tiling", "", "host", "kernel")
        self.assertEqual(len(self.adapter.requests), count + 1)
        self.assertFalse(self.path.exists())
        self.assertIsNone(client.user_id)

    def test_403_preserves_session_and_reports_permissions(self):
        self.seed()
        self.adapter.status = 403
        code, output = self.command("rank", "--problem-id", "p")
        self.assertEqual(code, 1)
        self.assertIn("权限", output)
        self.assertTrue(self.path.exists())

    def test_failed_login_removes_previous_account_and_omits_response_body(self):
        self.seed()
        self.adapter.status = 401
        with patch.object(sys.stdin, "isatty", return_value=True), \
                patch.object(cli.getpass, "getpass", return_value=TEST_PASSWORD):
            self.assertEqual(self.command("login", "--email", "new@example.com")[0], 1)
        self.assertFalse(self.path.exists())

    def test_logout_is_idempotent_and_local(self):
        self.seed()
        count = len(self.adapter.requests)
        self.assertEqual(self.command("logout")[0], 0)
        self.assertEqual(self.command("logout")[0], 0)
        self.assertFalse(self.path.exists())
        self.assertEqual(len(self.adapter.requests), count)

    def test_password_environment_is_not_used(self):
        with patch.dict(os.environ, {"CANNJUDGE_EMAIL": "test@example.com", "CANNJUDGE_PASSWORD": TEST_PASSWORD}):
            self.assertEqual(self.command("rank", "--problem-id", "p")[0], 1)
        self.assertEqual(self.adapter.requests, [])

    def test_normal_login_works_without_crypto_dependency(self):
        with patch.object(cli, "HAS_CRYPTO", False), patch.object(sys.stdin, "isatty", return_value=True), \
                patch.object(cli.getpass, "getpass", return_value=TEST_PASSWORD):
            self.assertEqual(self.command("login", "--email", "test@example.com")[0], 0)

    def test_login_without_cookie_does_not_claim_reusable_session(self):
        self.adapter.issue_cookie = False
        with patch.object(sys.stdin, "isatty", return_value=True), \
                patch.object(cli.getpass, "getpass", return_value=TEST_PASSWORD):
            code, output = self.command("login", "--email", "test@example.com")
        self.assertEqual(code, 1)
        self.assertIn("Cookie", output)
        self.assertFalse(self.path.exists())

    @unittest.skipUnless(os.name == "nt", "Windows-only DPAPI integration")
    def test_windows_encryption_failure_never_writes_plaintext(self):
        with patch.object(session_store, "_dpapi", side_effect=OSError("simulated")):
            with self.assertRaises(OSError):
                session_store.save_session(self.path, {"cookie": TEST_COOKIE})
        self.assertFalse(self.path.exists())

    @unittest.skipUnless(cli.HAS_CRYPTO, "optional RSA dependency unavailable")
    def test_existing_rsa_ciphertext_login_still_saves_session(self):
        key = cli.RSA.generate(2048)
        private = Path(self.temp.name) / "private.pem"
        private.write_bytes(key.export_key())
        ciphertext = base64.b64encode(cli.PKCS1_v1_5.new(key.publickey()).encrypt(TEST_PASSWORD.encode())).decode()
        self.assertEqual(self.command("login", "--email", "test@example.com", "--ciphertext", ciphertext,
                                      "--private-key", str(private))[0], 0)
        self.assertTrue(self.client().load_session())
        self.assertNotIn(ciphertext, json.dumps(session_store.load_session(self.path)))

    def test_atomic_write_failure_preserves_previous_session(self):
        self.seed()
        before = self.path.read_bytes()
        with patch.object(session_store.os, "replace", side_effect=OSError("simulated")):
            with self.assertRaises(OSError):
                session_store.save_session(self.path, {"new": "data"})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob(".session-*")), [])

    @unittest.skipUnless(os.name == "nt", "Windows-only DPAPI integration")
    def test_windows_storage_is_encrypted_and_tampering_fails(self):
        self.seed()
        content = self.path.read_bytes()
        self.assertTrue(content.startswith(session_store.DPAPI_PREFIX))
        self.assertNotIn(TEST_COOKIE.encode(), content)
        self.path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        with self.assertRaises(cli.CANNJudgeError):
            self.client().load_session()

    @unittest.skipIf(os.name == "nt", "POSIX-only permission integration")
    def test_posix_storage_permissions_and_insecure_file_rejection(self):
        self.seed()
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.path.parent.stat().st_mode), 0o700)
        self.path.chmod(0o644)
        with self.assertRaises(cli.CANNJudgeError):
            self.client().load_session()


if __name__ == "__main__":
    unittest.main()
