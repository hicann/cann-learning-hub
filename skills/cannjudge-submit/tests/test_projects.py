"""Offline project transport and authentication protocol regression coverage."""
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cannjudge_cli import CANNJudgeClient, save_captcha_svg
from project_files import collect_kernel_files, extract_package, collect_registry_files

SVG_FIXTURE = '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="40"><rect width="120" height="40" fill="white"/></svg>'


@pytest.fixture(autouse=True)
def agent_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def template():
    return {'data': {'files': [
        {'path': 'kernel.asc', 'content': '// implement', 'editable': True},
        {'path': 'main.asc', 'content': '// framework\n', 'editable': False},
        {'path': 'CMakeLists.txt', 'content': '# build\n', 'editable': False},
    ]}}


def project(root):
    for f in template()['data']['files']:
        (root / f['path']).write_text(f['content'], encoding='utf-8')
    (root / 'kernel.asc').write_text('// actual implementation 中文', encoding='utf-8')
    (root / 'helper.h').write_text('// helper', encoding='utf-8')
    (root / 'notes.txt').write_text('not submitted', encoding='utf-8')


def test_kernel_payload_contains_only_editable_and_allowed_helpers(tmp_path):
    project(tmp_path)
    client = CANNJudgeClient(tmp_path / 'session')
    client.user_id = 'test-user'
    client.get_problem = Mock(return_value={'code_template': 'npu_kernel_dev'})
    client.get_template = Mock(return_value=template())
    payload = client.prepare_submission('problem-id', tmp_path)
    assert set(payload) == {'problemId', 'userId', 'files'}
    assert {f['path'] for f in payload['files']} == {'helper.h', 'kernel.asc'}
    assert '中文' in next(f['content'] for f in payload['files'] if f['path'] == 'kernel.asc')
    client.session.post = Mock(return_value=Mock(status_code=200, json=lambda: {'data': {'submissionId': 'submission-id'}}))
    assert client.submit_payload(payload) == 'submission-id'
    assert client.session.post.call_args.kwargs['json'] == payload


def test_modified_protected_file_is_rejected(tmp_path):
    project(tmp_path)
    (tmp_path / 'main.asc').write_text('// changed', encoding='utf-8')
    with pytest.raises(ValueError, match='只读'):
        collect_kernel_files(tmp_path, template(), 'problem-id')


def test_wrong_problem_and_missing_source_are_rejected(tmp_path):
    project(tmp_path)
    manifest = tmp_path / '.cannjudge-project.json'
    manifest.write_text(json.dumps({'problem_id': 'different'}))
    with pytest.raises(ValueError, match='题目'):
        collect_kernel_files(tmp_path, template(), 'problem-id')
    manifest.unlink()
    (tmp_path / 'kernel.asc').unlink()
    with pytest.raises(ValueError, match='根目录'):
        collect_kernel_files(tmp_path, template(), 'problem-id')


@pytest.mark.parametrize('path', ['../escape.asc', '/absolute.asc', 'C:/absolute.asc', 'dir/../../escape.asc'])
def test_archive_traversal_never_writes_files(tmp_path, path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as z:
        z.writestr(path, '// malicious path')
    with pytest.raises(ValueError):
        extract_package(data.getvalue(), tmp_path)
    assert not (tmp_path / 'project').exists()


def test_existing_project_not_overwritten(tmp_path):
    (tmp_path / 'project').mkdir()
    with pytest.raises(ValueError, match='覆盖'):
        extract_package(b'', tmp_path)


def test_ambiguous_registry_files_rejected(tmp_path):
    (tmp_path / 'op_kernel').mkdir()
    for p in ('a_tiling.h', 'b_tiling.h'):
        (tmp_path / 'op_kernel' / p).write_text('// x')
    with pytest.raises(ValueError, match='唯一'):
        collect_registry_files(tmp_path)


def test_url_resolution_preserves_contest_scope(tmp_path):
    client = CANNJudgeClient(tmp_path / 'session')
    client._get_json = Mock(side_effect=[{'_id': 'group-id'}, [{'name': 'cann', '_id': 'contest-id'}],
                                       {'_id': 'problem-id', 'contest_id': 'contest-id'}])
    assert client.resolve_problem('https://cannjudge.cn/hc_cann_codelabs/cann/add')['_id'] == 'problem-id'
    assert client._get_json.call_args.args[1] == {'contestId': 'contest-id'}
    with pytest.raises(ValueError):
        client.resolve_problem('https://elsewhere.test/group/contest/add')


def test_phone_login_uses_current_captcha_protocol(tmp_path):
    client = CANNJudgeClient(tmp_path / 'session')
    client.session.post = Mock(return_value=Mock(json=lambda: {'_id': 'test-user'}))
    client.login('13800000000', 'fake-password', captcha_id='challenge', captcha_code='ABCD')
    assert client.session.post.call_args.kwargs['json'] == {
        'account': '13800000000', 'loginType': 'phone', 'password': 'fake-password',
        'captchaId': 'challenge', 'captchaCode': 'ABCD'}


def test_password_remains_hidden_in_interactive_captcha_login(tmp_path):
    client = CANNJudgeClient(tmp_path / 'session')
    client.get_captcha = Mock(return_value={'captchaId': 'challenge', 'image': SVG_FIXTURE})
    client.login = Mock(return_value={'_id': 'test-user'})
    with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', return_value='ABCD'), \
            patch('getpass.getpass', return_value='fake-password'), patch('webbrowser.open'):
        client.login_interactive('test@example.com', captcha=True)
    assert client.login.call_args.kwargs['captcha_code'] == 'ABCD'


@pytest.mark.parametrize('status', ['Pass', 'Accepted', 'Fail', 'Compile Error'])
def test_terminal_status_returns_without_unnecessary_wait(tmp_path, status):
    client = CANNJudgeClient(tmp_path / 'session')
    client.get_submission = Mock(return_value={'status': status, 'result': []})
    with patch('time.sleep') as sleep:
        assert client.wait_for_result('submission')['status'] == status
        sleep.assert_not_called()
    assert client.get_submission.call_count == 1


def test_missing_submission_id_does_not_retry_post(tmp_path):
    client = CANNJudgeClient(tmp_path / 'session')
    client.user_id = 'test-user'
    client.session.post = Mock(return_value=Mock(status_code=200, json=lambda: {'code': 0}))
    with pytest.raises(ValueError, match='重复提交'):
        client.submit_payload({'problemId': 'problem', 'userId': 'test-user', 'files': []})
    assert client.session.post.call_count == 1


@pytest.mark.parametrize('environment,platform,display', [
    ({}, 'linux', 'auto'),
    ({'SSH_CONNECTION': 'test', 'DISPLAY': ':0'}, 'linux', 'auto'),
    ({'WAYLAND_DISPLAY': 'wayland-0'}, 'linux', 'file'),
])
def test_linux_terminal_captcha_file_survives_until_user_input(tmp_path, environment, platform, display):
    client = CANNJudgeClient(tmp_path / 'session')
    client.get_captcha = Mock(return_value={'captchaId': 'challenge', 'image': SVG_FIXTURE})
    client.login = Mock(return_value={'_id': 'test-user'})
    def read_code(_prompt):
        images = list(tmp_path.glob('cannjudge-captcha-*.svg'))
        assert len(images) == 1
        assert images[0].read_bytes() == SVG_FIXTURE.encode('utf-8')
        assert not list(tmp_path.rglob('*.png'))
        if os.name != 'nt':
            assert images[0].stat().st_mode & 0o777 == 0o600
        return 'ABCD'

    with patch.dict(os.environ, environment, clear=True), patch('sys.platform', platform), \
            patch('sys.stdin.isatty', return_value=True), patch('builtins.input', side_effect=read_code), \
            patch('getpass.getpass', return_value='fake-password'), patch('webbrowser.open') as browser:
        client.login_interactive('test@example.com', captcha=True, captcha_display=display)
    browser.assert_not_called()
    assert len(list(tmp_path.glob('cannjudge-captcha-*.svg'))) == 1
    assert client.login.call_args.kwargs['captcha_code'] == 'ABCD'


def test_unavailable_browser_does_not_abort_login(tmp_path):
    client = CANNJudgeClient(tmp_path / 'session')
    client.get_captcha = Mock(return_value={'captchaId': 'challenge', 'image': SVG_FIXTURE})
    client.login = Mock(return_value={'_id': 'test-user'})
    with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', return_value='ABCD'), \
            patch('getpass.getpass', return_value='fake-password'), patch('webbrowser.open', side_effect=OSError):
        client.login_interactive('test@example.com', captcha=True, captcha_display='browser')
    client.login.assert_called_once()


def test_svg_preserves_original_bytes_and_does_not_overwrite(tmp_path):
    first = save_captcha_svg(SVG_FIXTURE)
    second = save_captcha_svg(SVG_FIXTURE)
    assert first != second and first.parent == tmp_path and second.parent == tmp_path
    assert first.read_bytes() == second.read_bytes() == SVG_FIXTURE.encode('utf-8')
    assert {p.suffix for p in tmp_path.iterdir()} == {'.svg'}


def test_empty_svg_does_not_leave_partial_image(tmp_path):
    with pytest.raises(ValueError, match='为空'):
        save_captcha_svg('')
    assert list(tmp_path.iterdir()) == []
