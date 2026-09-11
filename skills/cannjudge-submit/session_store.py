"""Persist login cookies without storing passwords or RSA ciphertext."""

import ctypes
import json
import os
import stat
import tempfile
from pathlib import Path


DEFAULT_SESSION_FILE = Path.home() / ".cannjudge" / "session.json"
DPAPI_PREFIX = b"CANNJUDGE-DPAPI\0"


def _dpapi(data: bytes, decrypt: bool = False) -> bytes:
    """Use Windows account-bound encryption; never fall back to plaintext."""
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = crypt32.CryptUnprotectData if decrypt else crypt32.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p,
                        ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                        wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source = Blob(len(data), buffer)
    target = Blob()
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise OSError("无法使用当前 Windows 账号保护或读取登录会话")
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        kernel32.LocalFree(target.data)


def save_session(path: Path, data: dict):
    """Write atomically: POSIX mode 0600; Windows DPAPI ciphertext."""
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    if os.name == "nt":
        payload = DPAPI_PREFIX + _dpapi(payload)
    path = Path(path).expanduser()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("会话文件不能是符号链接")
    fd, temporary = tempfile.mkstemp(prefix=".session-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_session(path: Path) -> dict:
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError("会话文件不能是符号链接")
    with path.open("rb") as stream:
        if os.name != "nt":
            metadata = os.fstat(stream.fileno())
            if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
                raise ValueError("会话文件须属于当前用户且权限为 600；请重新登录")
        payload = stream.read()
    if os.name == "nt":
        if not payload.startswith(DPAPI_PREFIX):
            raise ValueError("Windows 会话须使用当前账号加密；请重新登录")
        payload = _dpapi(payload[len(DPAPI_PREFIX):], decrypt=True)
    elif payload.startswith(DPAPI_PREFIX):
        raise ValueError("Windows 会话不能跨平台复用；请在当前机器重新登录")
    return json.loads(payload)
