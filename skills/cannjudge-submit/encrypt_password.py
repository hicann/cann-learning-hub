#!/usr/bin/env python3
"""
RSA 密码加密脚本

在个人 PC 上运行，用公钥加密 CANNJudge 密码，生成密文。
将密文提供给服务器端（CANNBot），服务器用私钥解密后登录。

用法：
    python3 encrypt_password.py

或者指定公钥路径：
    python3 encrypt_password.py --public-key /path/to/public.pem

依赖：
    pip install pycryptodome
"""

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
import base64
import argparse
import sys

DEFAULT_PUBLIC_KEY_PATH = "public.pem"


def encrypt_password(public_key_path: str, password: str) -> str:
    with open(public_key_path, "r", encoding="utf-8") as f:
        public_key_pem = f.read()

    rsa_key = RSA.importKey(public_key_pem)
    cipher = PKCS1_v1_5.new(rsa_key)
    encrypted = base64.b64encode(cipher.encrypt(password.encode())).decode()
    return encrypted


def main():
    parser = argparse.ArgumentParser(description="RSA 加密 CANNJudge 密码")
    parser.add_argument(
        "--public-key",
        default=DEFAULT_PUBLIC_KEY_PATH,
        help=f"公钥文件路径 (默认: {DEFAULT_PUBLIC_KEY_PATH})",
    )
    args = parser.parse_args()

    try:
        with open(args.public_key, "r", encoding="utf-8") as f:
            f.read()
    except FileNotFoundError:
        print(f"错误: 找不到公钥文件 '{args.public_key}'")
        print("请先将服务器上的 public.pem 拷贝到当前目录或指定路径。")
        sys.exit(1)

    pwd = input("请输入你的 CANNJudge 密码：").strip()
    if not pwd:
        print("错误: 密码不能为空")
        sys.exit(1)

    encrypted = encrypt_password(args.public_key, pwd)

    print("\n" + "=" * 60)
    print("加密完成！将下方密文提供给 CANNBot 即可登录")
    print("=" * 60)
    print(f"\n{encrypted}\n")


if __name__ == "__main__":
    main()
