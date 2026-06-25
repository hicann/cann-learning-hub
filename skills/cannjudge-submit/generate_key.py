#!/usr/bin/env python3
"""
RSA 密钥对生成脚本

在服务器上运行一次，生成 RSA-2048 密钥对：
  - private.pem → 留在服务器，用于解密密码（绝不外传）
  - public.pem  → 拷贝到个人 PC，用于加密密码

用法：
    python3 generate_key.py

依赖：
    pip install pycryptodome
"""

from Crypto.PublicKey import RSA
import os

PRIVATE_KEY_PATH = "private.pem"
PUBLIC_KEY_PATH = "public.pem"


def generate_keys():
    if os.path.exists(PRIVATE_KEY_PATH) and os.path.exists(PUBLIC_KEY_PATH):
        print(f"密钥文件已存在：{PRIVATE_KEY_PATH}, {PUBLIC_KEY_PATH}")
        print("如需重新生成，请先删除旧文件。")
        return

    key = RSA.generate(2048)

    private_key = key.export_key()
    with open(PRIVATE_KEY_PATH, "wb") as f:
        f.write(private_key)

    public_key = key.publickey().export_key()
    with open(PUBLIC_KEY_PATH, "wb") as f:
        f.write(public_key)

    print("=" * 60)
    print("RSA-2048 密钥对已生成")
    print("=" * 60)
    print(f"\n私钥已保存到: {PRIVATE_KEY_PATH}  (仅保留在服务器，绝不外传)")
    print(f"公钥已保存到: {PUBLIC_KEY_PATH}  (拷贝到个人 PC 用于加密)")
    print(f"\n公钥内容（也可直接复制下方内容到 PC）：\n")
    print(public_key.decode())


if __name__ == "__main__":
    generate_keys()
