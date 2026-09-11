# RSA 密文登录（兼容方式）

仅在用户需要通过密文向执行环境提供密码、无法使用交互终端时阅读。常规使用请先尝试已保存的会话，或让用户在自己的终端运行 `login --email 邮箱`。

RSA 密文登录仍沿用现有格式，需要 `pip install pycryptodome`。

1. 在执行环境运行 `python3 generate_key.py`，生成 `private.pem` 和 `public.pem`。
2. 私钥留在执行环境；只把公钥和 `encrypt_password.py` 下载到个人电脑。
3. 在个人电脑运行 `python3 encrypt_password.py --public-key public.pem`，按提示隐藏输入密码。
4. 使用生成的密文登录：

```bash
python3 cannjudge_cli.py login --email "your@email.com" --ciphertext "RSA密文" --private-key private.pem
```

登录成功后同样保存会话，后续下载、提交、查询和排行榜命令不需要再次提供密文。会话过期后，可在密码和密钥均未改变时复用原密文。

不要把明文密码输入聊天、命令参数、环境变量或写入文件。私钥和会话文件不应提交到仓库；密文也应作为敏感凭据保存。

密钥丢失时可以改用终端隐藏输入登录。若继续使用 RSA，在执行环境重新生成密钥并重新加密；旧密文不能使用新私钥解密。
