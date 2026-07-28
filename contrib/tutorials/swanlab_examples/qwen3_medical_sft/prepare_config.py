#!/usr/bin/env python3
"""
将训练产出的 checkpoint 路径写入统一执行器运行时 YAML 的 model_path 字段。

用法：
    python3 prepare_config.py <checkpoint绝对路径> [目标yaml路径]

示例：
    python3 prepare_config.py /home/project/qwen3_medical_sft/output_qwen3_medical/final \
        cann-recipes-infer/models/qwen/config/qwen3_custom_1tp.yaml
"""

import re
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("错误：请提供 checkpoint 绝对路径作为第一个参数")
        print(__doc__)
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    yaml_path = sys.argv[2] if len(sys.argv) > 2 else \
        "cann-recipes-infer/models/qwen/config/qwen3_custom_1tp.yaml"

    if not os.path.isfile(yaml_path):
        print(f"错误：找不到 YAML 文件 {yaml_path}")
        print("请先执行： cp cann-recipes-infer/models/qwen/config/qwen3_8b_1tp.yaml "
              f"{yaml_path}")
        sys.exit(1)

    with open(yaml_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = re.sub(
        r'model_path:\s*".*"',
        f'model_path: "{checkpoint_path}"',
        content,
    )

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"已将 {yaml_path} 中的 model_path 写入: {checkpoint_path}")
    for line in new_content.splitlines():
        if "model_path" in line:
            print(line.strip())


if __name__ == "__main__":
    main()
