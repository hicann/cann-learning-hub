#!/usr/bin/env python3
"""
CANNJudge 算子提交示例

展示如何使用 CANNJudge API 完成完整的提交流程。
"""

import requests
import time
import os
import zipfile
from pathlib import Path


def example_workflow():
    """
    完整的 CANNJudge 算子提交流程示例
    """
    
    # ========================================
    # 步骤 1: 登录
    # ========================================
    print("步骤 1: 登录 CANNJudge")
    print("-" * 60)
    
    # 注意: 实际使用时从用户输入或配置文件读取
    email = input("请输入邮箱: ")
    password = input("请输入密码: ")
    
    session = requests.Session()
    
    resp = session.post(
        "https://cannjudge.cn/api/users/login",
        json={"email": email, "password": password},
        timeout=30
    )
    
    if resp.status_code != 200:
        print(f"登录失败: {resp.text}")
        return
    
    user_info = resp.json()
    user_id = user_info["_id"]
    print(f"登录成功! 用户ID: {user_id}")
    
    # ========================================
    # 步骤 2: 获取题目信息
    # ========================================
    print("\n步骤 2: 获取题目信息")
    print("-" * 60)
    
    problem_id = input("请输入题目ID: ")
    
    resp = session.get(
        f"https://cannjudge.cn/api/problems/{problem_id}",
        timeout=30
    )
    
    if resp.status_code != 200:
        print(f"获取题目失败: {resp.text}")
        return
    
    problem_info = resp.json()
    print(f"题目名称: {problem_info.get('title', 'N/A')}")
    print(f"题目描述: {problem_info.get('desc', 'N/A')[:200]}...")
    
    # ========================================
    # 步骤 3: 下载工程模板
    # ========================================
    print("\n步骤 3: 下载工程模板")
    print("-" * 60)
    
    resp = session.get(
        f"https://cannjudge.cn/api/problems/{problem_id}/package",
        params={"userId": user_id},
        timeout=60
    )
    
    if resp.status_code != 200:
        print(f"下载失败: {resp.text}")
        return
    
    # 保存并解压
    output_dir = f"./{problem_info.get('name', 'project')}"
    os.makedirs(output_dir, exist_ok=True)
    
    zip_path = os.path.join(output_dir, "project.zip")
    with open(zip_path, "wb") as f:
        f.write(resp.content)
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    
    print(f"工程已下载到: {output_dir}")
    print("\n工程结构:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    # ========================================
    # 步骤 4: 实现算子
    # ========================================
    print("\n步骤 4: 实现算子")
    print("-" * 60)
    print(f"请在 {output_dir}/code 目录中实现算子")
    print("需要实现的文件:")
    print("  - op_kernel/*_tiling.h: Tiling 数据结构")
    print("  - op_host/*.cpp: Host 侧实现")
    print("  - op_kernel/*.cpp: Kernel 侧实现")
    
    input("\n实现完成后按 Enter 继续...")
    
    # ========================================
    # 步骤 5: 编译工程
    # ========================================
    print("\n步骤 5: 编译工程")
    print("-" * 60)
    
    code_dir = os.path.join(output_dir, "code")
    
    # 检查是否有 build.sh
    build_sh = os.path.join(code_dir, "build.sh")
    if os.path.exists(build_sh):
        print("找到 build.sh，请手动执行编译:")
        print(f"  cd {code_dir} && bash build.sh")
        input("编译完成后按 Enter 继续...")
    else:
        print("未找到 build.sh，请手动编译工程")
        input("编译完成后按 Enter 继续...")
    
    # ========================================
    # 步骤 6: 提交代码
    # ========================================
    print("\n步骤 6: 提交代码")
    print("-" * 60)
    
    # 查找代码文件
    code_path = Path(code_dir)
    
    tiling_h_files = list(code_path.glob("op_kernel/*_tiling.h"))
    tiling_key_h_files = list(code_path.glob("op_kernel/tiling_key_*.h"))
    host_cpp_files = list(code_path.glob("op_host/*.cpp"))
    kernel_cpp_files = [f for f in code_path.glob("op_kernel/*.cpp") 
                        if "_tiling" not in f.name]
    
    if not all([tiling_h_files, host_cpp_files, kernel_cpp_files]):
        print("错误: 找不到必要的代码文件")
        return
    
    # 读取文件内容
    tiling_h = tiling_h_files[0].read_text()
    tiling_key_h = tiling_key_h_files[0].read_text() if tiling_key_h_files else ""
    host_cpp = host_cpp_files[0].read_text()
    kernel_cpp = kernel_cpp_files[0].read_text()
    
    # 提交
    resp = session.post(
        "https://cannjudge.cn/api/submissions/submit",
        json={
            "problemId": problem_id,
            "userId": user_id,
            "tiling_h": tiling_h,
            "tiling_key_h": tiling_key_h,
            "host_cpp": host_cpp,
            "kernel_cpp": kernel_cpp
        },
        timeout=30
    )
    
    if resp.status_code != 200:
        print(f"提交失败: {resp.text}")
        return
    
    result = resp.json()
    submission_id = result["data"]["submissionId"]
    print(f"提交成功! Submission ID: {submission_id}")
    
    # ========================================
    # 步骤 7: 等待结果
    # ========================================
    print("\n步骤 7: 等待结果")
    print("-" * 60)
    
    for i in range(40):
        time.sleep(3)
        
        resp = session.get(
            f"https://cannjudge.cn/api/submissions/{submission_id}",
            timeout=30
        )
        
        if resp.status_code == 200:
            result = resp.json()
            status = result.get("status", "unknown")
            
            print(f"轮询 {i+1}: {status}")
            
            if status in ["Accepted", "Wrong Answer", 
                          "Compile Error", "Runtime Error"]:
                print(f"\n{'='*60}")
                print(f"最终状态: {status}")
                print(f"{'='*60}")
                
                if "result" in result:
                    print("\n测试用例结果:")
                    for idx, tc in enumerate(result["result"]):
                        tc_status = tc.get("testcase_status", "Unknown")
                        precision = tc.get("precision_ratio", "N/A")
                        time_ms = tc.get("time", "N/A")
                        print(f"  Case {idx+1}: {tc_status}")
                        print(f"           precision: {precision}, time: {time_ms}ms")
                
                break
    else:
        print("\n超时: 未能获取最终结果")
    
    # ========================================
    # 步骤 8: 查看排行榜
    # ========================================
    print("\n步骤 8: 查看排行榜")
    print("-" * 60)
    
    resp = session.get(
        f"https://cannjudge.cn/api/submissions/problem/{problem_id}/latest",
        timeout=30
    )
    
    if resp.status_code == 200:
        rankings = resp.json()
        
        # 按状态和分数排序
        sorted_rankings = sorted(
            rankings,
            key=lambda x: (x.get("status") == "Accepted", x.get("score", 0)),
            reverse=True
        )
        
        print(f"\n排行榜 (前20名):")
        for i, item in enumerate(sorted_rankings[:20]):
            user = item.get("user_id", "unknown")
            status = item.get("status", "unknown")
            score = item.get("score", "N/A")
            print(f"{i+1:3d}. {user:24s} | {status:15s} | score: {score}")


if __name__ == "__main__":
    example_workflow()
