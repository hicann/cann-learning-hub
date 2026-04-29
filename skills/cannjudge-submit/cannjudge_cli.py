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
from pathlib import Path


class CANNJudgeClient:
    """CANNJudge API 客户端"""
    
    BASE_URL = "https://cannjudge.cn"
    
    def __init__(self):
        self.session = requests.Session()
        self.user_id = None
        self.user_info = None
    
    def login(self, email: str, password: str) -> dict:
        """登录 CANNJudge"""
        resp = self.session.post(
            f"{self.BASE_URL}/api/users/login",
            json={"email": email, "password": password},
            timeout=30
        )
        resp.raise_for_status()
        
        self.user_info = resp.json()
        self.user_id = self.user_info["_id"]
        
        return self.user_info
    
    def get_problem(self, problem_id: str) -> dict:
        """获取题目信息"""
        resp = self.session.get(
            f"{self.BASE_URL}/api/problems/{problem_id}",
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    
    def download_package(self, problem_id: str, output_dir: str = None) -> str:
        """下载工程模板"""
        resp = self.session.get(
            f"{self.BASE_URL}/api/problems/{problem_id}/package",
            params={"userId": self.user_id},
            timeout=60
        )
        resp.raise_for_status()
        
        # 保存 zip 文件
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        zip_path = os.path.join(output_dir, "project.zip")
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        
        # 解压
        extract_dir = os.path.join(output_dir, "project")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        
        return extract_dir
    
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
        
        resp = self.session.post(
            f"{self.BASE_URL}/api/submissions/submit",
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        
        result = resp.json()
        return result["data"]["submissionId"]
    
    def get_submission(self, submission_id: str) -> dict:
        """查询提交结果"""
        resp = self.session.get(
            f"{self.BASE_URL}/api/submissions/{submission_id}",
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    
    def wait_for_result(self, submission_id: str, 
                        max_wait: int = 120, interval: int = 3) -> dict:
        """等待提交完成并返回结果"""
        for i in range(max_wait // interval):
            time.sleep(interval)
            result = self.get_submission(submission_id)
            status = result.get("status")
            
            if status in ["Accepted", "Wrong Answer", 
                          "Compile Error", "Runtime Error",
                          "Time Limit Exceeded"]:
                return result
        
        return self.get_submission(submission_id)
    
    def get_rankings(self, problem_id: str) -> list:
        """获取排行榜"""
        resp = self.session.get(
            f"{self.BASE_URL}/api/submissions/problem/{problem_id}/latest",
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()


def print_submission_result(result: dict):
    """打印提交结果"""
    status = result.get("status", "Unknown")
    print(f"\n{'='*60}")
    print(f"状态: {status}")
    print(f"{'='*60}")
    
    if "result" in result:
        print("\n测试用例结果:")
        for idx, tc in enumerate(result["result"]):
            tc_status = tc.get("testcase_status", "Unknown")
            precision = tc.get("precision_ratio", "N/A")
            time_ms = tc.get("time", "N/A")
            print(f"  Case {idx+1}: {tc_status}")
            print(f"           precision: {precision}, time: {time_ms}ms")


def main():
    parser = argparse.ArgumentParser(
        description="CANNJudge 算子提交辅助工具"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 登录命令
    login_parser = subparsers.add_parser("login", help="登录 CANNJudge")
    login_parser.add_argument("--email", required=True, help="登录邮箱")
    login_parser.add_argument("--password", required=True, help="登录密码")
    
    # 下载命令
    download_parser = subparsers.add_parser("download", help="下载工程模板")
    download_parser.add_argument("--problem-id", required=True, help="题目ID")
    download_parser.add_argument("--output", help="输出目录")
    
    # 提交命令
    submit_parser = subparsers.add_parser("submit", help="提交代码")
    submit_parser.add_argument("--problem-id", required=True, help="题目ID")
    submit_parser.add_argument("--project-dir", required=True, help="工程目录")
    
    # 查询命令
    query_parser = subparsers.add_parser("query", help="查询提交结果")
    query_parser.add_argument("--submission-id", required=True, help="提交ID")
    
    # 排行榜命令
    rank_parser = subparsers.add_parser("rank", help="查看排行榜")
    rank_parser.add_argument("--problem-id", required=True, help="题目ID")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = CANNJudgeClient()
    
    # 从环境变量或配置文件读取登录信息
    email = os.environ.get("CANNJUDGE_EMAIL", "")
    password = os.environ.get("CANNJUDGE_PASSWORD", "")
    
    if email and password:
        print("使用环境变量中的登录信息...")
        client.login(email, password)
    
    if args.command == "login":
        result = client.login(args.email, args.password)
        print(f"登录成功!")
        print(f"用户ID: {result['_id']}")
        print(f"昵称: {result.get('nickname', 'N/A')}")
        
    elif args.command == "download":
        if not client.user_id:
            print("错误: 请先登录")
            return
        
        print(f"下载题目 {args.problem_id} 的工程模板...")
        extract_dir = client.download_package(args.problem_id, args.output)
        print(f"工程已解压到: {extract_dir}")
        
    elif args.command == "submit":
        if not client.user_id:
            print("错误: 请先登录")
            return
        
        # 读取代码文件
        project_dir = Path(args.project_dir)
        
        # 查找 op_kernel 和 op_host 目录
        code_dir = project_dir / "code" if (project_dir / "code").exists() else project_dir
        
        tiling_h_path = list(code_dir.glob("op_kernel/*_tiling.h"))
        tiling_key_h_path = list(code_dir.glob("op_kernel/tiling_key_*.h"))
        host_cpp_path = list(code_dir.glob("op_host/*.cpp"))
        kernel_cpp_path = list(code_dir.glob("op_kernel/*.cpp"))
        
        # 过滤掉 tiling 相关的 cpp
        kernel_cpp_path = [p for p in kernel_cpp_path if "_tiling" not in p.name]
        
        if not all([tiling_h_path, host_cpp_path, kernel_cpp_path]):
            print("错误: 找不到必要的代码文件")
            return
        
        # 读取文件内容
        tiling_h = tiling_h_path[0].read_text()
        tiling_key_h = tiling_key_h_path[0].read_text() if tiling_key_h_path else ""
        host_cpp = host_cpp_path[0].read_text()
        kernel_cpp = kernel_cpp_path[0].read_text()
        
        print(f"提交代码到题目 {args.problem_id}...")
        submission_id = client.submit(
            args.problem_id, tiling_h, tiling_key_h, host_cpp, kernel_cpp
        )
        print(f"提交成功! Submission ID: {submission_id}")
        
        print("\n等待结果...")
        result = client.wait_for_result(submission_id)
        print_submission_result(result)
        
    elif args.command == "query":
        if not client.user_id:
            print("错误: 请先登录")
            return
        
        result = client.get_submission(args.submission_id)
        print_submission_result(result)
        
    elif args.command == "rank":
        if not client.user_id:
            print("错误: 请先登录")
            return
        
        rankings = client.get_rankings(args.problem_id)
        
        print(f"\n题目 {args.problem_id} 排行榜:")
        print("=" * 60)
        
        # 按状态和分数排序
        sorted_rankings = sorted(
            rankings,
            key=lambda x: (x.get("status") == "Accepted", x.get("score", 0)),
            reverse=True
        )
        
        for i, item in enumerate(sorted_rankings[:20]):
            user_id = item.get("user_id", "unknown")
            status = item.get("status", "unknown")
            score = item.get("score", "N/A")
            print(f"{i+1:3d}. {user_id:24s} | {status:15s} | score: {score}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
