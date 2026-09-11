#!/usr/bin/env python3
"""User-run interactive example using the same validated SDK as the CLI."""
from cannjudge_cli import CANNJudgeClient, print_submission_result


def example_workflow():
    client = CANNJudgeClient()
    if not client.load_session():
        print('请先在自己的终端运行: python3 cannjudge_cli.py login --captcha')
        return
    source = input('题目完整链接或 ID: ').strip()
    problem = client.resolve_problem(source)
    print(problem.get('title', ''), problem.get('code_template', 'registry'))
    print(problem.get('desc', ''))
    output = input('下载输出目录（请使用新目录）: ').strip()
    directory = client.download_package(problem['_id'], output, include_metadata=True)
    print(f'工程已下载: {directory}')
    print('根据模板完成可编辑源码。核函数题保留 run_kernel 接口和只读框架。')
    input('完成开发和检查后按 Enter 继续: ')
    payload = client.prepare_submission(problem['_id'], directory)
    print('提交文件:', [f['path'] for f in payload.get('files', [])] or '传统工程四字段')
    submission_id = client.submit_payload(payload)
    print('提交ID:', submission_id)
    print_submission_result(client.wait_for_result(submission_id))
    client.save_session()


if __name__ == '__main__':
    example_workflow()
