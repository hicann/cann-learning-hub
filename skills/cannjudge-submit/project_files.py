"""CANNJudge project packaging. Only editable source files enter submissions."""

import hashlib
import io
import json
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

MANIFEST = '.cannjudge-project.json'
PROTECTED_KERNEL_FILES = {'main.asc', 'judge.asc', 'data_utils.h'}
KERNEL_NAME = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.(asc|h)\Z')


def relative_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value or '\0' in value:
        raise ValueError('工程文件路径无效')
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('工程文件路径必须位于工程目录中')
    return path.as_posix()


def template_files(template):
    data = template.get('data', template)
    items = data.get('files', [])
    if not isinstance(items, list):
        raise ValueError('模板 files 格式错误')
    result = {}
    for item in items:
        path = relative_path(item.get('path') or item.get('key'))
        if path.casefold() in {key.casefold() for key in result}:
            raise ValueError('模板含重复文件路径')
        if not isinstance(item.get('content'), str):
            raise ValueError('模板文件缺少文本内容')
        result[path] = dict(item, path=path, editable=item.get('editable') is not False)
    return result


def extract_package(content, output_dir, problem=None, template=None):
    output = Path(output_dir)
    destination = output / 'project'
    if destination.exists() or (output / 'project.zip').exists():
        raise ValueError('输出目录已有 project 或 project.zip，请选择新目录以免覆盖代码')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        total = 0
        seen = set()
        for info in archive.infolist():
            path = relative_path(info.filename.rstrip('/'))
            if path.casefold() in seen:
                raise ValueError('压缩包含重复文件路径')
            seen.add(path.casefold())
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('压缩包不允许符号链接')
            total += info.file_size
            if total > 256 * 1024 * 1024:
                raise ValueError('解压工程超过 256 MiB，请手动检查下载包')
        output.mkdir(parents=True, exist_ok=True)
        (output / 'project.zip').write_bytes(content)
        destination.mkdir()
        archive.extractall(destination)
    if problem is not None:
        files = template_files(template or {})
        (destination / MANIFEST).write_text(json.dumps({
            'version': 1, 'problem_id': problem['_id'],
            'code_template': problem.get('code_template'),
            'files': [{'path': path, 'editable': item['editable'],
                       'sha256': hashlib.sha256(item['content'].encode('utf-8')).hexdigest()}
                      for path, item in files.items()],
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(destination)


def locate_root(project_dir, required_paths):
    base = Path(project_dir).resolve()
    if not base.is_dir():
        raise ValueError('工程目录不存在')
    candidates = [base, base / 'code', base / 'project', base / 'project' / 'code']
    candidates.extend(p for p in base.iterdir() if p.is_dir() and not p.is_symlink())
    matches = []
    for candidate in dict.fromkeys(candidates):
        if all((candidate / p).is_file() for p in required_paths):
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError('未找到唯一工程根目录，请指向包含所有可编辑模板文件的目录')
    return matches[0]


def read_source(root, name):
    path = root / relative_path(name)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'源码文件越出工程目录: {name}')
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError(f'源码文件过大: {name}')
    return path.read_text(encoding='utf-8-sig')


def collect_kernel_files(project_dir, template, problem_id):
    items = template_files(template)
    required = [p for p, item in items.items() if item['editable']]
    if not required:
        raise ValueError('核函数模板没有可编辑文件，停止提交')
    base = Path(project_dir).resolve()
    for manifest in (base / MANIFEST, base.parent / MANIFEST):
        if manifest.is_file():
            if json.loads(manifest.read_text(encoding='utf-8')).get('problem_id') != problem_id:
                raise ValueError('工程所属题目与提交目标不同')
    root = locate_root(base, required)
    for name, item in items.items():
        if not item['editable'] and (root / name).is_file():
            # Newline normalization permits Git/Windows checkouts.
            if read_source(root, name).replace('\r\n', '\n') != item['content'].replace('\r\n', '\n'):
                raise ValueError(f'平台只读模板文件已修改: {name}')
    names = set(required)
    for path in root.iterdir():
        if path.name in items or not path.is_file():
            continue
        if path.name in PROTECTED_KERNEL_FILES:
            raise ValueError(f'禁止添加评测框架文件: {path.name}')
        if path.suffix.lower() in {'.asc', '.h'}:
            if len(path.name) > 128 or not KERNEL_NAME.fullmatch(path.name):
                raise ValueError(f'不支持的核函数文件名: {path.name}')
            names.add(path.name)
    files = []
    for name in sorted(names):
        if name in PROTECTED_KERNEL_FILES:
            raise ValueError(f'禁止提交评测框架文件: {name}')
        if not KERNEL_NAME.fullmatch(name) or len(name) > 128:
            raise ValueError(f'核函数源码必须是根目录的 .asc 或 .h 文件: {name}')
        content = read_source(root, name)
        if not content.strip():
            raise ValueError(f'可编辑源码为空: {name}')
        files.append({'path': name, 'content': content})
    return files


def collect_registry_files(project_dir):
    root = Path(project_dir).resolve()
    if (root / 'code').is_dir():
        root /= 'code'
    patterns = {'tiling_h': 'op_kernel/*_tiling.h', 'tiling_key_h': 'op_kernel/tiling_key_*.h',
                'host_cpp': 'op_host/*.cpp', 'kernel_cpp': 'op_kernel/*.cpp'}
    result = {}
    for key, pattern in patterns.items():
        matches = sorted(root.glob(pattern))
        if key == 'kernel_cpp':
            matches = [p for p in matches if '_tiling' not in p.name]
        if not matches and key == 'tiling_key_h':
            result[key] = ''
        elif len(matches) != 1:
            raise ValueError(f'传统工程 {key} 必须有唯一匹配文件，实际 {len(matches)} 个')
        else:
            result[key] = read_source(root, matches[0].relative_to(root).as_posix())
    return result
