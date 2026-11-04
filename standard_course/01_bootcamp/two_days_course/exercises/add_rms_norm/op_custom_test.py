import json
import os
import subprocess
import sys
from pathlib import Path

print('[环境准备] 开始', flush=True)
os.environ.setdefault('QWEN3_1_7B_DOWNLOAD_BACKEND', 'modelscope')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')

def locate_repo_root():
    repo_roots = []
    for env_name in ('GITCODE_REPO_ROOT', 'CANN_LEARNING_HUB_ROOT', 'REPO_ROOT'):
        if os.environ.get(env_name):
            repo_roots.append(Path(os.environ[env_name]))
    try:
        cwd = Path.cwd()
        lineage = [cwd, *cwd.parents]
        repo_roots.extend(lineage)
        repo_roots.extend(base / 'cann-learning-hub' for base in lineage)
        for base in lineage:
            try:
                repo_roots.extend(base.glob('*/cann-learning-hub'))
            except OSError:
                pass
    except FileNotFoundError:
        pass
    repo_roots.append(Path('/opt/atomgit/cann-learning-hub'))

    seen = set()
    for repo_root in repo_roots:
        key = str(repo_root)
        if key in seen:
            continue
        seen.add(key)
        if (repo_root / 'tutorials/llm_inference/qwen3_1.7B/src').exists():
            return repo_root
    raise FileNotFoundError('Cannot locate cann-learning-hub repository root. Open this notebook inside cann-learning-hub or set CANN_LEARNING_HUB_ROOT to the repository root.')

REPO_ROOT = locate_repo_root()
os.chdir(REPO_ROOT)
TUTORIAL_DIR = REPO_ROOT / 'tutorials' / 'llm_inference' / 'qwen3_1.7B'
SRC_DIR = TUTORIAL_DIR / 'src'
SOURCE_DIR = REPO_ROOT / 'Sources' / 'model_inference_optimization' / 'qwen3_1.7B'
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from inference_scripts.environment_check import require_unified_environment
require_unified_environment(TUTORIAL_DIR / 'requirements.txt')

if not os.environ.get('ASCEND_TOOLKIT_HOME'):
    raise EnvironmentError('ASCEND_TOOLKIT_HOME is not set. Please activate the CANN environment before running this notebook.')
cann_script = Path(os.environ['ASCEND_TOOLKIT_HOME']) / 'set_env.sh'
env_cmd = f'source {cann_script} && env'
env = subprocess.check_output(f"bash -lc '{env_cmd}'", shell=True, text=True, cwd=REPO_ROOT)
for line in env.splitlines():
    if '=' in line:
        os.environ.__setitem__(*line.split('=', 1))

RECIPE_ROOT = SRC_DIR / 'inference_scripts' / 'recipe_qwen3_1_7b'
RECIPE_CONFIG_DIR = RECIPE_ROOT / 'models' / 'qwen' / 'config'
BASELINE_YAML_TEMPLATE = RECIPE_CONFIG_DIR / 'qwen3_1_7b_1tp.yaml'
PROFILE_YAML_TEMPLATE = RECIPE_CONFIG_DIR / 'qwen3_1_7b_1tp_profile.yaml'
OPTIMIZED_YAML_TEMPLATE = RECIPE_CONFIG_DIR / 'qwen3_1_7b_1tp_add_rmsnorm.yaml'
INFER_SH = RECIPE_ROOT / 'executor' / 'scripts' / 'infer.sh'
MODEL_ID = os.environ.get('QWEN3_1_7B_MODEL_PATH') or 'Qwen/Qwen3-1.7B'
MODEL_SOURCE_HINT = 'local_path' if os.environ.get('QWEN3_1_7B_MODEL_PATH') else os.environ['QWEN3_1_7B_DOWNLOAD_BACKEND']

print('仓库目录 =', REPO_ROOT)
print('教程目录 =', TUTORIAL_DIR)
print('运行目录 =', SOURCE_DIR)
print('recipes 目录 =', RECIPE_ROOT)
print('YAML 目录 =', RECIPE_CONFIG_DIR)
print('推理入口 =', INFER_SH)
print('可见 NPU =', os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
print('[环境准备] 完成', flush=True)

print('[Dense RMSNorm 精度检查] 开始')
from qwen3_npu_adaptation import run_add_rmsnorm_precision_check, run_rmsnorm_precision_check

precision_rows = {
    'RMSNorm': run_rmsnorm_precision_check(
        shapes=([1, 1, 2048], [1, 16, 2048], [1, 16, 32, 128]),
        dtype='bfloat16',
        device='npu:0',
        seed=2026,
    ),
    'Add+RMSNorm': run_add_rmsnorm_precision_check(
        shapes=([1, 1, 2048], [1, 16, 2048], [1, 16, 32, 128]),
        dtype='bfloat16',
        device='npu:0',
        seed=2026,
    ),
}
all_precision_rows = [row for group_rows in precision_rows.values() for row in group_rows]
PRECISION_CHECK_OK = all(row.get('status') == 'ok' for row in all_precision_rows)
PRECISION_CHECK_HAS_WARNING = any(row.get('status') == 'warning' for row in all_precision_rows)
PRECISION_CHECK_HARD_FAILED = any(row.get('status') == 'failed' for row in all_precision_rows)
PRECISION_CHECK_CAN_RUN = not PRECISION_CHECK_HARD_FAILED
precision_path = SOURCE_DIR / 'run_outputs' / 'add_rmsnorm_precision_check.json'
precision_path.parent.mkdir(parents=True, exist_ok=True)
precision_path.write_text(json.dumps(precision_rows, indent=2, ensure_ascii=False), encoding='utf-8')
for group_name, group_rows in precision_rows.items():
    ok_count = sum(row.get('status') == 'ok' for row in group_rows)
    warning_count = sum(row.get('status') == 'warning' for row in group_rows)
    failed_count = sum(row.get('status') == 'failed' for row in group_rows)
    max_diff = max(
        row.get('max_abs_diff', row.get('max_norm_abs_diff', 0.0))
        for row in group_rows
    )
    print(f'{group_name}: 通过 {ok_count}/{len(group_rows)}，告警 {warning_count}，失败 {failed_count}，最大误差={max_diff}')
print('精度阈值 = 0.01')
print('精度检查通过 =', '是' if PRECISION_CHECK_OK else '否')
print('精度检查告警 =', '是' if PRECISION_CHECK_HAS_WARNING else '否')
print('存在硬失败 =', '是' if PRECISION_CHECK_HARD_FAILED else '否')
print('继续 A/B =', '是' if PRECISION_CHECK_CAN_RUN else '否')
print('精度检查文件 =', precision_path)
print('[Dense RMSNorm 精度检查] 完成')

from inference_scripts.recipe_workflow import (
    prepare_runtime_yaml,
    resolve_model_path,
)
BASELINE_WORK_DIR = SOURCE_DIR / 'run_outputs' / 'baseline_demo'
BASELINE_YAML = SOURCE_DIR / 'recipe_yaml' / 'baseline_demo.yaml'
BASELINE_WORK_DIR.mkdir(parents=True, exist_ok=True)
resolved_model_path, model_source = resolve_model_path(str(MODEL_ID), quiet_model_io=False)
prepare_runtime_yaml(BASELINE_YAML_TEMPLATE, BASELINE_YAML, model_path=resolved_model_path)

print('[推理] 开始')
BASELINE_DEMO_YAML = SOURCE_DIR / 'recipe_yaml' / 'baseline_demo.yaml'
BASELINE_DEMO_LOG = (
    SOURCE_DIR / 'run_outputs' / 'baseline_demo' / 'recipe_logs'
    / 'baseline_demo_infer_stdout.log'
)
BASELINE_DEMO_LOG_REL = BASELINE_DEMO_LOG.relative_to(REPO_ROOT)
INFER_SH_REL = INFER_SH.relative_to(REPO_ROOT)
BASELINE_DEMO_LOG.parent.mkdir(parents=True, exist_ok=True)
if not BASELINE_DEMO_YAML.is_file():
    raise FileNotFoundError(f'Baseline demo yaml not found: {BASELINE_DEMO_YAML}')

baseline_infer_command = (
    'set -o pipefail; '
    f'bash "{INFER_SH_REL}" --model qwen --mode offline '
    f'--yaml "{BASELINE_DEMO_YAML.resolve()}" 2>&1 '
    f'| tee "{BASELINE_DEMO_LOG_REL}"'
)
subprocess.run(
    baseline_infer_command,
    shell=True,
    executable='/bin/bash',
    cwd=REPO_ROOT,
    check=True,
)
print(f'[Baseline 推理] 完成，日志 = {BASELINE_DEMO_LOG}')