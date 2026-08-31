#!/usr/bin/env python3
"""moe_router_fused 正确性回归测试（M4）

遍历 data/case_*/ 用例，逐个调用 C++ aclnn 测试程序（src/custom_op/test/run.sh），
解析其 [idx]/[wt] 统计行与 PASS/FAIL 结论，汇总输出。

用法:
    python3 tools/test_moe_router.py                 # 全部用例
    python3 tools/test_moe_router.py --case case_128_512_16_2
    python3 tools/test_moe_router.py --device 0 --repeats 3   # 每用例重复 3 次（确定性检查）

前置条件:
    - 已 source $ASCEND_HOME_PATH/set_env.sh
    - 已执行 src/custom_op/build.sh 生成算子包
退出码: 全部 PASS 返回 0，否则 1。
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUN_SH = ROOT / "src" / "custom_op" / "test" / "run.sh"

IDX_RE = re.compile(r"\[idx \] match=(\d+)/(\d+) tie_diff=(\d+) real_diff=(\d+)")
WT_RE = re.compile(r"\[wt  \] max_abs=([\d.eE+-]+) max_rel=([\d.eE+-]+) fail=(\d+)/(\d+)")
TILING_RE = re.compile(r"blockDim=(\d+) rowsPerCore=(\d+)")


def run_case(case_dir: Path, device: int) -> dict:
    """运行单个用例，解析测试程序输出。"""
    proc = subprocess.run(
        ["bash", str(RUN_SH), str(case_dir), "--device", str(device)],
        capture_output=True, text=True, timeout=600,
    )
    out = proc.stdout + proc.stderr
    # 以测试程序退出码为准（main.cpp: PASS 返回 0，FAIL 返回 1）；
    # 不用 endswith("PASS")——stderr 的 tiling 日志可能排在输出末尾
    info = {"name": case_dir.name, "pass": proc.returncode == 0, "raw": out}
    m = TILING_RE.search(out)
    if m:
        info["blockDim"], info["rowsPerCore"] = int(m.group(1)), int(m.group(2))
    m = IDX_RE.search(out)
    if m:
        info["idx_match"], info["idx_total"] = int(m.group(1)), int(m.group(2))
        info["idx_tie"], info["idx_realdiff"] = int(m.group(3)), int(m.group(4))
    m = WT_RE.search(out)
    if m:
        info["wt_max_abs"], info["wt_max_rel"] = float(m.group(1)), float(m.group(2))
        info["wt_fail"], info["wt_total"] = int(m.group(3)), int(m.group(4))
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="moe_router_fused accuracy regression")
    ap.add_argument("--case", help="只跑指定用例名（如 case_128_512_16_2）")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1, help="每用例重复次数（确定性检查）")
    args = ap.parse_args()

    if not DATA_DIR.is_dir():
        print(f"[FATAL] 未找到数据目录 {DATA_DIR}，请先运行 tools/gen_test_data.py", file=sys.stderr)
        return 2
    cases = sorted(DATA_DIR.glob("case_*"))
    if args.case:
        cases = [c for c in cases if c.name == args.case]
    if not cases:
        print("[FATAL] 没有匹配的用例", file=sys.stderr)
        return 2

    print(f"{'case':<24} {'tiling':<18} {'idx':<16} {'wt_max_abs':<12} result")
    n_fail = 0
    for case_dir in cases:
        for r in range(args.repeats):
            info = run_case(case_dir, args.device)
            tiling = f"bd={info.get('blockDim', '?')} rpc={info.get('rowsPerCore', '?')}"
            idx = f"{info.get('idx_match', '?')}/{info.get('idx_total', '?')}"
            wt = f"{info.get('wt_max_abs', float('nan')):.2e}"
            tag = "PASS" if info["pass"] else "FAIL"
            rep = f" (r{r + 1})" if args.repeats > 1 else ""
            print(f"{case_dir.name:<24} {tiling:<18} {idx:<16} {wt:<12} {tag}{rep}")
            if not info["pass"]:
                n_fail += 1
                print("  ---- raw output ----")
                print("  " + info["raw"].replace("\n", "\n  "))

    total = len(cases) * args.repeats
    print(f"\n[summary] {total - n_fail}/{total} PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
