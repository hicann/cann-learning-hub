#!/bin/bash
# 昇腾香橙派 310B 一键运行脚本
# 用法: bash run_orangepi.sh [stage]
#   stage: 1=加法8核, 2=加法32核, 3=三向量加法, 4=乘法, all=全部运行

cd "$(dirname "$0")"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

case "${1:-all}" in
    1)
        echo "===== 阶段一：双向量加法（8核）====="
        python3 add_8core_orangepi.py
        ;;
    2)
        echo "===== 阶段二：双向量加法（32核）====="
        python3 add_32core_orangepi.py
        ;;
    3)
        echo "===== 阶段三：三向量加法 ====="
        python3 add3_orangepi.py
        ;;
    4)
        echo "===== 阶段四：双向量乘法 ====="
        python3 mul_orangepi.py
        ;;
    all)
        echo "===== 阶段一：双向量加法（8核）====="
        python3 add_8core_orangepi.py
        echo ""
        echo "===== 阶段二：双向量加法（32核）====="
        python3 add_32core_orangepi.py
        echo ""
        echo "===== 阶段三：三向量加法 ====="
        python3 add3_orangepi.py
        echo ""
        echo "===== 阶段四：双向量乘法 ====="
        python3 mul_orangepi.py
        ;;
    *)
        echo "用法: bash run_orangepi.sh [1|2|3|4|all]"
        exit 1
        ;;
esac
