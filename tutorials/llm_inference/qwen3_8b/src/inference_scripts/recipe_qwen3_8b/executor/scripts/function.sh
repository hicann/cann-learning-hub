# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the CANN Open Software License Agreement Version 2.0.

#!/bin/bash
set -o pipefail

function launch()
{
    check_launch
    get_rank
    check_env_vars
    set_hccl
    launch_infer_task
}

function check_launch()
{
    SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
    PARENT_PARENT_DIR=$(cd "$SCRIPT_PATH/../.." &>/dev/null && pwd)
    INFER_PATH="${PARENT_PARENT_DIR}/executor/offline/infer.py"
    if command -v pgrep >/dev/null 2>&1; then
        if pgrep -f "python.*${INFER_PATH}" > /dev/null 2>&1; then
            echo "检测到正在执行 ${INFER_PATH} 的 Python 进程，脚本退出。"
            exit 1
        fi
    elif ps aux | grep -F "${INFER_PATH}" | grep -v grep > /dev/null 2>&1; then
        echo "检测到正在执行 ${INFER_PATH} 的 Python 进程，脚本退出。"
        exit 1
    fi
}

function get_rank()
{
    filename=$(basename "$YAML")
    world_size=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$YAML')); print(cfg.get('parallel_config', {}).get('world_size', cfg.get('world_size')))")
    platform_version=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML'))['model_config'].get('platform_version'))")
    if [ "$platform_version" = "950" ]; then
        chip_num=8
    else
        chip_num=16
    fi

    if [ -n "$world_size" ]; then
        export WORLD_SIZE=$world_size
        echo "world_size = $WORLD_SIZE"
        export SERVER_NUM=1
        LOCAL_HOST=$(hostname -I | awk -F " " '{print$1}')
        export IPs=($LOCAL_HOST)
    else
        echo "无法在 '$filename' 中找到 world_size"
        exit 1
    fi
}

function check_env_vars()
{
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    LOCAL_HOST=$(hostname -I | awk -F " " '{print$1}')
    export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp}
    export MASTER_PORT=${MASTER_PORT:-6038}
    export MASTER_ADDR=$LOCAL_HOST
    export RANK_OFFSET=0
    export MA_NUM_GPUS=$WORLD_SIZE
    export LOCAL_WORLD_SIZE=$WORLD_SIZE

    DATE=$(date +%Y%m%d)
    DIR_PREFIX="res"
    MODEL_NAME=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$YAML')); print(cfg.get('model_config', {}).get('model_name', cfg.get('model_name')))")
    PREFIX=$(basename "$YAML")
    PREFIX="${PREFIX%.*}"
    NAME=${MODEL_NAME}_${PREFIX}
    export CASE_NAME=$NAME

    SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
    PARENT_PARENT_DIR=$(cd "$SCRIPT_PATH/../.." &>/dev/null && pwd)
    export RES_PATH="${DIR_PREFIX}/${DATE}/${NAME}"
    export WORK_DIR="${PARENT_PARENT_DIR}/models/${MODEL_DIR}"
    DUMP_PRECISION_PATH=${WORK_DIR}'/'${RES_PATH}'/dump_data'
    mkdir -p ${WORK_DIR}'/'${RES_PATH}
    mkdir -p ${DUMP_PRECISION_PATH}
}

function set_hccl()
{
    export HCCL_IF_IP=$LOCAL_HOST
    export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-23456}

    micro_batch_mode=$(python3 -c "import yaml; \
        print(yaml.safe_load(open('$YAML')).get('model_config').get('micro_batch_mode', 0))")

    if [ "$platform_version" != "950" ]; then
        export HCCL_OP_EXPANSION_MODE=AIV
    fi

    if [[ ${micro_batch_mode} -eq 1 ]]; then
        unset HCCL_OP_EXPANSION_MODE
    fi

    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_EXEC_TIMEOUT=1200
}

function launch_infer_task()
{
    INFER_PATH=${PARENT_PARENT_DIR}/models/${MODEL_DIR}/infer.py
    if [ ! -f "${INFER_PATH}" ]; then
        INFER_PATH="${PARENT_PARENT_DIR}/executor/offline/infer.py"
    fi

    cores=$(cat /proc/cpuinfo | grep "processor" | wc -l)
    avg_core_per_rank=$(expr $cores \/ $MA_NUM_GPUS)
    if [ "$avg_core_per_rank" -lt 1 ]; then
        avg_core_per_rank=1
    fi
    core_gap=$(expr $avg_core_per_rank \- 1)

    for((i=0; i<${MA_NUM_GPUS}; i++))
    do
        start=$(expr $i \* $avg_core_per_rank)
        end=$(expr $start \+ $core_gap)
        cmdopt=$start"-"$end
        export LOCAL_RANK=$i
        export RANK_ID=$(expr $i + $RANK_OFFSET)
        cmd=(taskset -c "$cmdopt" python3 "${INFER_PATH}" --yaml_file_path="${YAML}")
        if [ $i -eq 0 ] && [[ ${LAUNCH_MODE:-0} -ne 1 ]]; then
            "${cmd[@]}" 2>&1 | tee ${WORK_DIR}/${RES_PATH}/log_${LOCAL_RANK}.log &
        else
            "${cmd[@]}" &> ${WORK_DIR}/${RES_PATH}/log_${LOCAL_RANK}.log &
        fi
    done
    wait
}

function check_result()
{
    file=${WORK_DIR}/${RES_PATH}/log_0.log
    echo "检查日志" $file

    if [ ! -f "$file" ]; then
        echo "ERROR: 日志 "$file" 不存在。"
        exit 1
    fi
    error_str=$(grep "ERROR" $file 2>/dev/null)
    if [ -n "$error_str" ]; then
        echo "CASE" ${CASE_NAME} "发现 ERROR，请检查。"
    fi
}
