#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 gen_data.py
g++ -I"." -I"$ASCEND_TOOLKIT_HOME/include" -I"$HOME/vendors/customize/op_api/include" \
    -L"$ASCEND_TOOLKIT_HOME/lib64" -L"$HOME/vendors/customize/op_api/lib" \
    aclnn_test.cpp -lcust_opapi -lnnopbase -lacl_rt -o execute_op
set +u
source "$HOME/vendors/customize/bin/set_env.bash"
set -u
./execute_op .
python3 verify_result.py
