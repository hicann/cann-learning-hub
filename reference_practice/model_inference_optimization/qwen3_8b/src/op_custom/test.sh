SRC_DIR=$1
cd ${SRC_DIR}/qmm_custom
bash build.sh
chmod +x build_out/*.run
build_out/*.run
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
cd ${SRC_DIR}/torch_ops_extension
bash build_and_install.sh
cd ${SRC_DIR}
python3 test_case_profiler.py