#!/bin/bash
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
python3 setup.py build bdist_wheel
pip3 install dist/custom_ops*.whl --force-reinstall

python3 test_op.py


