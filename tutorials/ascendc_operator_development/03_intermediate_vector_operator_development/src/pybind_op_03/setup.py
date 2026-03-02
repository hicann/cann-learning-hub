import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
exts = []
ext1 = NpuExtension(
    name="custom_ops_lib",
    sources=["./custom_op.cpp"],
    extra_compile_args = [
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
    ],
)
exts.append(ext1)

setup(
    name="custom_ops",
    ext_modules=exts,
    version='1.0',
    cmdclass={"build_ext": BuildExtension},
)
