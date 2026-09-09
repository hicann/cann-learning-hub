#!/usr/bin/env python3
"""Load the C++ fusion pass, compile an Add(x, 0) graph, and run it on NPU."""

import ctypes
import os
from pathlib import Path

import numpy as np
from ge.es.graph_builder import GraphBuilder
from ge.ge_global import GeApi
from ge.graph import Tensor
from ge.graph.types import DataType, Format
from ge.session import Session

DEVICE_ID = 0
GRAPH_ID = 1
SHAPE = [2, 3]

pass_library = Path(os.environ["GE_NOTEBOOK_PASS_SO"]).resolve()
if not pass_library.is_file():
    raise FileNotFoundError("Fusion Pass library not found: {}".format(pass_library))

# REG_FUSION_PASS 在 so 加载时执行静态注册。要在 ge_initialize 前加载。
loaded_pass = ctypes.CDLL(
    str(pass_library), mode=os.RTLD_NOW | os.RTLD_GLOBAL
)
print("[INFO] C++ Fusion Pass 已加载：{}".format(pass_library))

builder = GraphBuilder("CppPassAddZeroGraph")
x = builder.create_input(
    index=0, name="input_x", data_type=DataType.DT_FLOAT, shape=SHAPE
)
builder.set_graph_output(x + 0.0, 0)
graph = builder.build_and_reset()

input_array = np.arange(6, dtype=np.float32).reshape(SHAPE)
ge_api = GeApi()
ge_api.ge_initialize({
    "ge.exec.deviceId": str(DEVICE_ID),
    "ge.graphRunMode": "0",
})

session = None
input_tensor = None
outputs = None
try:
    session = Session()
    session.add_graph(GRAPH_ID, graph)
    input_tensor = Tensor(
        input_array.reshape(-1).tolist(),
        None,
        DataType.DT_FLOAT,
        Format.FORMAT_ND,
        SHAPE,
    )
    outputs = session.run_graph(GRAPH_ID, [input_tensor])
    actual = np.asarray(outputs[0].data, dtype=np.float32)
    np.testing.assert_allclose(actual, input_array, rtol=1e-6, atol=1e-6)
    print("[OK] 融合后的图已在 NPU 上执行并通过数值校验")
finally:
    outputs = None
    input_tensor = None
    # 释放 Session 引用，由 Session 析构统一释放图资源。
    session = None
    ge_api.ge_finalize()

# Keep the ctypes handle alive until GE has finalized.
assert loaded_pass is not None
