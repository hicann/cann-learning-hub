"""
Performance benchmark script for as_strided operator.
Runs a representative workload for msprof op profiling.
Uses T9 (large shape, Path A) as the primary profiling workload.
"""
import os
import sys
import torch
import torch_npu

# Load custom op libraries
OP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
opapi_lib = os.path.join(OP_DIR, "code", "build", "op_host", "libcust_opapi.so")
tiling_lib = os.path.join(OP_DIR, "code", "build", "op_host", "libcust_opmaster_rt2.0.so")
if os.path.exists(opapi_lib):
    torch.classes.load_library(opapi_lib)
if os.path.exists(tiling_lib):
    torch.classes.load_library(tiling_lib)

def main():
    # Representative workload: T9 - large input, Path A (input fits in UB)
    # input_shape=[50000], size=[10000], stride=[3], storage_offset=0, dtype=float32
    input_tensor = torch.arange(50000, dtype=torch.float32).npu()
    
    # Warm-up runs (msprof --warm-up handles this, but extra safety)
    for _ in range(3):
        _ = torch.as_strided(input_tensor, [10000], [3], 0)
    
    # Measured runs
    num_iters = 20
    for i in range(num_iters):
        output = torch.as_strided(input_tensor, [10000], [3], 0)
    
    # Force sync to ensure all ops complete
    torch.npu.synchronize()
    
    print(f"Profiling workload completed: {num_iters} iterations")
    print(f"Input shape: [50000], Output shape: [10000], stride=[3], offset=0, dtype=float32")

if __name__ == "__main__":
    main()
