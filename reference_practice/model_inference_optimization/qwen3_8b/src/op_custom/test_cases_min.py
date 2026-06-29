import os

import torch
import torch_npu
import custom_ops
#
### export ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/customize
### export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/customize/op_api/lib:${LD_LIBRARY_PATH}
### python test_cases_min.py
###
"""
export TUTORIAL_DIR=/mnt/workspace/gitCode/cann/cann-learning-hub/reference_practice/model_inference_optimization/qwen3_8b
export SRC_DIR=${TUTORIAL_DIR}/src/op_custom
cd ${SRC_DIR}
bash test.sh "${SRC_DIR}"



"""
#

torch.npu.config.allow_internal_format = True
torch.npu.config.allow_hf32 = False
torch.npu.set_compile_mode(jit_compile=False)

# M, K, N = 1, 4096, 4096
M, K, N = 1, 512, 512

def check_close(output, ref, dtype):
    if dtype == torch.int32:
        rtol, atol = 0, 0
        label = "INT32"
    else:
        rtol, atol = 0.01, 0.01
        label = "BF16"

    passed = torch.allclose(output.to(torch.float32), ref.to(torch.float32), rtol=rtol, atol=atol)
    print(f"{label} allclose: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("output[0, :5] =", output[0, :5].tolist())
        print("ref[0, :5]    =", ref[0, :5].tolist())
    return passed


def test_int32():
    print("[INT32] create cpu tensors", flush=True)
    x1_cpu = torch.randint(-128, 127, (M, K), dtype=torch.int8)
    x2_cpu = torch.randint(-128, 127, (K, N), dtype=torch.int8)
    scale_cpu = torch.randn([N], dtype=torch.float32)

    print("[INT32] copy tensors to NPU", flush=True)
    x1_npu = x1_cpu.npu()
    x2_npu_base = x2_cpu.npu()
    scale_npu = scale_cpu.npu()

    print("[INT32] format_cast x2 to FRACTAL_NZ", flush=True)
    x2_npu = torch_npu.npu_format_cast(x2_npu_base, 29)

    print("[INT32] launch torch.ops.custom.qmm_custom", flush=True)
    output_npu = torch.ops.custom.qmm_custom(
        x1_npu, x2_npu, scale_npu, pertoken_scale=None, dtype=0
    )
    print("[INT32] copy output to CPU / synchronize", flush=True)
    output = output_npu.cpu()

    print("[INT32] calculate CPU reference", flush=True)
    ref = torch.matmul(x1_cpu.to(torch.float64), x2_cpu.to(torch.float64)).round().to(torch.int32)

    assert output.shape == ref.shape, f"shape mismatch: {output.shape} vs {ref.shape}"
    assert output.dtype == torch.int32, f"dtype mismatch: {output.dtype}"
    return check_close(output, ref, torch.int32)


def test_bf16():
    print("[BF16] create cpu tensors", flush=True)
    x1_cpu = torch.randint(-128, 127, (M, K), dtype=torch.int8)
    x2_cpu = torch.randint(-128, 127, (K, N), dtype=torch.int8)
    scale_cpu = torch.randn([N], dtype=torch.float32)
    pertoken_scale_cpu = torch.randn([M], dtype=torch.float32)

    print("[BF16] copy tensors to NPU", flush=True)
    x1_npu = x1_cpu.npu()
    x2_npu_base = x2_cpu.npu()
    scale_npu = scale_cpu.npu()
    pertoken_scale_npu = pertoken_scale_cpu.npu()

    print("[BF16] format_cast x2 to FRACTAL_NZ", flush=True)
    x2_npu = torch_npu.npu_format_cast(x2_npu_base, 29)

    print("[BF16] launch torch.ops.custom.qmm_custom", flush=True)
    output_npu = torch.ops.custom.qmm_custom(
        x1_npu, x2_npu, scale_npu, pertoken_scale=pertoken_scale_npu, dtype=1
    )
    print("[BF16] copy output to CPU / synchronize", flush=True)
    output = output_npu.cpu()

    print("[BF16] calculate CPU reference", flush=True)
    int32_result = torch.matmul(x1_cpu.to(torch.float64), x2_cpu.to(torch.float64)).round().to(torch.int32)
    ref = (
        int32_result.to(torch.float32)
        * scale_cpu.unsqueeze(0)
        * pertoken_scale_cpu.unsqueeze(1)
    ).to(torch.bfloat16)

    assert output.shape == ref.shape, f"shape mismatch: {output.shape} vs {ref.shape}"
    assert output.dtype == torch.bfloat16, f"dtype mismatch: {output.dtype}"
    return check_close(output, ref, torch.bfloat16)


if __name__ == "__main__":
    print("ASCEND_HOME_PATH =", os.environ.get("ASCEND_HOME_PATH"))
    print("ASCEND_TOOLKIT_HOME =", os.environ.get("ASCEND_TOOLKIT_HOME"))
    print("ASCEND_OPP_PATH =", os.environ.get("ASCEND_OPP_PATH"))
    print("ASCEND_CUSTOM_OPP_PATH =", os.environ.get("ASCEND_CUSTOM_OPP_PATH"))
    print(f"Shape: M={M}, K={K}, N={N}")
    ok_int32 = test_int32()
    ok_bf16 = test_bf16()
    print(f"Summary: INT32={'PASS' if ok_int32 else 'FAIL'}, BF16={'PASS' if ok_bf16 else 'FAIL'}")
    raise SystemExit(0 if ok_int32 and ok_bf16 else 1)
