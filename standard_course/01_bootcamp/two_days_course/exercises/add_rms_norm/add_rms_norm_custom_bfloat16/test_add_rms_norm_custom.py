import os
import sys

import torch
import torch_npu

def main():
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(lib_dir, "build")
    sys.path.append(build_dir)

    # 加载 libascendc_ops.so，触发 TORCH_LIBRARY 注册。
    torch.ops.load_library(os.path.join(build_dir, "libascendc_ops.so"))

    torch.manual_seed(2026)
    shapes = [
        (1, 1),          # minimum shape
        (31, 256),       # fewer blocks than cores
        (48, 256),       # full tiles and idle tail cores
        (49, 512),       # row-tile tail on an aligned shape
        (65, 257),       # non-block-aligned N falls back to one row per tile
        (3, 2048),       # two rows per tile, per-row scalar broadcast fallback
        (1, 4096),       # maximum supported row length
        (1, 1, 2048),    # 3D, single row
        (1, 16, 2048),   # 3D, 16 rows
        (1, 16, 32, 128) # 4D, 512 rows
    ]

    for shape in shapes:
        n = shape[-1]
        x1 = torch.randn(*shape, dtype=torch.bfloat16).npu()
        x2 = torch.randn(*shape, dtype=torch.bfloat16).npu()
        gamma = torch.randn(n, dtype=torch.bfloat16).npu()

        y, rstd, x = torch.ops.ascendc_ops.ascendc_add_rms_norm_custom(x1, x2, gamma, 1e-6)

        x1_cpu = x1.cpu()  # bfloat16
        x2_cpu = x2.cpu()  # bfloat16
        gamma_cpu = gamma.cpu()  # bfloat16
        n = shape[-1]

        summed_bf16 = x1_cpu + x2_cpu  # bfloat16
        summed_fp32 = summed_bf16.float()  # float32
        mean = summed_fp32.pow(2).sum(dim=-1, keepdim=True) * (1.0 / n)
        rstd_ref = 1.0 / torch.sqrt(mean + 1e-6)  # 1/RMS
        normalized = summed_fp32 * rstd_ref  # float32
        normalized_bf16 = normalized.bfloat16()
        y_ref = gamma_cpu * normalized_bf16  # bfloat16
        x_ref = summed_bf16  # bfloat16
        rstd_ref = rstd_ref.squeeze(-1)  # float32 1/RMS，shape[:-1]

        assert tuple(y.shape) == shape, f"y shape {tuple(y.shape)} != {shape}"
        assert tuple(rstd.shape) == shape[:-1], f"rstd shape {tuple(rstd.shape)} != {shape[:-1]}"
        assert tuple(x.shape) == shape, f"x shape {tuple(x.shape)} != {shape}"
        assert y.dtype == torch.bfloat16 and rstd.dtype == torch.float32 and x.dtype == torch.bfloat16
        assert torch.allclose(x.cpu().float(), x_ref.float(), atol=1e-2, rtol=1e-2), f"x mismatch: {shape}"
        assert torch.allclose(rstd.cpu().float(), rstd_ref, atol=1e-4, rtol=1e-4), f"rstd mismatch: {shape}"
        assert torch.allclose(y.cpu().float(), y_ref.float(), atol=2e-2, rtol=2e-2), f"y mismatch: {shape}"
        print(f"[OK] shape {shape} passed.")

    print("[OK] torch.ops.ascendc_ops.ascendc_add_rms_norm_custom passed.")

if __name__ == "__main__":
    main()
