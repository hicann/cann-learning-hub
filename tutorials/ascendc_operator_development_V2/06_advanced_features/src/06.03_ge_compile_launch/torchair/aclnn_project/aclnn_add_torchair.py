import os
from pathlib import Path
from typing import Any

import torch
import torch_npu
import torchair
from torchair.ge import Tensor


LIB_PATH = Path(os.environ.get("ASCENDC_ACLNN_LIB", Path(__file__).with_name("build") / "libge_torchair_aclnn_ops.so"))
torch.ops.load_library(str(LIB_PATH))


@torchair.register_fx_node_ge_converter(torch.ops.ge_launch_samples.aclnn_add.default)
def convert_aclnn_add(x: Tensor, y: Tensor, z: Tensor = None, meta_outputs: Any = None):
    return torchair.ge.custom_op(
        "AddCustom",
        inputs={"x": x, "y": y},
        outputs=["z"],
    )


class AclnnAddModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.ge_launch_samples.aclnn_add(x, y)


def has_aclnn_opp():
    for item in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if not item:
            continue
        proto = Path(item) / "op_proto/inc/op_proto.h"
        if proto.is_file() and "REG_OP(AddCustom)" in proto.read_text(errors="ignore"):
            return True
    return False


def main():
    if not has_aclnn_opp():
        print("跳过 aclnn_add AddCustom：ASCEND_CUSTOM_OPP_PATH 中未找到 AddCustom OPP。")
        return

    shape = (8, 2048)
    torch.manual_seed(0)
    x_cpu = torch.rand(shape, device="cpu", dtype=torch.float32)
    y_cpu = torch.rand(shape, device="cpu", dtype=torch.float32)

    config = torchair.CompilerConfig()
    backend = torchair.get_npu_backend(compiler_config=config)
    opt_model = torch.compile(AclnnAddModel().npu(), fullgraph=True, backend=backend, dynamic=False)

    output = opt_model(x_cpu.npu(), y_cpu.npu()).cpu()
    golden = torch.add(x_cpu, y_cpu)
    torch.testing.assert_close(output, golden, rtol=1e-4, atol=1e-4)
    print("TorchAir aclnn_add AddCustom sample success.")


if __name__ == "__main__":
    main()
