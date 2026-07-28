import os
from pathlib import Path
from typing import Any

import torch
import torch_npu
import torchair
from torchair.ge import Tensor


LIB_PATH = Path(os.environ.get("ASCENDC_LAUNCH_LIB", Path(__file__).with_name("build") / "libge_torchair_launch_ops.so"))
torch.ops.load_library(str(LIB_PATH))


@torchair.register_fx_node_ge_converter(torch.ops.ge_launch_samples.launch_add.default)
def convert_launch_add(x: Tensor, y: Tensor, z: Tensor = None, meta_outputs: Any = None):
    return torchair.ge.custom_op(
        "LaunchAddCustom",
        inputs={"x": x, "y": y},
        outputs=["z"],
    )


class LaunchAddModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.ge_launch_samples.launch_add(x, y)


def main():
    shape = (8, 2048)
    torch.manual_seed(0)
    x_cpu = torch.rand(shape, device="cpu", dtype=torch.float32)
    y_cpu = torch.rand(shape, device="cpu", dtype=torch.float32)

    config = torchair.CompilerConfig()
    backend = torchair.get_npu_backend(compiler_config=config)
    opt_model = torch.compile(LaunchAddModel().npu(), fullgraph=True, backend=backend, dynamic=False)

    output = opt_model(x_cpu.npu(), y_cpu.npu()).cpu()
    golden = torch.add(x_cpu, y_cpu)
    torch.testing.assert_close(output, golden, rtol=1e-4, atol=1e-4)
    print("TorchAir LaunchAddCustom sample success.")


if __name__ == "__main__":
    main()
