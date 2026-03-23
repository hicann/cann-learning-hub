import torch
from torch._functorch.aot_autograd import aot_module_simplified
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig

class MM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y

def custom_backend(gm: torch.fx.GraphModule, example_inputs):
    compiler_config = CompilerConfig()
    compiler_config.mode = "reduce-overhead"
    compiler = torchair.get_compiler(compiler_config)
    return aot_module_simplified(gm, example_inputs, fw_compiler=compiler)

torch.npu.set_device(0)
x = torch.ones([2, 2], dtype=torch.int32).npu()
y = torch.ones([2, 2], dtype=torch.int32).npu()
model = torch.compile(MM().npu(), backend=custom_backend, dynamic=False)
ret = model(x, y)
print(ret)