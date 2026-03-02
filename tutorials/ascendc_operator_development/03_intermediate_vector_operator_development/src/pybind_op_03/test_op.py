import torch
import torch_npu
import custom_ops_lib
torch.npu.config.allow_internal_format = False

length = [8, 2048]
x = torch.rand(length, device='cpu', dtype=torch.float16)
y = torch.rand(length, device='cpu', dtype=torch.float16)
golden = x + y
output_npu = custom_ops_lib.npu_add_custom_template(x.npu(), y.npu())

print("is same:",torch.allclose(golden, output_npu.cpu(),rtol=0.001, atol=0.001))