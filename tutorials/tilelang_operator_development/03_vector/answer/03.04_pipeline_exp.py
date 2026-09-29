"""Exercise answer: Exp with one, two and three buffer versions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vector_kernels import unary_pipeline
from checks import input_tensor,check
import torch
# Initialize the NPU before CPU random-input generation.
torch.npu.init()
N,TILE,BLOCKS=1179648,1024,8
for stages in (1,2,3):
    kernel=unary_pipeline(N,TILE,BLOCKS,'exp',stages)
    for case in ('random','zero','constant','ramp'):
        print(f'Checking Exp stages={stages} {case}',flush=True)
        x=input_tensor((N,),case)
        host=x.cpu()
        assert x.dtype==torch.float32 and x.is_contiguous() and tuple(x.shape)==(N,)
        assert torch.isfinite(host).all().item() and host.abs().max().item()<=10
        check(kernel(x).cpu(),x.cpu().exp(),f'Exp stages={stages} {case}')
