"""ReLU pipeline correctness; kernel performance is collected by msprof separately."""
from vector_kernels import unary_serial,unary_pipeline,check_config
from checks import input_tensor,check
import torch
# Initialize the NPU before CPU random-input generation.
torch.npu.init()
N,TILE,BLOCKS=1179648,1024,8

def run():
    for mode in ('serial',1,2,3):
        kernel=unary_serial(N,TILE,BLOCKS,'relu') if mode=='serial' else unary_pipeline(N,TILE,BLOCKS,'relu',mode)
        for case in ('random','zero','constant','ramp'):
            print(f'Checking ReLU {mode=} {case}',flush=True)
            x=input_tensor((N,),case)
            host=x.cpu()
            assert x.dtype==torch.float32 and x.is_contiguous() and tuple(x.shape)==(N,)
            assert torch.isfinite(host).all().item() and host.abs().max().item()<=2048
            check(kernel(x).cpu(),x.cpu().relu(),f'ReLU {mode=} {case}',exact=True)
    try:
        check_config(1179648,16384,8,2,2)
    except ValueError as error:
        print(f'Expected UB budget rejection: {error}')
    else:
        raise AssertionError('256 KiB configuration must be rejected')
if __name__=='__main__':run()
