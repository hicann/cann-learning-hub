"""Complete Exp exercise for N=1179648, tile=1024, blocks=8."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vector_kernels import unary_serial
from checks import input_tensor,validate_input,check
if __name__=='__main__':
    import torch
    torch.npu.init()
    N,tile,blocks=1179648,1024,8
    kernel=unary_serial(N,tile,blocks,'exp')
    for case in ('random','zero','constant','ramp'):
        x=input_tensor((N,),case);validate_input(x,(N,),10)
        check(kernel(x),x.exp(),f'Exp exercise N={N} tile={tile} blocks={blocks} {case}')
    print('Verification passed!',flush=True)
