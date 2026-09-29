"""Shared host checks and synchronized wall-clock benchmark protocol."""
import time,statistics
import torch,torch_npu
SEED=20260911
RTOL,ATOL=1e-5,1e-6

def input_tensor(shape,case):
    torch.manual_seed(SEED)
    if case=='random': x=torch.randn(shape,dtype=torch.float32)
    elif case=='zero': x=torch.zeros(shape,dtype=torch.float32)
    elif case=='constant': x=torch.full(shape,2.0,dtype=torch.float32)
    elif case=='large': x=torch.linspace(-1000,1000,torch.tensor(shape).prod().item()).reshape(shape)
    elif case=='ramp': x=torch.linspace(-4,4,torch.tensor(shape).prod().item()).reshape(shape)
    else: raise ValueError(case)
    return x.to('npu')

def validate_input(x,shape,max_abs=2048):
    if x.device.type!='npu' or x.dtype!=torch.float32 or tuple(x.shape)!=tuple(shape) or not x.is_contiguous():
        raise ValueError('Require contiguous fp32 NPU tensor with declared shape')
    if not torch.isfinite(x).all().item(): raise ValueError('Only finite inputs are supported')
    if x.abs().max().item()>max_abs: raise ValueError(f'Input magnitude must not exceed {max_abs}')

def check(y,reference,label,exact=False):
    torch.npu.synchronize()
    torch.testing.assert_close(y,reference,rtol=0 if exact else RTOL,atol=0 if exact else ATOL)
    err=(y-reference).abs().max().item()
    print(f'{label}: full-output PASS; shape={tuple(y.shape)}; max_abs_error={err:.8g}',flush=True)

def bench(fn,warmup=10,repeats=30):
    for _ in range(warmup): fn()
    torch.npu.synchronize()
    samples=[]
    for _ in range(repeats):
        torch.npu.synchronize()
        start=time.perf_counter_ns()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns()-start)/1000)
    return {'backend':'host perf_counter_ns + npu synchronize','unit':'us','warmup':warmup,'repeat':repeats,'statistic':'median','median_us':statistics.median(samples),'min_us':min(samples),'max_us':max(samples),'samples_us':samples,'cache':'reuse same input; no cache flush','includes':'Python dispatch, output allocation, kernel, synchronization; excludes compilation'}
