from pathlib import Path
import sys,json
import torch,torch_npu,tilelang
from gemm_kernel import gemm
from gemm_k_demo import k_demo_gemm
from course_paths import OUTPUT_DIR
SEED=20260911
RTOL=ATOL=1e-4
E=(OUTPUT_DIR / 'generated')
E.mkdir(parents=True,exist_ok=True)

def init():
    torch.npu.init()
    torch.set_num_threads(8)

def inputs(M,N,K,case):
    g=torch.Generator().manual_seed(SEED)
    if case=='random':
        a=(torch.randn(M,K,generator=g)*.25);b=(torch.randn(N,K,generator=g)*.25)
    elif case=='zero':a=torch.zeros(M,K);b=torch.ones(N,K)
    elif case=='constant':a=torch.full((M,K),.5);b=torch.full((N,K),-.25)
    else:
        a=((torch.arange(M*K).reshape(M,K)%9)-4).float()/4
        b=((torch.arange(N*K).reshape(N,K)%7)-3).float()/4
    a=a.to(torch.bfloat16);b=b.to(torch.bfloat16)
    ref=a.float()@b.float().T
    return a.npu(),b.npu(),ref

def check_inputs(a,b,M,N,K):
    for x,shape in ((a,(M,K)),(b,(N,K))):
        if x.device.type!='npu' or x.dtype!=torch.bfloat16 or tuple(x.shape)!=shape or not x.is_contiguous():raise ValueError('Expected contiguous NPU bf16 A(M,K), B(N,K)')
        if not torch.isfinite(x).all().item():raise ValueError('Teaching inputs must be finite')

def compile_kernel(shape,tile=(128,128),stages=1,persistent=False):
    M,N,K=shape;BM,BN=tile
    k=tilelang.compile(gemm(M,N,K,BM,BN,stages,persistent),out_idx=[2])
    name=f'{M}_{N}_{K}_{BM}_{BN}_s{stages}_p{int(persistent)}'
    (E/(name+'.cpp')).write_text(k.get_kernel_source())
    return k,name

def validate(k,shape,label):
    for case in ('random','zero','constant','pattern'):
        a,b,ref=inputs(*shape,case);check_inputs(a,b,*shape)
        for repeat in range(2):
            y=k(a,b);torch.npu.synchronize();out=y.cpu()
            torch.testing.assert_close(out,ref,rtol=RTOL,atol=ATOL)
        print(f'{label} {case}: full-output PASS; shape={tuple(out.shape)}; max_abs_error={(out-ref).abs().max().item():.8g}',flush=True)

BASE_SHAPE=(512,36864,256)

def experiment(section):
    if section=='practice':
        import subprocess
        subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'answer/04.05_practice.py')],check=True)
        return
    init()
    if section=='k_demo':
        for shape in [(512,36864,2048)]:
            k=tilelang.compile(k_demo_gemm(*shape),out_idx=[2]);validate(k,shape,'K split '+str(shape))
    else:
        configs={
            'basic':[(BASE_SHAPE,(128,128),1,False)],
            'pipeline':[(BASE_SHAPE,(128,128),s,False) for s in (1,2,3)],
            'persistent':[(BASE_SHAPE,(128,128),2,True)],
        }[section]
        for shape,tile,stages,p in configs:
            k,name=compile_kernel(shape,tile,stages,p);validate(k,shape,name)
    print('Verification passed!',flush=True)

if __name__=='__main__':experiment(sys.argv[1])
