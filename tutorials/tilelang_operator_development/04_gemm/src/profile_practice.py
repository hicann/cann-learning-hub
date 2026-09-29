"""Validate and profile the five configurations across four practice stages."""
import sys,json
from run_gemm import init,validate,inputs,torch,tilelang,E
from gemm_practice36 import gemm
from gemm_reuse_persistent import gemm as persistent_gemm
CASES=[(64,128,1,False),(64,128,2,False),(128,128,2,False),(256,128,2,False),(256,128,2,True)]
def main(index,profile=True):
    init()
    BM,BN,stages,reuse=CASES[index]
    factory=persistent_gemm if reuse else gemm
    k=tilelang.compile(factory(512,36864,256,BM,BN,stages),out_idx=[2])
    name=f'practice_{index+1}'
    (E/(name+'.cpp')).write_text(k.get_kernel_source())
    validate(k,(512,36864,256),name)
    if profile:
        a,b,ref=inputs(512,36864,256,'random')
        for batch in range(3):
            for _ in range(40):
                torch.npu.synchronize();out=k(a,b);torch.npu.synchronize()
            torch.testing.assert_close(out.cpu(),ref,rtol=1e-4,atol=1e-4)
    print('Verification passed!',flush=True)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',type=int,choices=range(1,6))
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    main(args.step-1,profile=not args.validate_only)
