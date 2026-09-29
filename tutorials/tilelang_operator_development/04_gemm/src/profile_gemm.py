"""Full-output validation, then 3 batches of msprof kernel samples."""
import sys,json
from run_gemm import init,compile_kernel,validate,inputs,torch
init()
shape=tuple(map(int,sys.argv[1:4]));stages=int(sys.argv[4]);persistent=sys.argv[5]=='persistent'
k,name=compile_kernel(shape,stages=stages,persistent=persistent)
validate(k,shape,name)
a,b,ref=inputs(*shape,'random')
for batch in range(3):
    for _ in range(40):
        torch.npu.synchronize();out=k(a,b);torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(),ref,rtol=1e-4,atol=1e-4)
print(json.dumps(dict(shape=shape,stages=stages,persistent=persistent,blocks=8,validation_calls=8,batches=3,warmup_per_batch=10,samples_per_batch=30,correct=True)),flush=True)
