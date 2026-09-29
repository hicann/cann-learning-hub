import sys,json
from pathlib import Path
from course_paths import OUTPUT_DIR
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vector_kernels import unary_serial
from checks import input_tensor,validate_input,check
import torch
blocks=int(sys.argv[1]);N,tile=1179648,1024
kernel=unary_serial(N,tile,blocks,'relu')
x=input_tensor((N,),'ramp');validate_input(x,(N,),2048)
reference=x.cpu().relu()
y=kernel(x);torch.npu.synchronize();torch.testing.assert_close(y.cpu(),reference,rtol=0,atol=0)
artifacts=(OUTPUT_DIR / 'generated/msprof_consistent_cases/generated');artifacts.mkdir(parents=True,exist_ok=True)
(artifacts/f'kernel_blocks{blocks}.cpp').write_text(kernel.get_kernel_source())
# One correctness call, then 3 separately warmed batches of 10+30 calls.
for batch in range(3):
    for i in range(40):
        torch.npu.synchronize()
        y=kernel(x)
        torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(),reference,rtol=0,atol=0)
print(json.dumps({'blocks':blocks,'N':N,'tile':tile,'seed':20260911,'input':'ramp [-4,4] fp32','batches':3,'warmup_per_batch':10,'samples_per_batch':30,'total_kernel_calls':121,'full_output_correctness':'PASS','cache':'same input reused, no flush'}),flush=True)
