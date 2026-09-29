import sys,json
from pathlib import Path
from course_paths import OUTPUT_DIR
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vector_kernels import unary_serial,unary_pipeline
from checks import input_tensor
import torch
mode=sys.argv[1];tile=int(sys.argv[2]);N=1179648;blocks=8
kernel=unary_serial(N,tile,blocks,'relu') if mode=='serial' else unary_pipeline(N,tile,blocks,'relu',int(mode))
x=input_tensor((N,),'ramp');reference=x.cpu().relu()
y=kernel(x);torch.npu.synchronize();torch.testing.assert_close(y.cpu(),reference,rtol=0,atol=0)
artifacts=(OUTPUT_DIR / 'generated/msprof_consistent_cases/generated');artifacts.mkdir(parents=True,exist_ok=True)
(artifacts/f'kernel_{mode}_{tile}.cpp').write_text(kernel.get_kernel_source())
for batch in range(3):
 for i in range(40):
  torch.npu.synchronize();y=kernel(x);torch.npu.synchronize()
 torch.testing.assert_close(y.cpu(),reference,rtol=0,atol=0)
print(json.dumps(dict(mode=mode,N=N,tile=tile,blocks=blocks,batches=3,warmup_per_batch=10,samples_per_batch=30,correct=True)))
