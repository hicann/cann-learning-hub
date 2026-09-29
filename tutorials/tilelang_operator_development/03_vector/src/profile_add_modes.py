"""One config per msprof process: 1 check + 3*(10 warmup + 30 retained) launches."""
import sys, json
from pathlib import Path
import torch
from course_paths import OUTPUT_DIR
torch.npu.init()
from add_modes import add_simd, add_simt
from course_validation import cpu_input, checked_input
mode, workload = sys.argv[1:3]
N, tile, blocks = {"small": (64,64,1), "large": (1179648,16384,36)}[workload]
factory = {"simd": add_simd, "simt": add_simt}[mode]
kernel = factory(N,tile,blocks)
a = cpu_input((N,),"ramp"); b = cpu_input((N,),"constant")
x, z = checked_input(a), checked_input(b); reference = a+b
out = kernel(x,z); torch.npu.synchronize(); torch.testing.assert_close(out.cpu(),reference,rtol=0,atol=0)
artifacts=(OUTPUT_DIR / 'generated/msprof_simt_gm/generated'); artifacts.mkdir(parents=True,exist_ok=True)
(artifacts/f"{mode}_{workload}.cpp").write_text(kernel.get_kernel_source())
for batch in range(3):
    for i in range(40):
        torch.npu.synchronize(); out=kernel(x,z); torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(),reference,rtol=0,atol=0)
print(json.dumps(dict(mode=mode,N=N,tile=tile,blocks=blocks,correct=True,target_launches=121,retained=90)))
