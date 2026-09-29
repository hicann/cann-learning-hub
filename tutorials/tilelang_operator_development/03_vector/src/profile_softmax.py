"""Fair serial / multibuffer comparison: same shape, grid, tile and five outputs."""
import sys,json
from pathlib import Path
import torch
from softmax_kernel import softmax_kernel,softmax_pipeline
from course_validation import cpu_input,checked_input,check_softmax
from course_paths import OUTPUT_DIR
mode=sys.argv[1];M,C,rows,blocks=9216,128,8,36
if mode not in ("serial","2","3"):raise ValueError("mode must be serial, 2, or 3")
kernel=softmax_kernel(M,C,rows,blocks) if mode=="serial" else softmax_pipeline(M,C,rows,blocks,int(mode))
host=cpu_input((M,C),"random");x=checked_input(host)
outs=kernel(x);check_softmax(outs,host,"initial")
artifacts=(OUTPUT_DIR / 'generated/msprof_03_06/generated');artifacts.mkdir(parents=True,exist_ok=True)
(artifacts/f"kernel_{mode}.cpp").write_text(kernel.get_kernel_source())
for batch in range(3):
    for i in range(40):
        torch.npu.synchronize();outs=kernel(x);torch.npu.synchronize()
    check_softmax(outs,host,f"batch {batch}")
print(json.dumps(dict(mode=mode,M=M,C=C,rows=rows,blocks=blocks,outputs=5,correct=True,target_launches=121,retained=90)))
