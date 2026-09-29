"""Replace vadd with vmul; keep register load/store and GM/UB transfers."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from vector_kernels import binary_instructions
from course_validation import cpu_input,checked_input
import torch
kernel=binary_instructions("mul",64)
for case in ("random","zero","constant","ramp"):
    a=cpu_input((64,),case);b=cpu_input((64,),"ramp")
    y=kernel(checked_input(a),checked_input(b));torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(),a*b,rtol=0,atol=0)
    print(f"Mul {case}: full-output PASS",flush=True)
