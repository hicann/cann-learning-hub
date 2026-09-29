"""Negative input-domain checks; no invalid tensor is launched into a kernel."""
from checks import *
for name,x in [
 ('NaN',torch.full((8,64),float('nan'),device='npu')),
 ('Inf',torch.full((8,64),float('inf'),device='npu')),
 ('magnitude',torch.full((8,64),2049.,device='npu')),
 ('dtype',torch.zeros((8,64),device='npu',dtype=torch.float16)),
 ('stride',torch.zeros((64,8),device='npu').t()),
 ('shape',torch.zeros((8,65),device='npu'))]:
    try: validate_input(x,(8,64))
    except ValueError as e: print(name,'rejected:',e)
    else: raise AssertionError('Unsupported input accepted: '+name)
try:validate_input(torch.full((64,),11.,device='npu'),(64,),10)
except ValueError as e:print('Exp magnitude rejected:',e)
else:raise AssertionError('Unsupported Exp input accepted')
print('Verification passed!')
