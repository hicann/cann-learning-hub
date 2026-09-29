"""Complete NPU result checks; deliberate capacity failure runs separately."""
import argparse,json
from pathlib import Path
from relu_basic import make_relu,make_exp
from checks import input_tensor,validate_input,check,SEED,RTOL,ATOL
from course_paths import OUTPUT_DIR

def run(op='relu'):
    factory=make_relu if op=='relu' else make_exp
    rows=[]
    for N in (64,1024,8192):
        kernel=factory(N)
        for case in ('random','zero','constant','ramp'):
            x=input_tensor((N,),case)
            validate_input(x,(N,),10 if op=='exp' else 2048)
            reference=x.relu() if op=='relu' else x.exp()
            y=kernel(x)
            check(y,reference,f'{op} N={N} {case}',exact=op=='relu')
            rows.append({'op':op,'N':N,'case':case,'max_abs_error':(y-reference).abs().max().item(),'status':'PASS'})
    evidence=(OUTPUT_DIR / 'validation');evidence.mkdir(parents=True,exist_ok=True)
    (evidence/f'03.02_{op}_validation.json').write_text(json.dumps({'seed':SEED,'rtol':0 if op=='relu' else RTOL,'atol':0 if op=='relu' else ATOL,'results':rows},indent=2)+'\n')
    print('Verification passed!',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('op',choices=['relu','exp','capacity']);op=parser.parse_args().op
    if op=='capacity':make_relu(65536)
    else:run(op)
