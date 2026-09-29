import argparse,json
from pathlib import Path
from vector_kernels import *
from checks import *
from course_paths import OUTPUT_DIR

def run(section):
    results=[]
    if section=='single':
        for op in ('relu','exp'):
            for N in (64,1024,8192):
                k=unary_serial(N,N,1,op)
                for case in ('random','zero','constant','ramp'):
                    x=input_tensor((N,),case);validate_input(x,(N,),10 if op=="exp" else 2048)
                    ref=x.relu() if op=='relu' else x.exp()
                    check(k(x),ref,f'{op} N={N} {case}',exact=op=='relu')
    elif section=='tiling':
        for N,tile,blocks in ((65536,8192,8),(524288,8192,8),(1179648,1024,1),(1179648,1024,8)):
            k=unary_serial(N,tile,blocks,'relu')
            for case in ('random','zero','constant','ramp'):
                x=input_tensor((N,),case);validate_input(x,(N,),2048)
                check(k(x),x.relu(),f'ReLU N={N} tile={tile} blocks={blocks} {case}',True)
        for bad in [(1000,64,1),(8192,128,3),(0,64,1)]:
            try: check_config(*bad)
            except ValueError as e: print('Current no-tail contract rejection:',bad,str(e))
            else: raise AssertionError('Invalid shape accepted')
    elif section=='modes':
        N=64;a=input_tensor((N,),'ramp');b=input_tensor((N,),'constant')
        kernels=[('SimdVF register instructions',add_serial(N,N,1)),('SimtVF Parallel threads=128',add_simt(N)),('SimdVF instructions',binary_instructions('add'))]
        for label,k in kernels:
            for case in ('random','zero','constant','ramp'):
                a=input_tensor((N,),case);b=input_tensor((N,),'constant')
                check(k(a,b),a+b,label+' '+case,True)
            print(label,'input A=',a.cpu().tolist(),'input B=',b.cpu().tolist(),'output=',k(a,b).cpu().tolist())
        k=binary_instructions('mul')
        for case in ('random','zero','constant','ramp'):
            a=input_tensor((N,),case);check(k(a,b),a*b,'Mul instructions '+case,True)
    elif section=='pipeline':
        from run_pipeline import run as run_relu_pipeline
        run_relu_pipeline()
    else: raise ValueError(section)
    if results:
        path=(OUTPUT_DIR / 'validation')/(section+'_measurements.json');path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results,indent=2))
        for r in results: print('Measurement:',{k:v for k,v in r.items() if k!='samples_us'})
    print('Verification passed!',flush=True)

if __name__=='__main__':
    # Initialize NPU before CPU random input generation, as in run_pipeline.py.
    torch.npu.init()
    p=argparse.ArgumentParser();p.add_argument('section',choices=['single','tiling','modes','pipeline']);run(p.parse_args().section)
