"""Summarize a fresh msprof capture from this chapter's profiling entry points."""
from pathlib import Path
import argparse,csv,json,statistics

def summarize(path,blocks):
    files=[path] if path.is_file() else sorted(path.rglob('op_summary*.csv'))
    if len(files)!=1:raise ValueError(f'Expected one op_summary CSV; found {len(files)}. Pass one capture directory or CSV.')
    with files[0].open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r.get('Op Name')=='main_kernel']
    rows.sort(key=lambda r:float(r['Task Start Time(us)']))
    if len(rows)!=128:raise ValueError(f'Expected 8 validation + 120 profiling calls, got {len(rows)}. Check the capture log for errors.')
    if any(r['Task Type']!='AI_CORE' or int(r['Block Num'])!=blocks for r in rows):raise ValueError('Kernel type or core count does not match')
    times=[float(r['Task Duration(us)']) for r in rows][8:]
    batches=[times[i*40+10:(i+1)*40] for i in range(3)]
    samples=sum(batches,[])
    result={'csv':str(files[0]),'blocks':blocks,'samples':90,'batch_median_us':[statistics.median(b) for b in batches],'median_us':statistics.median(samples)}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('path',type=Path)
    p.add_argument('--blocks',type=int,choices=[8,36],required=True)
    p.add_argument('--json',type=Path)
    p.add_argument('--log',type=Path,required=True,help='Combined stdout/stderr log from the same msprof capture')
    a=p.parse_args()
    log=a.log.read_text()
    if log.count('full-output PASS')!=4 or 'Traceback' in log or not ('Verification passed!' in log or '"correct": true' in log):
        raise ValueError('Capture log does not confirm successful validation and profiling')
    r=summarize(a.path,a.blocks)
    r['validation_log']=str(a.log)
    if a.json:a.json.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r,ensure_ascii=False,indent=2))
    print('Correctness log and timing samples verified.')
