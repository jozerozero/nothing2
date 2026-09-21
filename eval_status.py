"""Read-only progress and strict completion summary for the frozen FT50 panel."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import subprocess
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    man=json.loads((a.root/'manifest.json').read_text());now=time.time()
    completed=Counter();suites=defaultdict(Counter);gpus=set();forward_verified=0;invalid=[];elapsed=[]
    for f in (a.root/'results').glob('step-*/row-*.json'):
        r=json.loads(f.read_text())
        try:
            assert r['complete'] and r['manifest_id']==man['manifest_id']
            assert all(math.isfinite(r['metrics'][k]) for k in ('rmse','r2','mae'))
            assert r['actual_forward_block_calls'] and all(c==[3]*12 for c in r['actual_forward_block_calls'])
            assert r['gpu']['runtime_visible_count']==1
            assert r['data_audit']['test_rows_filtered']==0 and not r['data_audit']['query_chunking']
            completed[r['checkpoint_step']]+=1;suites[r['checkpoint_step']][r['suite']]+=1
            gpus.add((r['node'],r['gpu']['uuid']));forward_verified+=1;elapsed.append(r['elapsed_seconds'])
        except (KeyError,AssertionError,TypeError):invalid.append(str(f))
    workers=[]
    for f in (a.root/'workers').glob('*.json'):
        r=json.loads(f.read_text());r['heartbeat_age_s']=now-r.get('heartbeat_epoch',r.get('started_epoch',r.get('ended_epoch',now)))
        workers.append(r)
    errors=[]
    for f in (a.root/'errors').glob('step-*/row-*.json'):
        r=json.loads(f.read_text())
        errors.append({k:r.get(k) for k in ('checkpoint_step','dataset_index','reason','returncode','error_tail')})
    claims=[]
    for f in (a.root/'claims').glob('step-*/row-*.json'):
        r=json.loads(f.read_text());claims.append({**r,'age_s':now-r['started_epoch']})
    receipts=[json.loads(f.read_text()) for f in (a.root/'launches').glob('*.json')]
    parents=','.join(r['parent'] for r in receipts)
    queue=subprocess.check_output(['squeue','--steps','-j',parents,'-h','-o','%i|%j|%M|%N'],text=True) if parents else ''
    out={'epoch':now,'manifest_id':man['manifest_id'],'target_finetuned_units':11200,
         'complete_finetuned_units':sum(v for k,v in completed.items() if k!=22175),
         'source_baseline_complete':completed[22175],
         'complete_checkpoint_count':sum(completed[c['step']]==224 for c in man['checkpoints'] if c['step']!=22175),
         'per_checkpoint':[{'step':c['step'],'complete':completed[c['step']],'target':224,'suite_counts':suites[c['step']]} for c in man['checkpoints']],
         'physical_GPUs_with_valid_result':len(gpus),'invalid_results':invalid,
         'workers':workers,'active_claims':claims,'errors':errors,'slurm_steps':queue,'launches':receipts}
    print(json.dumps(out,indent=2,allow_nan=False))
if __name__=='__main__':main()
