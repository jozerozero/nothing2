"""Four real GPUs alternating two isolated TabSwift protocols, with shared read-only helpers."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import time
import tabfm_default_dispatch as q
from tabfm_local_tmp import activate

def load_plan(path):
    plan=q.read(path)
    q.require(plan['plan_id']==q.digest({k:v for k,v in plan.items() if k!='plan_id'}),'Plan digest mismatch')
    for record in plan['source_records']:q.verify_file(record)
    q.require(len(plan['campaign_manifests'])==2,'Exactly two isolated variants required')
    campaigns=[]
    for record in plan['campaign_manifests']:
        p=q.verify_file(record);man,tasks=q.load_campaign(p)
        q.require(man['worker_python']==plan['worker_python'],'Worker environment differs')
        campaigns.append((p,man,tasks))
    q.require([m['protocol']['variant'] for _,m,_ in campaigns]==['official16','budget32x8'], 'Protocol order mismatch')
    q.require(len({m['output_root'] for _,m,_ in campaigns})==2,'Protocol outputs collide')
    return plan,campaigns

NATIVE_VALID=q.valid_result
def validated_result(path,man,task):
    result=NATIVE_VALID(path,man,task)
    if man.get('protocol',{}).get('variant')=='budget32x8':
        q.require(result['actual_ensemble_count']==man['protocol']['n_estimators'][task['task_kind']],
                  'Strict actual ensemble budget failed')
    return result
q.valid_result=validated_result

def pending_order(campaigns):
    queues=[sorted(tasks,key=lambda t:(q.work_size(t),t['task_kind'],t['dataset_index'])) for _,_,tasks in campaigns]
    q.require(len(queues[0])==len(queues[1])==681,'Paired full-scope queues required')
    for index in range(681):
        for variant in range(2):
            yield variant,queues[variant][index]

def run(campaigns):
    owners=[q.binding(man) for _,man,_ in campaigns]
    for _,man,tasks in campaigns:q.check_smoke(man,tasks)
    counts=[Counter(),Counter()]
    for idx,task in pending_order(campaigns):
        path,man,_=campaigns[idx]
        output=q.task_path(man,'results',task)
        if output.exists():
            if q.read(output).get('complete') is True:q.valid_result(output,man,task)
            q.require(q.task_path(man,'claims',task).exists(),'Unclaimed existing output')
            continue
        if not q.claim(man,task,owners[idx]):continue
        ok=q.launch(man,path,task,owners[idx])
        counts[idx]['success' if ok else 'failed']+=1
    results=[]
    for idx,(_,man,_) in enumerate(campaigns):
        owner=owners[idx]
        rec=dict(owner,finished_epoch=time.time(),attempts=dict(counts[idx]),
                 reason='No unclaimed tasks; individual lane exit is not proof of campaign completion')
        q.atomic(man,Path(man['output_root'])/'worker_done'/owner['job']/f"rank-{owner['rank']}.json",rec)
        results.append(rec)
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['preflight','check','smoke','check-smoke','run','status'])
    p.add_argument('--plan',required=True,type=Path);args=p.parse_args()
    plan,campaigns=load_plan(args.plan)
    if args.mode in ('preflight','smoke','run'):
        env=q.normalize_visibility(os.environ);os.environ.clear();os.environ.update(env)
        first=campaigns[0][1]
        audit=Path(first['output_root'])/'runtime_tmp'/os.environ['SLURM_JOB_ID']/os.environ['SLURM_STEP_ID']/f"rank-{os.environ['SLURM_PROCID']}"
        runtime=activate(first,plan,audit)
        q.atomic(first,audit/'dual-protocol-runtime.json',{
            'plan_id':plan['plan_id'],'runtime_tmpdir':runtime['new_TMPDIR'],
            'campaign_manifest_ids':[m['manifest_id'] for _,m,_ in campaigns],
            'shared_runtime_only':True,
            'note':'Child environment receipts live here for both protocols; owner.manifest_id identifies each child. Results and claims remain separate.'})
    signal.signal(signal.SIGTERM,q.stop);signal.signal(signal.SIGINT,q.stop)
    if args.mode=='run':value=run(campaigns)
    else:
        value=[]
        for path,man,tasks in campaigns:
            if args.mode=='preflight':result=q.preflight(man)
            elif args.mode=='check':result=q.check_preflight(man)
            elif args.mode=='smoke':result=q.smoke(man,path,tasks)
            elif args.mode=='check-smoke':result=q.check_smoke(man,tasks,publish=True)
            else:result=q.status(man,tasks)
            value.append({'variant':man['protocol']['variant'],'result':result})
    print(json.dumps(value,sort_keys=True),flush=True)

if __name__=='__main__':main()
