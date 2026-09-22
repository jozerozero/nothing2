"""One audited continuation batch: 9 R19 nodes plus one sequenced gap node.

No old ledger, dataset, fitted result, trial or claim is deleted or rewritten.
Held-first submit is never automatically retried after an uncertain response.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

import table6_restart_ops as old
from table6_restart_deadline import parse_fields

ROOT, REPO = old.ROOT, old.REPO
STAGE = ROOT/'stage/table6_continuation_20260922_1500_v1'
LEDGER = STAGE/'submission.json'
PREVIOUS = [str(i) for i in range(208377,208387)]
GAP_JOB = '208823'

def run(args, timeout=90):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((args,p.returncode,p.stdout[-2000:],p.stderr[-2000:]))
    return p.stdout.strip()

def require(ok, message):
    if not ok: raise RuntimeError(message)

def modules():
    import table6_r19_node8_v2 as r19
    import table6_gap190_node8 as gap
    return r19, gap

def guard():
    r19,gap=modules()
    q=run(['squeue','--me','-h','-o','%i|%j|%T|%Z'])
    for line in q.splitlines():
        fields=line.split('|');require(len(fields)==4,'malformed queue')
        job,name,state,work=fields
        require(not name.startswith(('t6r22g','t6r22u','t6g22r')),'active old/new restart job: '+line)
        require(work not in (str(old.STAGE),str(r19.STAGE)),'active R19 stage: '+line)
        if work==str(gap.STAGE): require(job==GAP_JOB,'unexpected active missing190: '+line)
    a=run(['sacct','-n','-P','-X','-j',','.join(PREVIOUS),'-o','JobIDRaw,State%30,ExitCode'])
    found={}
    for line in a.splitlines():
        row=line.split('|')
        if row[0] in PREVIOUS:
            require(row[0] not in found,'duplicate accounting')
            require(row[1].split()[0].rstrip('+') in ('FAILED','COMPLETED','TIMEOUT','CANCELLED','NODE_FAIL','OUT_OF_MEMORY'), 'old job not terminal')
            found[row[0]]=row[1:3]
    require(set(found)==set(PREVIOUS),'old exact job accounting missing')
    gap.verify()
    return {'previous_accounting':found,'queue':q,'epoch':time.time()}

def validate_fields(d,item,held=True):
    r19,gap=modules();m=r19 if item['kind']=='r19' else gap
    cpus=64 if item['kind']=='r19' else 128
    expected={'JobId':item['job_id'],'JobName':item['name'],'Partition':'faculty','Account':'faculty-acc',
      'QOS':'bgqos','NumTasks':'8','NumCPUs':str(cpus),'CPUs/Task':str(cpus//8),'MinMemoryNode':'512G',
      'TimeLimit':'02:00:00','Nice':'0','Requeue':'0','Command':str(m.STAGE/'run.sh'),
      'WorkDir':str(m.STAGE),'StdOut':str(m.LOGS/f"slurm-{item['job_id']}.out"),
      'StdErr':str(m.LOGS/f"slurm-{item['job_id']}.err")}
    require(all(d.get(k)==v for k,v in expected.items()), 'Slurm contract drift: '+str({k:(d.get(k),v) for k,v in expected.items() if d.get(k)!=v}))
    require(d.get('NumNodes') in ('1','1-1'),'wrong node count')
    require({'gres/gpu=8','cpu='+str(cpus)}<=set(d.get('ReqTRES','').split(',')),'wrong GPU/CPU TRES')
    require(d.get('NtasksPerN:B:S:C','').split(':')[0]=='8','wrong ranks/node')
    dep=d.get('Dependency')
    if item['kind']=='r19': require(dep=='(null)','unexpected R19 dependency')
    else: require(dep=='(null)' or re.fullmatch('afterany:'+GAP_JOB+r'\((?:unfulfilled|fulfilled)\)',dep or ''),'wrong gap dependency')
    if held: require(d.get('JobState')=='PENDING' and d.get('Reason')=='JobHeldUser','not held')
    return m

def control(item):
    d=parse_fields(run(['scontrol','show','job','-o',item['job_id']]))
    m=validate_fields(d,item)
    expected=getattr(m,'EXCLUDE',old.EXCLUDE)
    require(set(run(['scontrol','show','hostnames',d['ExcNodeList']]).splitlines())==set(run(['scontrol','show','hostnames',expected]).splitlines()),'excluded-node drift')
    spool=STAGE/('spool-'+item['job_id']+'.sh')
    require(not spool.is_symlink(),'spool symlink')
    if not spool.exists():run(['scontrol','write','batch_script',item['job_id'],str(spool)])
    require(spool.read_text()==m.script(),'spooled source drift')
    return d

def main(mode):
    r19,gap=modules()
    if mode=='prepare':
        require(not LEDGER.exists(),'existing/uncertain submission: no retry')
        audit=guard()
        if not r19.STAGE.exists(): r19.prepare()
        r19.verify();gap.verify()
        STAGE.mkdir(parents=True,exist_ok=True)
        old.atomic(STAGE/'preparation.json',audit)
        return audit
    if mode=='submit':
        require(not LEDGER.exists(),'existing/uncertain submission: no retry')
        audit=guard();r19.verify();gap.verify()
        work=old.workload();require(work['eligible']>0,'no R19 eligible unfinished work')
        STAGE.mkdir(parents=True,exist_ok=True)
        state={'state':'submitting','created_epoch':time.time(),'jobs':[],'workload':work,
               'audit':audit,'repo_commit':run(['git','-C',str(REPO),'rev-parse','HEAD']),
               'scope':'9 full8GPU R19 allocations plus 1 missing190 after208823; 2h each; frozen studies/seeds'}
        old.atomic(LEDGER,state)
        roster=[('r19',f't6r22u{i:02d}',r19) for i in range(1,10)]+[('gap','t6g22r2',gap)]
        for kind,name,module in roster:
            state['submission_intent']={'kind':kind,'name':name,'epoch':time.time()};old.atomic(LEDGER,state)
            argv=['sbatch','--hold','--parsable','--job-name='+name]
            if kind=='gap':argv+=['--dependency=afterany:'+GAP_JOB]
            raw=run(argv+[str(module.STAGE/'run.sh')]);job=raw.split(';')[0]
            require(job.isdigit(),'uncertain sbatch receipt')
            item={'kind':kind,'name':name,'job_id':job,'state':'held'}
            state['jobs'].append(item);old.atomic(LEDGER,state)
            control(item)
        state['state']='verified_held';old.atomic(LEDGER,state);return state
    if mode=='release':
        state=json.loads(LEDGER.read_text());require(state['state']=='verified_held','not verified held')
        r19.verify();gap.verify()
        for item in state['jobs']:control(item)
        state['state']='releasing';old.atomic(LEDGER,state)
        for item in state['jobs']:
            run(['scontrol','release',item['job_id']]);item['state']='released';old.atomic(LEDGER,state)
        state['state']='released';old.atomic(LEDGER,state);return state
    if mode=='status':
        state=json.loads(LEDGER.read_text());ids=','.join(i['job_id'] for i in state['jobs'])
        return {'ledger':state,'queue':run(['squeue','-j',ids,'-h','-o','%i|%j|%T|%q|%D|%C|%M|%l|%R'])}
    raise ValueError(mode)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','submit','release','status'))
    print(json.dumps(main(p.parse_args().mode),allow_nan=False))
