"""Start one bounded child step inside the user-authorized idle allocation."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/reg_loop3_step22175_finetune50_20260921_v1')
with (ROOT/'launch.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'launch_receipt.json').exists(), 'launch already recorded; refuse duplicate'
    assert not (ROOT/'run50').exists(), 'training output exists; refuse duplicate'
    inspection=json.loads((ROOT/'cpu_inspection'/'inspection.json').read_text())
    assert inspection['complete'] and inspection['strict_load'] and inspection['datasets']==19
    contract=json.loads((ROOT/'cpu_inspection'/'contract.json').read_text())
    assert contract['checkpoint_sha256']=='e1ef1e19f2cb1bede387341c301ae046a55604efaeb9eb773e493b01a933d0b4'
    info=subprocess.check_output(['scontrol','show','job','204827','-o'],text=True)
    fields=dict(re.findall(r'(\w+)=(\S+)',info))
    assert fields['JobState']=='RUNNING' and fields['NodeList']=='auh7-1b-gpu-257'
    assert fields['UserId'].startswith('guangyi.chen(') and fields['QOS']=='bgqos'
    assert int(fields['NumNodes'])==1 and int(fields['NumCPUs'])>=4
    cmd=['srun','--jobid=204827','--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=4',
         '--mem=32G','--gpus-per-task=1','--gpu-bind=single:1','--time=00:30:00','--unbuffered',
         '--job-name=t25ft50','bash',str(ROOT/'repo'/'run_ft50.sh')]
    log=ROOT/'training.log'
    with log.open('x') as handle:
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'parent_job':204827,'node':'auh7-1b-gpu-257','gpus':1,'cpus':4,'memory':'32G','time_limit':'00:30:00',
             'pid':proc.pid,'started_at_epoch':time.time(),'command':cmd,'log':str(log),
             'git_commit':subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip(),
             'training_output':str(ROOT/'run50'),'parent_unchanged':True,'new_allocation_submitted':False}
    temp=ROOT/'launch_receipt.json.tmp'
    temp.write_text(json.dumps(receipt,indent=2)+'\n');os.replace(temp,ROOT/'launch_receipt.json')
    print(json.dumps(receipt),flush=True)
