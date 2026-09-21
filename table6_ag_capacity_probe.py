"""Two short, read-only host measurements; publish an exact-target proof."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from table6_restart_ag import publish
from table6_restart_ag_sidecar import STAGE, validate_parent, validate_proof

def main(parent, node, token):
    assert parent.isdigit() and token.isidentifier()
    assert socket.gethostname()==node and os.environ['SLURM_JOB_ID']==parent
    raw = subprocess.run(['scontrol','show','job','-o',parent],capture_output=True,text=True,check=True).stdout
    fields = validate_parent(raw,parent,node)
    allocated = int(fields['NumCPUs'])
    samples=[]
    for index in range(2):
        measured = subprocess.run(['vmstat','1','2'],capture_output=True,text=True,check=True).stdout
        values = measured.strip().splitlines()[-1].split()
        assert len(values)>=17
        idle = float(values[14]); assert 0<=idle<=100
        available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        samples.append({'observed_epoch':time.time(),'available_cpus':math.floor(allocated*idle/100),
                        'busy_cpus':allocated*(100-idle)/100, 'available_memory_bytes':available,
                        'vmstat_raw':measured,'interpretation':'whole-node measured idle capacity, not future reservation'})
    proof={'allow_cpu_sidecar':True,'parent_job_id':parent,'node':node,'observations':samples,
           'parent_snapshot':raw,'policy':'CPU nice19/I/O idle, Slurm exact exclusive acceptance; no parent mutation'}
    validate_proof(proof,parent,node)
    path=STAGE/'capacity'/(token+'.json')
    publish(path,proof)
    print(json.dumps({'path':str(path),'proof':proof},allow_nan=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--node',required=True);p.add_argument('--token',required=True)
    a=p.parse_args();main(a.parent,a.node,a.token)
