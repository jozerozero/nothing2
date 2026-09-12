"""Submit this authorized pair once, verify held allocations, then release."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

from prepare_deployment import STAGE, HERE, BASE, ROOT


def command(*args):
    p = subprocess.run(args, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f'{args}: {p.stdout}\n{p.stderr}')
    return p.stdout.strip()


def atomic(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    os.replace(temp, path)


def main():
    state_path = STAGE/'submission_state.json'
    assert not state_path.exists(), 'refuse duplicate submission state'
    receipt = json.loads((STAGE/'cpu_smoke.json').read_text())
    assert receipt['status'] == 'PASS_T25_G5SC_LOOP34_SMOKE'
    assert receipt['tested_passes'] == [1, 3, 4]
    # Compare every original command-line training argument before allocating.
    args = lambda s: dict(re.findall(r'^\s+--(\w+) (.*?)\s*\\?$', s, re.M))
    old = args((HERE/'run_tabiclv2_reg_fixed4096_8node_v1.sh').read_text())
    new = args((STAGE/'run_training.sh').read_text())
    assert set(new)-set(old) == {'shared_depth_icl_num_passes'}
    assert all(new[k] == v for k,v in old.items()), 'baseline CLI drift'
    names = ['rgt25sc3l1', 'rgt25sc4l1']
    assert not command('squeue','-h','-u','guangyi.chen','-n',','.join(names),'-o','%i'), 'active duplicate'
    for loop in (3,4):
        command('sbatch','--test-only',str(STAGE/f'loop{loop}.slurm'))
    fd = os.open(str(STAGE/'submission.intent'), os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o644)
    os.close(fd)
    state = {'baseline_training':151162, 'requested_by_user':'T25 + G5SC gated Loop3 and Loop4',
             'started_epoch':time.time(), 'source_revision':command('git','-C',str(HERE.parent),'rev-parse','HEAD'),
             'qos':'bgqos','account':'test-acc','components':{}, 'cpu_smoke':receipt}
    atomic(state_path,state)
    for loop in (3,4):
        raw = command('sbatch','--hold','--parsable',str(STAGE/f'loop{loop}.slurm'))
        job = raw.split(';')[0]
        assert job.isdigit(), raw
        arm = state['components'][str(loop)] = {'job_id':job,'name':f'rgt25sc{loop}l1',
            'state':'SUBMITTED_HELD','gpu_count':64,'nodes':8,
            'checkpoint_dir':str(ROOT/f'checkpoints/t25_g5sc_loop{loop}_bg64_20260912_v1/rgt25sc{loop}l1-{job}')}
        atomic(state_path,state)
        control = command('scontrol','show','job','-o',job)
        fields = dict(re.findall(r'(\w+)=(\S+)',control))
        for key,value in {'JobName':arm['name'],'Account':'test-acc','QOS':'bgqos',
                          'Partition':'faculty','NumNodes':'8','NumTasks':'8','NumCPUs':'1024',
                          'CPUs/Task':'128','Requeue':'0','Nice':'0','TimeLimit':'3-00:00:00'}.items():
            if key == 'CPUs/Task':
                assert 'CPUs/Task=128' in control
            else:
                assert fields.get(key) == value, (key,fields.get(key),value)
        assert 'gres/gpu=64' in fields['ReqTRES'] and 'mem=16T' in fields['ReqTRES'], control
        assert fields['WorkDir'] == str(STAGE)
        assert fields['Command'] == str(STAGE/f'loop{loop}.slurm')
        assert fields['Dependency'] == '(null)', control
        arm['held_control'] = control
        atomic(state_path,state)
    for loop in (3,4):
        arm = state['components'][str(loop)]
        command('scontrol','release',arm['job_id'])
        arm['released_control'] = command('scontrol','show','job','-o',arm['job_id'])
        arm['state'] = 'RELEASED'
        atomic(state_path,state)
    state['completed_epoch'] = time.time()
    atomic(state_path,state)
    print(json.dumps(state),flush=True)


if __name__=='__main__':
    main()
