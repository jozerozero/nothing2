"""Held-first Table6 continuation; no old jobs, plans, results or claims edited."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
REPO = Path(__file__).resolve().parent
NAME = 'table6_completion_restart_20260922_v1'
STAGE = ROOT / 'stage' / NAME
LOGS = ROOT / 'logs' / NAME
LEDGER = STAGE / 'submission.json'
BASE = ROOT / 'stage/table6_remaining19_standard_hpo_bg8_20260912_v1'
OUT = ROOT / 'evaluation/table6_remaining19_standard_hpo_bg8_20260912_v1'
PY = '/vast/users/guangyi.chen/anaconda3/envs/tabicl/bin/python'
PLAN = '47c3d448249235ad4f7ae6248424a1f8f38094ae204d2d4ece472b4effb61aff'
EXCLUDE = 'auh7-1b-gpu-[193,195,207,216,228,239,274,287,292,296]'
FILES = ('table6_restart_ops.py', 'table6_restart_gpu.py', 'table6_restart_deadline.py')
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED'}


def run(args, timeout=120):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((args, p.returncode, p.stdout[-3000:], p.stderr[-3000:]))
    return p.stdout.strip()


def atomic(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sources():
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in FILES}


def queue():
    return run(['squeue', '-u', 'guangyi.chen', '-h', '-o', '%i|%j|%T|%Z'])


def workload():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONPATH=f'{BASE}:{BASE}/TALENT:{ROOT}/stage/table6_nonfoundation_all_benchmarks_20260823_v1/python_packages')
    check = subprocess.run([PY, '-B', str(BASE/'manage.py'), 'check-inputs'], env=env,
                           capture_output=True, text=True, timeout=300)
    assert check.returncode == 0 and 'IMMUTABLE_INPUTS_PASS' in check.stdout, check.stderr
    import sys
    sys.path.insert(0, str(BASE))
    from common import load_plan, CPU_METHODS
    from worker import validate_complete
    plan = load_plan()
    assert plan['plan_id'] == PLAN and len(plan['pairs']) == 12482
    count = dict(complete=0, eligible=0, errors=0, deferred=0)
    for pair in plan['pairs']:
        if pair['method'] in CPU_METHODS:
            continue
        key = pair['key']
        result = OUT/'results'/pair['method']/(key+'.json')
        if result.exists():
            validate_complete(json.loads(result.read_text()), pair, plan)
            count['complete'] += 1
        elif (OUT/'errors'/(key+'.json')).exists():
            count['errors'] += 1
        elif (OUT/'short2h_deferred'/(key+'.json')).exists():
            count['deferred'] += 1
        else:
            count['eligible'] += 1
    return count


def script():
    return f'''#!/usr/bin/env bash
#SBATCH --job-name=t6r22g
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@90
#SBATCH --nice=0
#SBATCH --no-requeue
#SBATCH --distribution=block:block
#SBATCH --exclude={EXCLUDE}
#SBATCH --chdir={STAGE}
#SBATCH --output={LOGS}/slurm-%j.out
#SBATCH --error={LOGS}/slurm-%j.err
set -euo pipefail
export T6_BASE_STAGE={BASE}
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH={BASE}:{BASE}/TALENT:{ROOT}/stage/table6_nonfoundation_all_benchmarks_20260823_v1/python_packages
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{PY} -B {REPO}/table6_restart_ops.py verify
BUDGET_EXPORTS="$({PY} -B {REPO}/table6_restart_deadline.py --job-id "$SLURM_JOB_ID" --format exports)"
eval "$BUDGET_EXPORTS"
export JOB_BUDGET_END_EPOCH JOB_BUDGET_END_MONOTONIC
{PY} -B {BASE}/manage.py check-inputs
srun --exact --exclusive --input=none --nodes=1 --ntasks=8 --ntasks-per-node=8 \\
 --cpus-per-task=8 --gpus-per-task=1 --mem=512G --cpu-bind=cores --gpu-bind=single:1 --kill-on-bad-exit=1 \\
 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \\
 {PY} {BASE}/rocm_gpu_entry.py {PY} -B {REPO}/table6_restart_gpu.py launch --mode gpu
'''


def verify():
    deployment = json.loads((STAGE/'deployment.json').read_text())
    assert deployment['source_hashes'] == sources(), 'restart sources changed'
    assert (STAGE/'run.sh').read_text() == script(), 'submitted script changed'
    assert deployment['plan_id'] == PLAN


def control(job, job_name, held=False):
    text = run(['scontrol', 'show', 'job', '-o', str(job)])
    d = dict(re.findall(r'(\S+?)=(\S+)', text))
    expected = {'Partition':'faculty', 'Account':'faculty-acc', 'QOS':'bgqos',
                'NumTasks':'8', 'CPUs/Task':'8', 'NumCPUs':'64', 'TimeLimit':'02:00:00',
                'Nice':'0', 'Requeue':'0', 'Dependency':'(null)', 'WorkDir':str(STAGE),
                'Command':str(STAGE/'run.sh'), 'MinMemoryNode':'512G',
                'StdOut':str(LOGS/f'slurm-{job}.out'), 'StdErr':str(LOGS/f'slurm-{job}.err')}
    for key, value in expected.items():
        assert d.get(key) == value, (key, d.get(key), value)
    assert {'gres/gpu=8', 'cpu=64'} <= set(d['ReqTRES'].split(',')), d['ReqTRES']
    assert d.get('NumNodes') in ('1','1-1')
    assert 'gres/gpu=1' in d.get('TresPerTask','').split(',')
    assert d.get('NtasksPerN:B:S:C','').split(':')[0] == '8'
    assert re.fullmatch(r't6r22g(?:0[1-9]|10)', job_name)
    assert d.get('JobId') == str(job) and d.get('JobName') == job_name
    excluded = set(run(['scontrol','show','hostnames',d['ExcNodeList']]).splitlines())
    assert excluded == set(run(['scontrol','show','hostnames',EXCLUDE]).splitlines())
    if held:
        assert d['JobState'] == 'PENDING' and d['Reason'] == 'JobHeldUser', d
    spool = STAGE / f'spool-{job}.sh'
    assert not spool.is_symlink()
    if not spool.exists():
        run(['scontrol', 'write', 'batch_script', str(job), str(spool)])
    assert script() == spool.read_text(), 'spooled script differs'
    return d


def main(mode, jobs=10):
    assert 1 <= jobs <= 10
    if mode == 'verify':
        verify()
        return {'verified': True}
    if mode == 'status':
        return {'queue': queue(), 'ledger': json.loads(LEDGER.read_text()) if LEDGER.exists() else None,
                'workload': workload(), 'observed_epoch': time.time()}
    if mode == 'submit':
        assert not LEDGER.exists(), 'existing or uncertain ledger: inspect, never resubmit'
        q = queue()
        assert not any('|t6r22g' in x for x in q.splitlines()), 'existing restart queue entries'
        work = workload()
        assert work['eligible'] > 0
        STAGE.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        with (STAGE/'run.sh').open('x') as f:
            f.write(script())
        run(['bash', '-n', str(STAGE/'run.sh')])
        atomic(STAGE/'deployment.json', {'source_hashes': sources(), 'plan_id': PLAN,
               'repo_commit': run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])})
        state = {'plan_id': PLAN, 'created_epoch': time.time(), 'before':work, 'jobs':[],
                 'state':'submitting', 'scope':'16 GPU methods; original 100 HPO / 15 final seeds',
                 'claim_policy':'original nonblocking per-pair flock; preserve complete results'}
        atomic(LEDGER, state)
        for index in range(jobs):
            state['submission_attempt'] = index
            atomic(LEDGER, state)
            # Uncertain response is deliberately not retried.
            raw = run(['sbatch','--hold','--parsable',f'--job-name=t6r22g{index+1:02d}',str(STAGE/'run.sh')])
            job = raw.split(';')[0]
            assert job.isdigit(), raw
            state['jobs'].append({'job_id':job, 'job_name':f't6r22g{index+1:02d}', 'state':'held'})
            atomic(LEDGER, state)
            control(job, f't6r22g{index+1:02d}', held=True)
        state['state'] = 'verified_held'
        atomic(LEDGER, state)
        return state
    if mode == 'release':
        verify()
        state = json.loads(LEDGER.read_text())
        assert state['state'] == 'verified_held', state['state']
        for job in state['jobs']:
            control(job['job_id'], job['job_name'], held=True)
        state['state'] = 'releasing'
        atomic(LEDGER, state)
        for job in state['jobs']:
            run(['scontrol','release',job['job_id']])
            job['state'] = 'released'
            atomic(LEDGER, state)
        state['state'] = 'released'
        atomic(LEDGER, state)
        return state
    raise ValueError(mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('status','submit','release','verify'))
    parser.add_argument('--jobs', type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(main(args.mode, args.jobs), allow_nan=False))
