"""Submit one isolated, CPU-only AutoGluon continuation; held first."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from table6_restart_ops import ROOT, REPO, PY, EXCLUDE, run, atomic
from table6_restart_deadline import parse_fields

STAGE = ROOT/'stage/table6_autogluon_restart_20260922_v1'
LOGS = ROOT/'logs/table6_autogluon_restart_20260922_v1'
BASE = ROOT/'stage/table6_standard457_fixedbest_bg1_gpu_recovery_20260909_v2'
OUT = ROOT/'evaluation/table6_standard457_fixedbest_gt1_20260909_v1'
LEDGER = STAGE/'submission.json'
PLAN = '95ed50cd7348ed167f71ff159f6af14cc351e67959d6d227f9201bdc024fbb85'
FILES = ('table6_restart_ag.py','table6_restart_ag_ops.py',
         'table6_restart_deadline.py','table6_restart_ops.py')


def sources():
    return {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in FILES}


def precheck():
    assert run(['git','-C',str(BASE),'rev-parse','HEAD']) == 'cab60a46949c52782b7a909b48b49fc7096cf767'
    assert not run(['git','-C',str(BASE),'status','--porcelain','--untracked-files=no'])
    plan = json.loads((BASE/'plan.json').read_text())
    assert plan['plan_id'] == PLAN and plan['seeds'] == list(range(15)) and plan['hpo_trials'] == 0
    pairs = [p for p in plan['pairs'] if p['method']=='AutoGluon']
    assert len(pairs) == 457
    absent = sum(not (OUT/'results/AutoGluon'/(p['dataset']+'.json')).exists() for p in pairs)
    return {'target':457, 'absent_aggregate_files':absent,
            'caveat':'count is not strict completion audit; workers strictly validate before reuse'}


def script():
    fastai = ROOT/'stage/limix2m_table6_deep_cpu_augmented_fastai_20260827_v2/fastai_overlay'
    packages = ROOT/'stage/table6_nonfoundation_all_benchmarks_20260823_v1/python_packages'
    return f'''#!/usr/bin/env bash
#SBATCH --job-name=t6ag22cpu
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus=0
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@90
#SBATCH --nice=0
#SBATCH --no-requeue
#SBATCH --exclude={EXCLUDE}
#SBATCH --chdir={STAGE}
#SBATCH --output={LOGS}/slurm-%j.out
#SBATCH --error={LOGS}/slurm-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH={BASE}:{BASE}/runtime:{fastai}:{packages}
{PY} -B {REPO}/table6_restart_ag_ops.py verify
BUDGET_EXPORTS="$({PY} -B {REPO}/table6_restart_deadline.py --job-id "$SLURM_JOB_ID" --format exports)"
eval "$BUDGET_EXPORTS"
srun --exact --exclusive --input=none --nodes=1 --ntasks=4 --ntasks-per-node=4 \\
 --cpus-per-task=16 --gpus-per-task=0 --gres=none --mem=512G --cpu-bind=cores --kill-on-bad-exit=1 \\
 env CPU_ONLY=1 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 GPU_DEVICE_ORDINAL=-1 \\
 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \\
 nice -n 19 ionice -c 3 {PY} -B {REPO}/table6_restart_ag.py launch
'''


def verify():
    d = json.loads((STAGE/'deployment.json').read_text())
    assert d['source_hashes'] == sources() and d['plan_id'] == PLAN
    assert (STAGE/'run.sh').read_text() == script()
    precheck()


def require_no_gpus(fields):
    """Reject typed GPUs and per-node/per-task GPU requests as well as totals."""
    for field in ('ReqTRES', 'AllocTRES', 'TresPerNode', 'TresPerTask', 'TresPerJob', 'Gres'):
        for token in fields.get(field, '').split(','):
            if token.startswith(('gres/gpu', 'gpu')):
                match = re.fullmatch(r'(?:gres/)?gpu(?::[^,:=]+)?[:=](\d+)', token)
                assert match and int(match.group(1)) == 0, (field, token, 'CPU-only contract')


def control(job, held=True):
    assert isinstance(job, str) and job.isdigit(), 'numeric job identity required'
    raw = run(['scontrol','show','job','-o',job])
    d = parse_fields(raw)
    expected = {'JobId':job,'JobName':'t6ag22cpu','Partition':'faculty','Account':'faculty-acc',
      'QOS':'bgqos','NumTasks':'4','CPUs/Task':'16','NumCPUs':'64','TimeLimit':'02:00:00',
      'MinMemoryNode':'512G','Nice':'0','Requeue':'0','Dependency':'(null)',
      'WorkDir':str(STAGE),'Command':str(STAGE/'run.sh'),
      'StdOut':str(LOGS/f'slurm-{job}.out'),'StdErr':str(LOGS/f'slurm-{job}.err')}
    if held:
        expected.update(JobState='PENDING',Reason='JobHeldUser')
    for key,value in expected.items():
        assert d.get(key) == value,(key,d.get(key),value)
    assert d['NumNodes'] in ('1','1-1') and d['NtasksPerN:B:S:C'].split(':')[0]=='4'
    tres = dict(t.split('=', 1) for t in d['ReqTRES'].split(','))
    assert tres['cpu']=='64'
    require_no_gpus(d)
    assert set(run(['scontrol','show','hostnames',d['ExcNodeList']]).splitlines()) == set(run(['scontrol','show','hostnames',EXCLUDE]).splitlines())
    spool=STAGE/f'spool-{job}.sh'
    assert not spool.is_symlink()
    if not spool.exists():
        run(['scontrol','write','batch_script',job,str(spool)])
    assert spool.read_text()==script()
    return d


def main(mode):
    if mode=='verify':
        verify()
        return {'verified':True}
    if mode=='status':
        return {'ledger':json.loads(LEDGER.read_text()) if LEDGER.exists() else None,
                'queue':run(['squeue','-u','guangyi.chen','-h','-o','%i|%j|%T|%q|%b|%R']),
                'workload':precheck()}
    if mode=='submit':
        assert not LEDGER.exists(),'existing/uncertain submission; never repeat sbatch'
        queue=run(['squeue','-u','guangyi.chen','-h','-o','%i|%j|%Z'])
        assert not any('|t6ag22cpu|' in x for x in queue.splitlines())
        before=precheck()
        assert before['absent_aggregate_files']>0
        STAGE.mkdir(parents=True,exist_ok=True)
        LOGS.mkdir(parents=True,exist_ok=True)
        with (STAGE/'run.sh').open('x') as f:f.write(script())
        run(['bash','-n',str(STAGE/'run.sh')])
        atomic(STAGE/'deployment.json',{'source_hashes':sources(),'plan_id':PLAN,
          'repo_commit':run(['git','-C',str(REPO),'rev-parse','HEAD'])})
        state={'state':'submitting','created_epoch':time.time(),'plan_id':PLAN,'before':before}
        atomic(LEDGER,state)
        raw=run(['sbatch','--hold','--parsable',str(STAGE/'run.sh')])
        job=raw.split(';')[0]
        assert job.isdigit(),raw
        state.update(job_id=job,state='held')
        atomic(LEDGER,state)
        control(job)
        state['state']='verified_held'
        atomic(LEDGER,state)
        return state
    if mode=='release':
        verify()
        state=json.loads(LEDGER.read_text())
        assert state['state']=='verified_held'
        control(state['job_id'])
        state['state']='releasing'
        atomic(LEDGER,state)
        run(['scontrol','release',state['job_id']])
        state['state']='released'
        atomic(LEDGER,state)
        return state
    raise ValueError(mode)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('verify','status','submit','release'))
    print(json.dumps(main(p.parse_args().mode),allow_nan=False))
