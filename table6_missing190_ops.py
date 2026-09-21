"""One held-first, short allocation for nine real smokes then missing190 HPO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import time
from table6_restart_ops import ROOT, REPO, PY, EXCLUDE, run, atomic
from table6_restart_deadline import parse_fields

NAME = 'table6_missing190_standard_hpo_20260922_v1'
STAGE = ROOT/'stage'/NAME
OUT = ROOT/'evaluation'/NAME
LOGS = ROOT/'logs'/NAME
PLAN = OUT/'plan.json'
LEDGER = STAGE/'submission.json'
BASE = ROOT/'stage/table6_remaining19_standard_hpo_bg8_20260912_v1'
PREDECESSOR = '208377'
FILES = ('table6_missing190_ops.py', 'table6_missing190_plan.py',
         'table6_missing190_fit.py', 'table6_missing190_worker.py',
         'table6_restart_deadline.py', 'table6_restart_ops.py')

def sources():
    return {name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in FILES}

def script():
    return f'''#!/usr/bin/env bash
#SBATCH --job-name=t6gap190q
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@90
#SBATCH --nice=0
#SBATCH --no-requeue
#SBATCH --dependency=afterany:{PREDECESSOR}
#SBATCH --exclude={EXCLUDE}
#SBATCH --chdir={STAGE}
#SBATCH --output={LOGS}/slurm-%j.out
#SBATCH --error={LOGS}/slurm-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH={BASE}:{BASE}/TALENT:{ROOT}/stage/table6_nonfoundation_all_benchmarks_20260823_v1/python_packages
export T6_BASE_STAGE={BASE} T6_MISSING190_PLAN={PLAN}
{PY} -B {REPO}/table6_missing190_ops.py verify
BUDGET_EXPORTS="$({PY} -B {REPO}/table6_restart_deadline.py --job-id "$SLURM_JOB_ID" --format exports)"
eval "$BUDGET_EXPORTS"
{PY} -B {BASE}/manage.py check-inputs
srun --exact --exclusive --input=none --nodes=1 --ntasks=1 --cpus-per-task=16 \\
 --gpus-per-task=1 --mem=256G --cpu-bind=cores --gpu-bind=single:1 --kill-on-bad-exit=1 \\
 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \\
 {PY} {BASE}/rocm_gpu_entry.py {PY} -B {REPO}/table6_missing190_ops.py node
'''

def plan():
    from table6_missing190_fit import validate_plan
    return validate_plan(json.loads(PLAN.read_text()))

def verify():
    deployment = json.loads((STAGE/'deployment.json').read_text())
    assert sources() == deployment['source_hashes'], 'queued/runtime source changed'
    assert (STAGE/'run.sh').read_text() == script()
    assert plan()['plan_id'] == deployment['plan_id']

def control(job):
    d = parse_fields(run(['scontrol','show','job','-o',job]))
    expected = {'JobId':job,'JobName':'t6gap190q','JobState':'PENDING','Reason':'JobHeldUser',
      'Partition':'faculty','Account':'faculty-acc','QOS':'bgqos','NumTasks':'1',
      'NumCPUs':'16','CPUs/Task':'16','TimeLimit':'02:00:00','MinMemoryNode':'256G',
      'Nice':'0','Requeue':'0','Command':str(STAGE/'run.sh'),'WorkDir':str(STAGE),
      'StdOut':str(LOGS/f'slurm-{job}.out'),'StdErr':str(LOGS/f'slurm-{job}.err')}
    assert all(d.get(k)==v for k,v in expected.items()), {k:(d.get(k),v) for k,v in expected.items() if d.get(k)!=v}
    assert d['NumNodes'] in ('1','1-1')
    assert 'gres/gpu=1' in d['ReqTRES'].split(',') and 'gres/gpu=1' in d['TresPerTask'].split(',')
    assert d['Dependency'] in (f'afterany:{PREDECESSOR}(unfulfilled)',f'afterany:{PREDECESSOR}(fulfilled)'), d['Dependency']
    assert set(run(['scontrol','show','hostnames',d['ExcNodeList']]).splitlines()) == set(run(['scontrol','show','hostnames',EXCLUDE]).splitlines())
    spool = STAGE/f'spool-{job}.sh'
    assert not spool.is_symlink()
    if not spool.exists():
        run(['scontrol','write','batch_script',job,str(spool)])
    assert spool.read_text()==script()

def main(mode):
    if mode=='prepare':
        from table6_missing190_plan import build, publish_new, FIXED_NAME, R19_NAME
        assert not PLAN.exists(), 'plan already prepared; do not overwrite'
        value = build(ROOT/'stage'/FIXED_NAME/'plan.json', ROOT/'stage'/R19_NAME/'plan.json', OUT)
        publish_new(PLAN,value)
        return {'plan_id':value['plan_id'],'pairs':len(value['pairs']),'datasets':len(value['rows']),
                'method_counts':value['pair_counts_by_method']}
    if mode=='verify':
        verify()
        return {'verified':True}
    if mode=='node':
        verify()
        from table6_missing190_fit import METHODS
        from table6_restart_deadline import EnvironmentBudget
        assert os.environ['SLURM_PROCID']=='0' and os.environ['SLURM_NTASKS']=='1'
        budget = EnvironmentBudget.from_environment()
        stopped = []
        active = [None]
        def signal_stop(signum, frame):
            stopped.append(signum)
            child = active[0]
            if child is not None and child.poll() is None:
                try:
                    child.send_signal(signum)
                except ProcessLookupError:
                    pass
        signal.signal(signal.SIGUSR1, signal_stop)
        signal.signal(signal.SIGTERM, signal_stop)
        command = [PY,'-B',str(REPO/'table6_missing190_worker.py')]
        actions = [['smoke','--plan',str(PLAN),'--method',method] for method in METHODS]
        actions += [['preflight','--plan',str(PLAN),'--mode','gpu'],
                    ['gate','--plan',str(PLAN),'--ranks','1']]
        for arguments in actions:
            if stopped or budget.remaining()<=180:
                return {'state':'paused_before_formal','reason':'allocation_budget_or_signal'}
            child = subprocess.Popen(command+arguments)
            active[0] = child
            try:
                # Also cover a signal arriving between the safe point and
                # installing the owned child handle.
                if stopped and child.poll() is None:
                    child.send_signal(stopped[-1])
                returncode = child.wait()
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                raise
            finally:
                active[0] = None
            if stopped or budget.remaining()<=180:
                return {'state':'paused_before_formal','reason':'allocation_budget_or_signal'}
            if returncode:
                raise subprocess.CalledProcessError(returncode, command+arguments)
        # Replace the launcher so Slurm signals reach the bounded formal worker
        # directly, with no unhandled supervisor SIGUSR1 cancelling children.
        os.execv(PY, command+['worker','--plan',str(PLAN),'--mode','gpu'])
    if mode=='submit':
        assert not LEDGER.exists(), 'existing/uncertain submission; no duplicate'
        p = plan()
        previous = parse_fields(run(['scontrol','show','job','-o',PREDECESSOR]))
        assert previous['JobId']==PREDECESSOR and previous['JobName']=='t6r22g01'
        assert previous['JobState'] in ('PENDING','RUNNING','CONFIGURING')
        assert previous['WorkDir']==str(ROOT/'stage/table6_completion_restart_20260922_v1')
        queue = run(['squeue','-u','guangyi.chen','-h','-o','%i|%j|%Z'])
        assert not any('|t6gap190q|' in line for line in queue.splitlines())
        STAGE.mkdir(parents=True,exist_ok=True); LOGS.mkdir(parents=True,exist_ok=True)
        with (STAGE/'run.sh').open('x') as stream: stream.write(script())
        run(['bash','-n',str(STAGE/'run.sh')])
        atomic(STAGE/'deployment.json',{'source_hashes':sources(),'plan_id':p['plan_id'],
               'repo_commit':run(['git','-C',str(REPO),'rev-parse','HEAD'])})
        state = {'state':'submitting','created_epoch':time.time(),'plan_id':p['plan_id'],
                 'afterany':PREDECESSOR,'scope':'190 missing classification pairs; nine real smokes before formal 100-HPO/15-seed fits',
                 'execution':'GPU-reserved rank; tree model children CPU-only'}
        atomic(LEDGER,state)
        raw = run(['sbatch','--hold','--parsable',str(STAGE/'run.sh')])
        job = raw.split(';')[0]; assert job.isdigit(), raw
        state.update(job_id=job,state='held'); atomic(LEDGER,state)
        control(job); state['state']='verified_held'; atomic(LEDGER,state)
        return state
    if mode=='release':
        verify(); state=json.loads(LEDGER.read_text()); assert state['state']=='verified_held'
        control(state['job_id']); state['state']='releasing'; atomic(LEDGER,state)
        run(['scontrol','release',state['job_id']])
        state['state']='released'; atomic(LEDGER,state)
        return state
    raise ValueError(mode)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','verify','node','submit','release'))
    print(json.dumps(main(parser.parse_args().mode),allow_nan=False))
