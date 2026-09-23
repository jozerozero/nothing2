#!/usr/bin/env python3
"""Observe unchanged native TabSwift probabilities in an isolated GPU diagnostic.

No canonical result, claim, manifest, model or prediction is changed. The original
worker's probability check still executes and may fail; that failure is evidence.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
STAGE = Path(__file__).resolve().parent
REPO = ROOT/'stage/reg_loop3_step22175_finetune50_20260921_v1/repo'
DEFAULT_OUT = ROOT/'evaluation/tabfm_tabswift_comparison_20260923_v1/probability_diagnostic_v1'
PYTHON = ROOT/'stage/tabswift_standard681_20260922_v1/venv/bin/python'
NAME = 'tsw23probdiag'
EXCLUDE = 'auh7-1b-gpu-[193,195,207,216,228,239,274,287,292,296]'
CAMPAIGNS = [ROOT/'evaluation'/('tabswift_'+v+'_standard681_20260922_v1')/'manifest.json'
             for v in ('official16', 'budget32x8')]
MANIFEST_IDS = ['2636231e398e59cd98ed23f94780f4fc4012be52b9106680b8d39a59398482cf',
                '68160d6bd3b1df003be016e40192ba2fb1797c82ce0561c90b494316e1e7ebc5']
# Replaced at implementation time with hashes of the existing frozen source files.
NATIVE_HASHES = {'allocated_gpu_uuid.py': '7938873c53685448c4245f27e5527151bd85dd277fd400b1adabb757fa859ec9', 'tabswift_one.py': 'deffc74232f62158980eeeda091babd96ccd0d71c1bfdc955a1ffcd0fcca1090', 'tabswift_ensemble.py': '2dc869495f370f7c59e25ec0f62e17fe83c8635fd282628d86836d4d1df15aa6', 'foundation_alloc8.py': 'a3af090d69edb0e4d4e438a35b44dddaae86c0cd3f55ec410407ae6e3eae6568', 'shared_foundation_full_v2.py': 'c49f5509f059ac08f7ea215e57e476b3b54bfa26817752efe35daf768850566d', 'shared_foundation_sidecar.py': '01097f4ea725e39590594d5123579afb92a6c0149a8947f05a0e55a0776f4fe4', 'tabfm_default_dispatch.py': '7f3c7672ff0100247edf0b150fa7cf683d5543e8ab28284105470177d20ab30b', 'tabfm_default_one.py': '087eb33047fda520194d7ed7489e87f9420e87a2128e1b4817bd45be8e931543', 'tabfm_local_tmp.py': '1c07095857ecbb5f90e453c10342ef33053196a03d118604c695d70fbe015ef4', 'table6_restart_deadline.py': '047352b1b52c947ea3b2017692c422d65f451b56dca18283564383b51a8b9950', 'table6_restart_ag.py': '3ca3bf4d620b026f29409a9361630f0110227c116a68317a975948ccb79ecfaf', 'pfn_mitra_one.py': '070d4f6f24ac5183654191e5c870a1dfb2a2ec7de57e153f72a33c4dd3fe4500', 'eval_one.py': '73dffec1851e73c6a188218d375b79b8ba864b3b58772e7edeba93ba61ab03d5', 'classification32_dispatch.py': 'bbd215e0aa27d6c40554b99d4b7364e75e0c7999adb678508f1bf8261a9b7dad', 'classification32_campaign.py': '70019996bc1276f212e9ecbe0d51da66f3b5582d0a11cd5a2151b72145f8eb77', 'classification32_submit.py': 'e714b7ffa925b06cf3f067c6688d861b3b6e32750f09f20abe4f0fa38cf83ec8'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/unsafe JSON: '+str(path))
    return json.loads(path.read_text())


def publish(path, value):
    path = Path(path)
    require(not path.is_symlink() and not path.parent.is_symlink(), 'Refuse symlink publication')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run(argv, timeout=60):
    p = subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=timeout)
    require(p.returncode == 0, repr((argv, p.returncode, p.stdout[-3000:], p.stderr[-3000:])))
    return p.stdout.strip()


def native():
    require(bool(NATIVE_HASHES), 'Missing fixed native source hashes')
    for name, expected in NATIVE_HASHES.items():
        path = REPO/name
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                'Frozen native source changed: '+name)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    import allocated_gpu_uuid as binding
    import tabswift_one as worker
    import foundation_alloc8 as runtime
    require(Path(binding.__file__).resolve() == REPO/'allocated_gpu_uuid.py' and
            Path(worker.__file__).resolve() == REPO/'tabswift_one.py' and
            Path(runtime.__file__).resolve() == REPO/'foundation_alloc8.py', 'Wrong native imports')
    return binding, worker, runtime


def shell_script(plan_path, out):
    Q = shlex.quote
    return f'''#!/bin/bash
#SBATCH --job-name={NAME}
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus=8
#SBATCH --mem=256G
#SBATCH --time=00:10:00
#SBATCH --nice=0
#SBATCH --no-requeue
#SBATCH --exclude={EXCLUDE}
#SBATCH --chdir={out}
#SBATCH --output={out}/slurm-%j.out
#SBATCH --error={out}/slurm-%j.err
set -euo pipefail
[[ $# == 0 ]] || exit 2
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
exec {Q(str(PYTHON))} -B {Q(str(Path(__file__).resolve()))} launch --plan {Q(str(plan_path))}
'''


def prepare(out):
    out = Path(out)
    require(out.is_absolute() and out.is_relative_to(ROOT/'evaluation') and not out.exists(),
            'Prepare requires a new directory below evaluation')
    require(not any(p.is_symlink() for p in [out, *out.parents]), 'Symlink output ancestor')
    _binding, worker, _runtime = native()
    records = []
    for path, mid in zip(CAMPAIGNS, MANIFEST_IDS):
        man = worker.load_campaign(path)
        require(man['manifest_id'] == mid and man['worker_python'] == str(PYTHON), 'Frozen campaign changed')
        records.append({'path': str(path), 'sha256': sha(path), 'manifest_id': mid})
    body = shell_script(out/'plan.json', out)
    check = subprocess.run(['bash', '-n'], input=body, text=True, capture_output=True)
    require(check.returncode == 0, 'Invalid generated shell: '+check.stderr)
    out.mkdir(parents=True, exist_ok=False)
    with (out/'run.sh').open('x') as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    plan = {'schema': 'tabswift_probability_diagnostic_v1', 'output_root': str(out),
            'entry': str(Path(__file__).resolve()), 'entry_sha256': sha(__file__),
            'script_sha256': sha(out/'run.sh'), 'native_hashes': NATIVE_HASHES,
            'campaigns': records, 'worker_python': str(PYTHON),
            'resources': {'gpus': 8, 'cpus': 64, 'rank_cpus': 4, 'mem_gib': 256, 'seconds': 600},
            'rank_tasks': [{'campaign': i % 2, 'dataset_index': 234 if i < 2 else 150}
                           if i < 4 else None for i in range(8)],
            'scientific_worker_unmodified': True, 'canonical_publication': False, 'epoch': time.time()}
    plan['plan_id'] = digest(plan)
    publish(out/'plan.json', plan)
    verify(out/'plan.json')
    return {'state': 'prepared', 'plan': str(out/'plan.json'), 'plan_id': plan['plan_id']}


def verify(path):
    path = Path(path)
    require(path.is_absolute() and path.name == 'plan.json', 'Absolute diagnostic plan required')
    plan = read(path)
    require(plan['plan_id'] == digest({k:v for k,v in plan.items() if k != 'plan_id'}), 'Plan digest changed')
    require(plan['entry'] == str(Path(__file__).resolve()) and plan['entry_sha256'] == sha(__file__) and
            plan['native_hashes'] == NATIVE_HASHES and plan['worker_python'] == str(PYTHON), 'Runtime identity changed')
    out = Path(plan['output_root'])
    require(path == out/'plan.json' and out.is_relative_to(ROOT/'evaluation') and
            not any(p.is_symlink() for p in [out, *out.parents]), 'Unsafe diagnostic output')
    require(plan['resources'] == {'gpus':8,'cpus':64,'rank_cpus':4,'mem_gib':256,'seconds':600}, 'Resources changed')
    require(sha(out/'run.sh') == plan['script_sha256'] and
            (out/'run.sh').read_text() == shell_script(path, out), 'Generated script changed')
    _binding, worker, _runtime = native()
    require(len(plan['campaigns']) == 2, 'Two original protocols required')
    for rec, expected_path, mid in zip(plan['campaigns'], CAMPAIGNS, MANIFEST_IDS):
        require(rec['path'] == str(expected_path) and rec['manifest_id'] == mid and
                sha(expected_path) == rec['sha256'], 'Campaign identity changed')
        require(worker.load_campaign(expected_path)['manifest_id'] == mid, 'Wrong scientific campaign')
    require(plan['rank_tasks'] == [{'campaign':i%2,'dataset_index':234 if i<2 else 150}
                                 if i<4 else None for i in range(8)], 'Diagnostic membership changed')
    return plan


def verify_job(job, raw, out, held):
    fields = dict(re.findall(r'([^\s=]+)=([^\s]+)', raw))
    expected = {'JobId':job, 'JobName':NAME, 'Partition':'faculty', 'Account':'faculty-acc',
                'QOS':'bgqos', 'NumTasks':'8', 'NumCPUs':'64', 'CPUs/Task':'8',
                'MinMemoryNode':'256G', 'TimeLimit':'00:10:00', 'Nice':'0', 'Requeue':'0',
                'Dependency':'(null)', 'Command':str(out/'run.sh'), 'WorkDir':str(out),
                'StdOut':str(out/f'slurm-{job}.out'), 'StdErr':str(out/f'slurm-{job}.err')}
    for key, value in expected.items():
        require(fields.get(key) == value, 'Scheduler contract changed: '+key+'='+str(fields.get(key)))
    require(fields.get('UserId','').endswith('('+str(os.getuid())+')') and
            fields.get('NumNodes') in ('1','1-1') and fields.get('TresPerTask') == 'cpu=8' and
            fields.get('NtasksPerN:B:S:C','').split(':')[0] == '8', 'Wrong owned one-node/eight-task allocation')
    tres = dict(s.split('=',1) for s in fields['ReqTRES'].split(','))
    require(tres.get('gres/gpu') == '8' and tres.get('cpu') == '64' and tres.get('mem') == '256G', 'Wrong requested TRES')
    excluded = set(run(['scontrol','show','hostnames',fields.get('ExcNodeList','')]).splitlines())
    wanted = set(run(['scontrol','show','hostnames',EXCLUDE]).splitlines())
    require(excluded and excluded <= wanted, 'Excluded-node contract missing')
    for key in ('NodeList','SchedNodeList'):
        if fields.get(key) not in (None,'(null)','None'):
            require(not set(run(['scontrol','show','hostnames',fields[key]]).splitlines()) & wanted,
                    'Allocated an excluded node')
    require(fields.get('JobState') == 'PENDING' and fields.get('Reason') == 'JobHeldUser'
            if held else fields.get('Reason') != 'JobHeldUser', 'Scheduler hold/release state unverified')
    return fields


def submit(path):
    plan = verify(path); out = Path(plan['output_root'])
    with (out/'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'submission-attempt.json').exists():
            return {'state':'attempt_already_exists_no_automatic_retry',
                    'submission':read(out/'submission.json') if (out/'submission.json').exists() else None}
        queue = run(['squeue','--me','-h','-o','%i|%j|%T'])
        require(not any(len(line.split('|'))>1 and line.split('|')[1] == NAME for line in queue.splitlines()),
                'Active matching diagnostic exists')
        args = ['sbatch','--hold','--parsable',str(out/'run.sh')]
        publish(out/'submission-attempt.json', {'plan_id':plan['plan_id'],'command':args,'epoch':time.time()})
        reply = run(args)
        require(re.fullmatch(r'[0-9]+(?:;[^;\s]+)?',reply), 'Uncertain submission; inspect journal, do not retry')
        job = reply.split(';')[0]
        publish(out/'submission.json', {'job_id':job,'plan_id':plan['plan_id'],'response':reply,'epoch':time.time()})
        raw = run(['scontrol','show','job','-o',job])
        fields = verify_job(job,raw,out,True)
        publish(out/'held-verified.json', {'job_id':job,'fields':fields,'plan_id':plan['plan_id']})
        verify(path)
        publish(out/'release-attempt.json', {'job_id':job,'epoch':time.time()})
        run(['scontrol','release',job])
        fields = verify_job(job,run(['scontrol','show','job','-o',job]),out,False)
        receipt = {'job_id':job,'fields':fields,'state':'released','plan_id':plan['plan_id'],'epoch':time.time()}
        publish(out/'release.json',receipt)
        return receipt


def launch(path):
    plan = verify(path); out = Path(plan['output_root'])
    require(os.environ.get('SLURM_JOB_ID','').isdigit() and os.environ.get('SLURM_JOB_NUM_NODES') == '1',
            'Launch only inside an allocated single-node Slurm job')
    binding, _worker, _runtime = native()
    env = dict(os.environ)
    for key in (*binding.MASKS,'PYTHONPATH','PYTHONHOME'):
        env.pop(key,None)
    env.update(PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    common = ['srun','--exact','--nodes=1','--gpus=8','--gpu-bind=none','--cpu-bind=threads',
              '--input=none','--unbuffered','--export=ALL']
    mapping = out/'mapping.json'
    publish(out/'job-start.json', {'job_id':os.environ['SLURM_JOB_ID'],'plan_id':plan['plan_id'],'epoch':time.time()})
    receipt = {'job_id':os.environ['SLURM_JOB_ID'],'plan_id':plan['plan_id'],'state':'failed'}
    try:
        subprocess.run(common+['--ntasks=1','--ntasks-per-node=1','--cpus-per-task=64','--kill-on-bad-exit=1',
            str(PYTHON),'-B',str(REPO/'allocated_gpu_uuid.py'),'bootstrap','--mapping',str(mapping),
            '--expected-cpus','64','--expected-mem-gib','256','--expected-seconds','600'],env=env,check=True)
        subprocess.run(common+['--ntasks=8','--ntasks-per-node=8','--cpus-per-task=4','--kill-on-bad-exit=0',
            str(PYTHON),'-B',str(REPO/'allocated_gpu_uuid.py'),'exec','--mapping',str(mapping),'--',
            str(PYTHON),'-B',str(Path(__file__).resolve()),'rank','--plan',str(path)],env=env,check=True)
        records = [read(out/'ranks'/f'rank-{i}'/'receipt.json') for i in range(8)]
        ready = [r.get('binding',{}) for r in records]
        require(all(r.get('rank') == i and r.get('runtime_visible_count') == 1 and
                    len(r.get('cpu_affinity',[])) == 4 for i,r in enumerate(ready)) and
                len({r.get('uuid') for r in ready}) == len({r.get('pci') for r in ready}) == 8 and
                len({c for r in ready for c in r['cpu_affinity']}) == 32, 'Incomplete/colliding eight-rank proof')
        receipt.update(state='diagnostics_collected_not_scientific_results',ranks=records)
    except BaseException:
        receipt['traceback'] = traceback.format_exc()
        raise
    finally:
        receipt['finished_epoch'] = time.time(); publish(out/'job-finished.json',receipt)
    return receipt


def rank_binding(plan):
    binding, _worker, _runtime = native()
    mapping = binding.load_mapping(Path(plan['output_root'])/'mapping.json')
    require(mapping['expected_cpus'] == 64 and mapping['expected_mem_gib'] == 256 and
            mapping['expected_seconds'] == 600, 'Wrong mapping resource contract')
    expected = binding.rank_environment(mapping,os.environ)
    require(all(os.environ.get(k) == expected.get(k) for k in
                (*binding.MASKS,'EXPECTED_GPU_UUID','EXPECTED_GPU_PCI_BUS_ID')), 'Must enter through native UUID binding')
    require(len(os.sched_getaffinity(0)) == 4, 'Rank must have exactly four CPUs')
    from table6_restart_deadline import EnvironmentBudget
    budget = EnvironmentBudget.from_environment()
    require(budget.remaining() > 30, 'Insufficient inherited diagnostic budget')
    import torch
    from pfn_mitra_one import gpu_identity
    gpu = gpu_identity(torch); rank = int(os.environ['SLURM_PROCID'])
    require(gpu['uuid'] == mapping['gpus'][rank]['uuid'] and gpu['pci_bus_id'] == mapping['gpus'][rank]['pci'],
            'Actual GPU differs from physical mapping')
    return {'job':mapping['job'],'node':socket.gethostname(),'rank':rank,'uuid':gpu['uuid'],
            'pci':gpu['pci_bus_id'],'runtime_visible_count':gpu['runtime_visible_count'],
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'mapping_id':mapping['mapping_id']}, budget


def rank(path):
    plan = verify(path); index = int(os.environ['SLURM_PROCID'])
    require(0 <= index < 8, 'Wrong rank')
    out = Path(plan['output_root'])/'ranks'/f'rank-{index}'
    out.mkdir(parents=True,exist_ok=False)
    receipt = {'rank':index,'plan_id':plan['plan_id'],'canonical_publication':False,'state':'error'}
    child = None
    try:
        identity,budget = rank_binding(plan); receipt['binding'] = identity
        if plan['rank_tasks'][index] is None:
            receipt['state'] = 'idle_rank_verified'
        else:
            _binding,_worker,runtime = native()
            command = [str(PYTHON),'-B',str(Path(__file__).resolve()),'observe','--plan',str(path)]
            env = dict(os.environ)
            # Native short TMPDIR changes only temporary storage, not model settings.
            env['TMPDIR'] = tempfile.mkdtemp(prefix='tswdiag-',dir='/tmp')
            with (out/'worker.log').open('x') as log:
                child = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    runtime.resource_guard(runtime.allocation_snapshot(),budget)
                    try: child.wait(timeout=1)
                    except subprocess.TimeoutExpired: pass
            receipt.update(state='diagnostic_worker_finished',exit_code=child.returncode,
                           observation=str(out/'observation.json'))
            require(child.returncode == 0 and (out/'observation.json').is_file(), 'Observation worker failed')
    except BaseException:
        receipt.update(state='error',traceback=traceback.format_exc())
    finally:
        if child is not None and child.poll() is None:
            _binding,_worker,runtime = native(); runtime.q.stop_child(child)
        receipt['finished_epoch'] = time.time(); publish(out/'receipt.json',receipt)
    return receipt


def observe(path):
    plan = verify(path); index = int(os.environ['SLURM_PROCID'])
    out = Path(plan['output_root'])/'ranks'/f'rank-{index}'
    identity,_budget = rank_binding(plan)
    task = plan['rank_tasks'][index]; require(task is not None,'Idle rank cannot observe')
    _binding,worker,_runtime = native()
    import numpy as np
    import tabswift_ensemble as ensemble
    original = ensemble.configure
    observation = {'plan_id':plan['plan_id'],'binding':identity,'task':task,'canonical_publication':False,
                   'original_evaluate_called':True,'prediction_returned_unchanged':True}

    def configure(estimator,*args,**kwargs):
        handle = original(estimator,*args,**kwargs)
        predict = estimator.predict_proba
        def capture(X,*a,**k):
            result = predict(X,*a,**k)
            raw = np.asarray(result)
            cast = raw.astype(np.float32)
            finite = np.isfinite(raw)
            stats = {'dtype':str(raw.dtype),'shape':list(raw.shape),'test_rows':len(X),
                     'classes':int(estimator.n_classes_),'nonfinite_count':int((~finite).sum()),
                     'negative_count':int((raw<0).sum()),'above_one_count':int((raw>1).sum()),
                     'minimum':float(raw[finite].min()) if finite.any() else None,
                     'maximum':float(raw[finite].max()) if finite.any() else None,
                     'old_shape_check':raw.shape==(len(X),estimator.n_classes_),
                     'old_finite_check':bool(np.isfinite(cast).all()),
                     'old_nonnegative_check':bool((cast>=0).all()),
                     'old_sum_check':bool(np.allclose(cast.sum(axis=1),1.,atol=1e-5)),
                     'old_sum_atol':1e-5,'old_sum_rtol':1e-5,
                     'argmax_unchanged_by_float32_cast':bool(np.array_equal(raw.argmax(axis=1),cast.argmax(axis=1))),
                     'dtype_epsilon':float(np.finfo(raw.dtype).eps)}
            sums = raw.sum(axis=1,dtype=np.float64)
            if np.isfinite(sums).all():
                errors = np.abs(sums-1)
                stats.update(row_sums_float64=sums.tolist(),max_abs_row_sum_error=float(errors.max()),
                    row_sum_error_quantiles=[float(v) for v in np.quantile(errors,[0,.5,.9,.99,1])],
                    rows_failing_old_sum=int((~np.isclose(cast.sum(axis=1),1.,atol=1e-5)).sum()))
            with (out/'raw_probability.npy').open('xb') as stream:
                np.save(stream,raw,allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
            stats.update(raw_probability_file=str(out/'raw_probability.npy'),
                         raw_probability_sha256=sha(out/'raw_probability.npy'))
            observation['probability'] = stats
            observation['ensemble_audit'] = handle.finish(len(X))
            return result
        estimator.predict_proba = capture
        return handle

    ensemble.configure = configure
    campaign_path = Path(plan['campaigns'][task['campaign']]['path'])
    try:
        campaign = worker.load_campaign(campaign_path)
        args = argparse.Namespace(campaign=campaign_path,task_kind='classification',
            dataset_index=task['dataset_index'],threads=4,output=out/'unpublished-result.json')
        result = worker.evaluate(args,campaign)
        observation.update(original_validation_passed=True,original_worker_metrics=result['metrics'])
    except BaseException as exc:
        observation.update(original_validation_passed=False,error=repr(exc),traceback=traceback.format_exc())
    finally:
        ensemble.configure = original
        observation['finished_epoch'] = time.time()
        publish(out/'observation.json',observation)
    return {'observation':str(out/'observation.json'),'original_validation_passed':observation['original_validation_passed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','verify','submit','launch','rank','observe'))
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUT)
    args = parser.parse_args()
    require(__debug__, 'Optimized Python unsupported')
    if args.mode == 'prepare': value = prepare(args.output_root)
    else:
        require(args.plan is not None,'--plan required')
        if args.mode == 'verify':
            plan = verify(args.plan); value = {'valid':True,'plan_id':plan['plan_id']}
        else: value = {'submit':submit,'launch':launch,'rank':rank,'observe':observe}[args.mode](args.plan)
    print(json.dumps(value,sort_keys=True,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
