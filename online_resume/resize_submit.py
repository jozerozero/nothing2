"""One pending-job resize, preserving completed data and recovery provenance."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from recovery import ROOT, STAGE, OUT, NAME, TAG, POLICY, signatures, validate_shard
import gpu_shard_group_scheduler_stage1 as scheduler

PREVIOUS = 181614
PREVIOUS_TAG = 'e4_g5sc_loop3_resume181407_eval_fp32_online_24gpu_gt_step50_20260911_r2'
PREVIOUS_STAGE = ROOT / 'stage' / PREVIOUS_TAG / 'online_resume'
RESOURCES = dict(account='faculty-acc', partition='faculty', qos='bgqos', nodes=4,
                 gpus=32, tasks_per_node=8, group_count=8, cpus_per_task=4,
                 memory_per_node='120G', time_limit='3-00:00:00', nice=0,
                 dependency=None, requeue=False, gpu_binding='single:1')

def run(args):
    p = subprocess.run(args, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError((args, p.returncode, p.stdout, p.stderr))
    return p.stdout.strip()

def job_control(job):
    raw = run(['scontrol', 'show', 'job', str(job), '-o'])
    fields = dict(x.split('=', 1) for x in raw.split() if '=' in x)
    return fields, raw

def queue_conflicts(allowed):
    lines = run(['squeue', '--me', '-h', '-o', '%A|%j|%T']).splitlines()
    names = {NAME, 'e4g5sc3r24gt2', 'e4g5sc3r24gt1', 'e4g5sc3on24gt2'}
    return [line for line in lines if line.split('|')[1] in names and int(line.split('|')[0]) not in allowed]

def preserved_data():
    old = json.loads((PREVIOUS_STAGE / 'submission_state.json').read_text())
    assert old['evaluation_job'] == PREVIOUS and old['registration_complete'] is True
    assert old['retained_steps'] == [8950] and old['failed_job'] == 181580
    assert old['output_root'] == str(OUT)
    roots = [Path(x) for x in old['checkpoint_roots']]
    resume = json.loads((roots[1] / 'resume_record.json').read_text())
    assert resume['old_job'] == 178786 and resume['step'] == 14500
    assert resume['environment']['G5_LOOP_PASSES'] == '3'
    assert resume['source_checkpoint'] == str(roots[0] / 'step-14500.ckpt')
    assert resume['all_checkpoint_states_verified_equal'] is True
    policy = json.loads(POLICY.read_text())
    a = SimpleNamespace(checkpoint_root=roots[0], resume_checkpoint_root=roots[1],
                        output_root=OUT, claims_root=OUT / '.claims-v1', retained_steps=[8950],
                        expected_dataset_names={n for s in policy['shards'] for n in s})
    old_env = os.environ.get('SLURM_JOB_ID')
    os.environ['SLURM_JOB_ID'] = 'resize-preflight'
    try:
        assert scheduler.completed_step(a, 8950)
    finally:
        if old_env is None: os.environ.pop('SLURM_JOB_ID', None)
        else: os.environ['SLURM_JOB_ID'] = old_env
    claims = {int(p.stem.split('-')[1]): json.loads(p.read_text()) for p in a.claims_root.glob('step-*.json')}
    assert sorted(claims) == [8950], 'Pending job unexpectedly produced claims; stop resize'
    assert str(claims[8950]['job_id']) == '181580'
    assert sorted(p.name for p in OUT.glob('step-*')) == ['step-8950']
    files = list((OUT / 'step-8950').iterdir()) + list(a.claims_root.glob('*.json'))
    for key, info in old['reusable_shards'].items():
        step, index = map(int, key.split('/'))
        checkpoint, _ = scheduler.checkpoint_for_step(a, step)
        assert validate_shard(Path(info['path']), checkpoint, step, index, policy) == info['signatures']
        files.extend(Path(p) for p in info['signatures'])
    return old, signatures(set(files))

def source_check():
    assert Path(__file__).resolve().parent == STAGE
    for line in (STAGE / 'source.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256((STAGE / name).read_bytes()).hexdigest() == digest, name
    for file in STAGE.glob('*.py'): compile(file.read_text(), str(file), 'exec')
    for file in STAGE.glob('*.sh'): run(['bash', '-n', str(file)])
    for name in ['gpu_shard_worker_deterministic.py']:
        assert (STAGE / name).read_bytes() == (PREVIOUS_STAGE / name).read_bytes()
    return run(['git', '-C', str(STAGE.parent), 'rev-parse', 'HEAD'])

def verify_new(job):
    f, raw = job_control(job)
    expected = dict(JobName=NAME, Account='faculty-acc', QOS='bgqos', Partition='faculty',
                    NumTasks='32', NumCPUs='128', **{'CPUs/Task': '4'},
                    TimeLimit='3-00:00:00', Nice='0', Requeue='0', Dependency='(null)',
                    Command=str(STAGE / 'slurm.sh'), WorkDir='/vast/users/guangyi.chen',
                    StdOut=str(ROOT / 'logs' / TAG / f'slurm-{job}.out'),
                    StdErr=str(ROOT / 'logs' / TAG / f'slurm-{job}.err'))
    for k, v in expected.items(): assert f[k] == v, (k, f.get(k), v)
    assert f['NumNodes'] in ('4', '4-4')
    assert f['NtasksPerN:B:S:C'].split(':')[0] == '8'
    assert f['MinMemoryNode'] == '120G'
    assert 'gres/gpu=32' in f['ReqTRES'].split(',')
    assert set(f['TresPerTask'].split(',')) == {'cpu=4', 'gres/gpu=1'}
    assert f['TresBind'] == 'gres/gpu:per_task:1'
    assert f['JobState'] == 'PENDING' and f['Reason'] == 'JobHeldUser'
    spooled = run(['scontrol', 'write', 'batch_script', str(job), '-'])
    assert spooled[spooled.index('#!/usr/bin/env bash'):].strip() == (STAGE / 'slurm.sh').read_text().strip(), 'Spooled script mismatch'
    return raw

def main(mode):
    receipt = STAGE / 'submission_state.json'
    if mode == 'prepare':
        assert not receipt.exists() and not (STAGE / 'resize_intent.json').exists()
        commit = source_check()
        f, raw = job_control(PREVIOUS)
        assert f['JobState'] == 'PENDING' and f['RunTime'] == '00:00:00'
        assert f['Command'] == str(PREVIOUS_STAGE / 'slurm.sh')
        assert not queue_conflicts({PREVIOUS})
        old, sig = preserved_data()
        (ROOT / 'logs' / TAG).mkdir(parents=True, exist_ok=True)
        tests = run(['python3', '-m', 'unittest', 'test_resume', '-v'])
        dry = subprocess.run(['sbatch', '--test-only', str(STAGE / 'slurm.sh')], text=True, capture_output=True)
        assert dry.returncode == 0, dry.stdout + dry.stderr
        audit = dict(observed_epoch=time.time(), previous_job=PREVIOUS, previous_control=raw,
                     signatures=sig, source_commit=commit, resources=RESOURCES,
                     tests=11, test_only=dry.stdout + dry.stderr)
        scheduler.atomic_json(STAGE / 'resize_prepare.json', audit)
        print(json.dumps(audit), flush=True)
    elif mode == 'submit-held':
        assert not receipt.exists() and not (STAGE / 'resize_intent.json').exists()
        prep = json.loads((STAGE / 'resize_prepare.json').read_text())
        assert 12 <= time.time() - prep['observed_epoch'] < 3600
        assert source_check() == prep['source_commit']
        f, _ = job_control(PREVIOUS)
        assert f['JobState'] == 'PENDING' and f['RunTime'] == '00:00:00'
        run(['scontrol', 'hold', str(PREVIOUS)])
        f, _ = job_control(PREVIOUS)
        assert f['JobState'] == 'PENDING' and f['Reason'] == 'JobHeldUser' and f['RunTime'] == '00:00:00'
        old, sig = preserved_data()
        assert sig == prep['signatures'] and not queue_conflicts({PREVIOUS})
        with (STAGE / 'resize_intent.json').open('x') as handle:
            json.dump(dict(previous_job=PREVIOUS, epoch=time.time(), job_name=NAME), handle)
        p = subprocess.run(['sbatch', '--parsable', '--hold', str(STAGE / 'slurm.sh')], capture_output=True, text=True)
        if p.returncode:
            scheduler.atomic_json(STAGE / 'resize_submit_failure.json', dict(stdout=p.stdout, stderr=p.stderr, rc=p.returncode))
            run(['scontrol', 'release', str(PREVIOUS)])
            raise RuntimeError(p.stdout + p.stderr)
        job = p.stdout.strip().split(';')[0]
        assert job.isdigit(), p.stdout
        old.update(evaluation_job=int(job), previous_evaluation_job=PREVIOUS, stage=str(STAGE),
                   job_name=NAME, resources=RESOURCES, submitted_epoch=time.time(),
                   source_commit=prep['source_commit'], registration_complete=True,
                   previous_never_started=True, resize_signatures=sig, resize_phase='held')
        scheduler.atomic_json(receipt, old)
        old['held_control'] = verify_new(job)
        scheduler.atomic_json(receipt, old)
        print(json.dumps(old), flush=True)
    elif mode == 'cutover':
        audit = json.loads(receipt.read_text())
        assert audit['resize_phase'] == 'held'
        job = audit['evaluation_job']
        verify_new(job)
        f, _ = job_control(PREVIOUS)
        assert f['JobState'] == 'PENDING' and f['Reason'] == 'JobHeldUser' and f['RunTime'] == '00:00:00'
        assert not queue_conflicts({PREVIOUS, job})
        with (OUT / '.fp32-online-worksteal.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _, sig = preserved_data()
            assert sig == audit['resize_signatures']
            audit['resize_phase'] = 'before_cancel'; scheduler.atomic_json(receipt, audit)
            run(['scancel', str(PREVIOUS)])
            f, raw = job_control(PREVIOUS)
            assert f['JobState'] == 'CANCELLED' and f['RunTime'] == '00:00:00', raw
            audit['cancelled_previous_control'] = raw
            assert preserved_data()[1] == sig
            audit['resize_phase'] = 'previous_cancelled'; scheduler.atomic_json(receipt, audit)
        run(['scontrol', 'release', str(job)])
        audit['resize_phase'] = 'released'
        audit['released_epoch'] = time.time()
        audit['current_control'] = job_control(job)[1]
        scheduler.atomic_json(receipt, audit)
        print(json.dumps(audit), flush=True)
    else:
        audit = json.loads(receipt.read_text())
        audit['fresh_control'] = job_control(audit['evaluation_job'])[1]
        print(json.dumps(audit), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'submit-held', 'cutover', 'status'])
    main(parser.parse_args().mode)
