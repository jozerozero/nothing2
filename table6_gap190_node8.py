#!/usr/bin/env python3
"""Independent full-node UUID-bound launch for the frozen missing190 campaign.

Prepare/verify only: submission and replacement of any older allocation belong
to the external controller. No old source, plan, study, lock or result changes.
Eight real Slurm ranks run native smoke/preflight/gate/worker subprocesses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

import table6_missing190_ops as original
from table6_restart_ag import publish
from table6_restart_deadline import EnvironmentBudget

ROOT, REPO, PY, BASE = original.ROOT, original.REPO, original.PY, original.BASE
NAME = 'table6_gap190_node8_uuid_20260922_v1'
STAGE = ROOT/'stage'/NAME
LOGS = ROOT/'logs'/NAME
PLAN_ID = 'f35686cd72710801e14ccf00e1d44f108d87627ee404927a178efc9c32c86066'
FILES = tuple(sorted(set(original.FILES) | {
    'table6_gap190_node8.py', 'allocated_gpu_uuid.py', 'table6_restart_ag.py',
    'pfn_mitra_one.py', 'eval_one.py', 'table6_existing_gap190.py'}))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {str(REPO/name): sha(REPO/name) for name in FILES} | {
        str(BASE/'rocm_gpu_entry.py'): sha(BASE/'rocm_gpu_entry.py')}


def script():
    return f'''#!/usr/bin/env bash
#SBATCH --job-name=t6gap8uuid
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gpus=8
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@90
#SBATCH --nice=0
#SBATCH --no-requeue
#SBATCH --distribution=block:block
#SBATCH --exclude={original.EXCLUDE}
#SBATCH --chdir={STAGE}
#SBATCH --output={LOGS}/slurm-%j.out
#SBATCH --error={LOGS}/slurm-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export T6_BASE_STAGE={BASE} T6_MISSING190_PLAN={original.PLAN}
export PYTHONPATH={BASE}:{BASE}/TALENT:{ROOT}/stage/table6_nonfoundation_all_benchmarks_20260823_v1/python_packages
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
{PY} -B {REPO}/table6_gap190_node8.py verify
{PY} -B {BASE}/manage.py check-inputs
RUNTIME_DIR={STAGE}/runtime/job-$SLURM_JOB_ID
mkdir -p "$RUNTIME_DIR"
MAPPING="$RUNTIME_DIR/mapping.json"
srun --exact --exclusive --input=none --nodes=1 --ntasks=1 --ntasks-per-node=1 \\
 --cpus-per-task=128 --gpus=8 --mem=512G --cpu-bind=cores --gpu-bind=none --kill-on-bad-exit=1 \\
 {PY} -B {REPO}/allocated_gpu_uuid.py bootstrap --mapping "$MAPPING" \\
 --expected-cpus 128 --expected-mem-gib 512 --expected-seconds 7200
for PHASE in smoke preflight gate worker; do
 srun --exact --exclusive --input=none --nodes=1 --ntasks=8 --ntasks-per-node=8 \\
  --cpus-per-task=16 --gpus=8 --mem=512G --cpu-bind=cores --gpu-bind=none --kill-on-bad-exit=1 \\
  {PY} -B {REPO}/allocated_gpu_uuid.py exec --mapping "$MAPPING" -- \\
  {PY} -B {BASE}/rocm_gpu_entry.py {PY} -B {REPO}/table6_gap190_node8.py rank \\
  --mapping "$MAPPING" --phase "$PHASE"
done
'''


def prepare():
    require(not STAGE.exists(), 'independent runtime stage already exists; never overwrite')
    original.verify()
    scientific = original.plan()
    require(scientific['plan_id'] == PLAN_ID, 'wrong frozen missing190 scientific plan')
    pinned = sources()  # Fail before writing if a required source is missing.
    body = script()
    checked = subprocess.run(['bash', '-n'], input=body, capture_output=True, text=True)
    require(checked.returncode == 0, 'generated Slurm shell syntax invalid: '+checked.stderr)
    STAGE.mkdir(parents=True, exist_ok=False)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (STAGE/'run.sh').open('x') as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    deployment = {'schema': 'table6_gap190_node8_uuid_v1', 'stage': str(STAGE),
        'source_hashes': pinned, 'script_sha256': sha(STAGE/'run.sh'),
        'scientific_plan': {'path': str(original.PLAN), 'sha256': sha(original.PLAN), 'plan_id': PLAN_ID},
        'resources': {'nodes':1, 'gpus':8, 'cpus':128, 'cpus_per_rank':16, 'mem_gib':512,
                      'seconds':7200, 'dependency': None},
        'scope': 'same missing190 native nine smokes/HPO100/15seeds, original pair flocks/results',
        'created_epoch': time.time()}
    publish(STAGE/'deployment.json', deployment)
    return deployment


def verify():
    require(not STAGE.is_symlink() and not (STAGE/'deployment.json').is_symlink(), 'runtime stage symlink refused')
    deployment = json.loads((STAGE/'deployment.json').read_text())
    require(deployment['schema'] == 'table6_gap190_node8_uuid_v1' and deployment['stage'] == str(STAGE),
            'wrong independent deployment')
    require(deployment['source_hashes'] == sources(), 'new or frozen runtime source changed')
    require((STAGE/'run.sh').read_text() == script() and sha(STAGE/'run.sh') == deployment['script_sha256'],
            'prepared Slurm script changed')
    require(deployment['scientific_plan'] == {'path':str(original.PLAN), 'sha256':sha(original.PLAN), 'plan_id':PLAN_ID},
            'scientific plan file changed')
    original.verify()  # Old deployment/run.sh remain unchanged even for a new job.
    require(original.plan()['plan_id'] == PLAN_ID, 'wrong frozen scientific plan')
    return deployment


def methods_for_rank(methods, rank):
    require(type(rank) is int and 0 <= rank < 8 and len(methods) == 9, 'invalid native smoke roster/rank')
    return [method for index, method in enumerate(methods) if index % 8 == rank]


def rank_contract(mapping):
    import allocated_gpu_uuid as binding
    expected = binding.rank_environment(mapping, os.environ)
    require(mapping['expected_cpus'] == 128 and mapping['expected_mem_gib'] == 512 and
            mapping['expected_seconds'] == 7200, 'wrong full-node resource budget')
    for key in ('ROCR_VISIBLE_DEVICES', 'EXPECTED_GPU_UUID', 'EXPECTED_GPU_PCI_BUS_ID',
                'ALLOCATED_GPU_MAPPING_ID', 'JOB_BUDGET_END_MONOTONIC', 'JOB_BUDGET_MONOTONIC_HOST',
                'JOB_BUDGET_JOB_ID', 'JOB_BUDGET_END_EPOCH'):
        require(os.environ.get(key) == expected[key], 'bound rank environment mismatch: '+key)
    require(all(key not in os.environ for key in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')),
            'GPU alias double masking detected')
    require(len(os.sched_getaffinity(0)) == 16, 'exactly16 bound CPU cores required')
    budget = EnvironmentBudget.from_environment()
    require(budget.monotonic_end is not None and budget.remaining() <= 6901, 'missing/excessive monotonic budget')
    return int(os.environ['SLURM_PROCID']), budget


class Paused(RuntimeError):
    pass


def run_worker(arguments, budget):
    """Fresh native process, signals forwarded; no model/request monkeypatch."""
    if budget.remaining() <= 180:
        raise Paused('no new native action window')
    from table6_existing_gap190 import stop_owned
    command = [PY, '-B', str(REPO/'table6_missing190_worker.py'), *arguments,
               '--plan', str(original.PLAN)]
    stopped = []
    child = None
    def stop(signum, _frame):
        stopped.append(signum)
        if child is not None and child.poll() is None:
            try: child.send_signal(signum)
            except ProcessLookupError: pass
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    try:
        child = subprocess.Popen(command, start_new_session=True)
        while child.poll() is None:
            if stopped or budget.remaining() <= 90:
                stop_owned(child)
                raise Paused('native action stopped for signal/allocation budget')
            try: child.wait(timeout=1)
            except subprocess.TimeoutExpired: pass
        require(child.wait() == 0, 'native action failed: '+str(arguments))
        if stopped or budget.remaining() <= 180:
            raise Paused('native action reached allocation shutdown margin')
    finally:
        if child is not None and child.poll() is None:
            stop_owned(child)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def validate_native_preflight(mapping, rank):
    record = json.loads((original.OUT/'preflight'/mapping['job']/f'gpu-{rank}.json').read_text())
    expected = mapping['gpus'][rank]
    domain, bus, slot = record['physical_gpu'].lower().split(':')
    pci = f'{int(domain,16):04x}:{bus}:{slot}'
    require(record['passed'] is True and record['job_id'] == mapping['job'] and record['rank'] == rank and
            record['host'] == mapping['node'] and record['plan_id'] == PLAN_ID and record['mode'] == 'gpu' and
            record['devices'] == 1 and pci == expected['pci'] and
            set(record['cpu_affinity']) == set(os.sched_getaffinity(0)), 'native preflight differs from UUID/current CPU assignment')
    return record


def rank_action(mapping_path, phase):
    deployment = verify()
    import allocated_gpu_uuid as binding
    mapping = binding.load_mapping(mapping_path)
    require(Path(mapping_path) == STAGE/'runtime'/('job-'+mapping['job'])/'mapping.json', 'mapping outside this deployment/job')
    rank, budget = rank_contract(mapping)
    require(budget.remaining() > 180, 'insufficient remaining native action window')
    os.environ.pop('CPU_ONLY', None)
    private = tempfile.mkdtemp(prefix='t6g8-', dir='/tmp')
    os.environ['TMPDIR'] = private; tempfile.tempdir = None
    identity = None
    if phase != 'gate':
        import torch
        from pfn_mitra_one import gpu_identity
        identity = gpu_identity(torch)  # Loaded HIP, actual PCI/UUID and arithmetic.
    receipt = {'phase':phase, 'job':mapping['job'], 'node':socket.gethostname(),
               'rank':rank, 'step':os.environ['SLURM_STEP_ID'], 'mapping_id':mapping['mapping_id'],
               'source_hashes':deployment['source_hashes'], 'scientific_plan_id':PLAN_ID,
               'physical_gpu':identity, 'cpu_affinity':sorted(os.sched_getaffinity(0)), 'TMPDIR':private}
    audit = STAGE/'runtime'/('job-'+mapping['job'])
    publish(audit/f'{phase}-rank-{rank}-start.json', receipt)
    if phase == 'smoke':
        from table6_missing190_fit import METHODS
        for method in methods_for_rank(tuple(METHODS), rank):
            run_worker(['smoke','--method',method], budget)
    elif phase == 'preflight':
        run_worker(['preflight','--mode','gpu'], budget)
        validate_native_preflight(mapping, rank)
    elif phase == 'gate':
        if rank == 0:
            run_worker(['gate','--ranks','8'], budget)  # Validates all nine response bytes and source pins.
    elif phase == 'worker':
        validate_native_preflight(mapping, rank)
        # Native worker revalidates the complete nine-model gate before taking a
        # pair flock. Exec drops this probe's Torch context before model fitting.
        publish(audit/f'{phase}-rank-{rank}-handoff.json', receipt)
        os.execv(PY, [PY,'-B',str(REPO/'table6_missing190_worker.py'),'worker',
                      '--plan',str(original.PLAN),'--mode','gpu'])
    else:
        raise ValueError(phase)
    publish(audit/f'{phase}-rank-{rank}-complete.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare','verify','rank'))
    parser.add_argument('--mapping', type=Path)
    parser.add_argument('--phase', choices=('smoke','preflight','gate','worker'))
    args = parser.parse_args(argv)
    require(__debug__, 'native scientific assertions must remain enabled')
    try:
        if args.mode == 'prepare': result = prepare()
        elif args.mode == 'verify': result = verify()
        else:
            require(args.mapping is not None and args.phase is not None, 'rank requires mapping and phase')
            result = rank_action(args.mapping, args.phase)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    except Paused as exc:
        print(json.dumps({'state':'paused','reason':str(exc)}), flush=True)
        raise SystemExit(75)


if __name__ == '__main__':
    main()
