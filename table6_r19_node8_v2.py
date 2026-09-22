#!/usr/bin/env python3
"""New operational full-node UUID wrapper for the unchanged R19 short worker.

Only prepare/verify and genuine-rank handoff are provided. Submission, terminal
job checks and held-first release belong to the external controller. Scientific
plans, original preflight/barriers/model smokes, locks, HPO studies, final seeds,
existing errors/deferred pairs and success publication are never changed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

import table6_restart_ops as original
import table6_restart_gpu as restart
from table6_restart_ag import publish
from table6_restart_deadline import EnvironmentBudget

ROOT, REPO, PY, BASE = original.ROOT, original.REPO, original.PY, original.BASE
NAME = 'table6_r19_node8_uuid_20260922_v2'
STAGE, LOGS = ROOT/'stage'/NAME, ROOT/'logs'/NAME
PLAN = BASE/'plan.json'
PLAN_ID = original.PLAN
SCHEMA = 'table6_r19_node8_uuid_v2'
FILES = tuple(sorted(set(original.FILES) | {
    'table6_r19_node8_v2.py', 'allocated_gpu_uuid.py', 'table6_restart_ag.py',
    'pfn_mitra_one.py', 'eval_one.py'}))
NATIVE_FILES = ('common.py', 'worker.py', 'fit.py', 'data.py', 'manage.py',
                'rocm_gpu_entry.py', 'deployment.json')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'missing/symlink source: '+str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    value = {str(REPO/name): sha(REPO/name) for name in FILES}
    value.update({str(BASE/name): sha(BASE/name) for name in NATIVE_FILES})
    value[str(restart.SOURCE)] = sha(restart.SOURCE)
    require(value[str(restart.SOURCE)] == restart.EXPECTED, 'frozen short worker changed')
    return value


def scientific_plan():
    require(PLAN.is_file() and not PLAN.is_symlink(), 'missing/symlink scientific plan')
    plan = json.loads(PLAN.read_text())
    require(plan['plan_id'] == PLAN_ID and plan['hpo_trials'] == 100 and
            plan['seeds'] == list(range(15)) and len(plan['pairs']) == 12482 and
            plan['classification_memberships'] == 457 and plan['regression_memberships'] == 224,
            'frozen R19 scientific contract changed')
    return {'path': str(PLAN), 'sha256': sha(PLAN), 'plan_id': PLAN_ID}


def script():
    return f'''#!/usr/bin/env bash
#SBATCH --job-name=t6r19u8v2
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
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
export T6_BASE_STAGE={BASE}
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH={BASE}:{BASE}/TALENT:{ROOT}/stage/table6_nonfoundation_all_benchmarks_20260823_v1/python_packages
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
{PY} -B {REPO}/table6_r19_node8_v2.py verify
{PY} -B {BASE}/manage.py check-inputs
RUNTIME_DIR={STAGE}/runtime/job-$SLURM_JOB_ID
mkdir -p "$RUNTIME_DIR"
MAPPING="$RUNTIME_DIR/mapping.json"
srun --exact --exclusive --input=none --nodes=1 --ntasks=1 --ntasks-per-node=1 \\
 --cpus-per-task=64 --gpus=8 --mem=512G --cpu-bind=cores --gpu-bind=none --kill-on-bad-exit=1 \\
 {PY} -B {REPO}/allocated_gpu_uuid.py bootstrap --mapping "$MAPPING" \\
 --expected-cpus 64 --expected-mem-gib 512 --expected-seconds 7200
srun --exact --exclusive --input=none --nodes=1 --ntasks=8 --ntasks-per-node=8 \\
 --cpus-per-task=8 --gpus=8 --mem=512G --cpu-bind=cores --gpu-bind=none --kill-on-bad-exit=1 \\
 {PY} -B {REPO}/allocated_gpu_uuid.py exec --mapping "$MAPPING" -- \\
 {PY} -B {BASE}/rocm_gpu_entry.py {PY} -B {REPO}/table6_r19_node8_v2.py rank --mapping "$MAPPING"
'''


def prepare():
    require(not STAGE.exists() and not STAGE.is_symlink(), 'independent runtime stage already exists; never overwrite')
    original.verify()
    plan, pinned, body = scientific_plan(), sources(), script()
    syntax = subprocess.run(['bash', '-n'], input=body, text=True, capture_output=True)
    require(syntax.returncode == 0, 'generated shell syntax invalid: '+syntax.stderr)
    STAGE.mkdir(parents=True, exist_ok=False)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (STAGE/'run.sh').open('x') as handle:
        handle.write(body); handle.flush(); os.fsync(handle.fileno())
    record = {'schema': SCHEMA, 'stage': str(STAGE), 'source_hashes': pinned,
              'script_sha256': sha(STAGE/'run.sh'), 'scientific_plan': plan,
              'resources': {'nodes': 1, 'gpus': 8, 'cpus': 64, 'cpus_per_rank': 8,
                            'mem_gib': 512, 'seconds': 7200, 'dependency': None},
              'scope': 'unchanged short worker launch:8 native preflights/barriers/32 model smokes; original HPO100/15seeds/flocks/results',
              'created_epoch': time.time()}
    publish(STAGE/'deployment.json', record)
    return record


def verify():
    require(not STAGE.is_symlink() and not (STAGE/'deployment.json').is_symlink(), 'runtime stage symlink refused')
    record = json.loads((STAGE/'deployment.json').read_text())
    require(record['schema'] == SCHEMA and record['stage'] == str(STAGE), 'wrong runtime deployment')
    require(record['source_hashes'] == sources(), 'new or frozen runtime source changed')
    require((STAGE/'run.sh').read_text() == script() and record['script_sha256'] == sha(STAGE/'run.sh'),
            'prepared Slurm script changed')
    require(record['scientific_plan'] == scientific_plan(), 'scientific plan changed')
    original.verify()
    return record


def rank_contract(mapping):
    import allocated_gpu_uuid as binding
    expected = binding.rank_environment(mapping, os.environ)
    require(mapping['expected_cpus'] == 64 and mapping['expected_mem_gib'] == 512 and
            mapping['expected_seconds'] == 7200, 'wrong full-node resource budget')
    for key in ('ROCR_VISIBLE_DEVICES', 'EXPECTED_GPU_UUID', 'EXPECTED_GPU_PCI_BUS_ID',
                'ALLOCATED_GPU_MAPPING_ID', 'JOB_BUDGET_END_MONOTONIC', 'JOB_BUDGET_MONOTONIC_HOST',
                'JOB_BUDGET_JOB_ID', 'JOB_BUDGET_END_EPOCH'):
        require(os.environ.get(key) == expected[key], 'bound rank environment mismatch: '+key)
    require(all(key not in os.environ for key in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')),
            'GPU alias double masking detected')
    require(os.environ.get('SLURM_CPUS_PER_TASK') == '8' and len(os.sched_getaffinity(0)) == 8,
            'exactly8 Slurm-bound CPU cores required')
    budget = EnvironmentBudget.from_environment()
    require(budget.monotonic_end is not None and 180 < budget.remaining() <= 6901,
            'missing/exhausted/excessive shared monotonic budget')
    return int(os.environ['SLURM_PROCID'])


def native_command():
    # Do not bypass the original 8-rank preflight and actual-model-smoke barriers.
    return [PY, '-B', str(REPO/'table6_restart_gpu.py'), 'launch', '--mode', 'gpu']


def rank_action(mapping_path):
    deployment = verify()
    import allocated_gpu_uuid as binding
    mapping = binding.load_mapping(mapping_path)
    require(Path(mapping_path) == STAGE/'runtime'/('job-'+mapping['job'])/'mapping.json',
            'mapping outside this deployment/job')
    rank = rank_contract(mapping)
    require(Path(os.environ.get('T6_BASE_STAGE', '')).resolve() == BASE, 'wrong scientific stage')
    os.environ.pop('CPU_ONLY', None)
    os.environ['TMPDIR'] = tempfile.mkdtemp(prefix='t6r19u8-', dir='/tmp')
    tempfile.tempdir = None
    import torch
    from pfn_mitra_one import gpu_identity
    identity = gpu_identity(torch)  # Real tensor arithmetic plus UUID/PCI, not ordinal inference.
    audit = STAGE/'runtime'/('job-'+mapping['job'])
    publish(audit/f'uuid-rank-{rank}-handoff.json', {
        'job': mapping['job'], 'step': os.environ['SLURM_STEP_ID'], 'rank': rank,
        'node': socket.gethostname(), 'mapping_id': mapping['mapping_id'],
        'scientific_plan_id': PLAN_ID, 'physical_gpu': identity,
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'source_hashes': deployment['source_hashes'],
        'command': native_command(), 'native_startup_gates_not_skipped': True})
    # Replace probe process, dropping its Torch context; the unchanged worker
    # reruns native arithmetic/preflight, all8 barrier and all32 model smokes.
    os.execv(PY, native_command())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'verify', 'rank'))
    parser.add_argument('--mapping', type=Path)
    args = parser.parse_args(argv)
    require(__debug__, 'native scientific assertions must remain enabled')
    if args.mode == 'prepare':
        result = prepare()
    elif args.mode == 'verify':
        result = verify()
    else:
        require(args.mapping is not None, 'rank requires immutable mapping')
        result = rank_action(args.mapping)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
