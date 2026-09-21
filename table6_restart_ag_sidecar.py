"""Proof-authorized, CPU-only AG sidecar inside one existing Slurm allocation.

No allocation submission, parent mutation, discovery, retry, or --overlap.
One immutable parent claim prevents duplicate/uncertain launch attempts. The
frozen four-rank worker and original evaluation output contracts are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from table6_restart_deadline import derive_deadline, parse_duration, parse_fields
from table6_restart_ag import ROOT, BASE, PLAN, CPU_ENV, publish, require, read

REPO = Path(__file__).resolve().parent
STAGE = ROOT / 'stage/table6_autogluon_sidecar_20260922_v2'
PYTHON = '/vast/users/guangyi.chen/anaconda3/envs/tabicl/bin/python'
WORKER_SHA = '3ca3bf4d620b026f29409a9361630f0110227c116a68317a975948ccb79ecfaf'
FILES = ('table6_restart_ag_sidecar.py', 'table6_restart_ag.py', 'table6_restart_deadline.py')
MEMORY = 512 * 1024 ** 3
PROOF_MAX_AGE = 300


def source_hashes():
    hashes = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in FILES}
    require(hashes['table6_restart_ag.py'] == WORKER_SHA, 'frozen AG worker changed')
    return hashes


def validate_proof(proof, parent, node, now=None):
    now = time.time() if now is None else now
    require(proof.get('allow_cpu_sidecar') is True and str(proof.get('parent_job_id')) == parent
            and proof.get('node') == node, 'proof does not authorize this exact parent/node')
    observations = proof.get('observations', [])
    require(len(observations) >= 2, 'two capacity observations required')
    times = []
    for sample in observations:
        observed = float(sample['observed_epoch'])
        require(math.isfinite(observed) and 0 <= now - observed <= PROOF_MAX_AGE, 'stale/future capacity proof')
        require(int(sample['available_cpus']) >= 64 and int(sample['available_memory_bytes']) >= MEMORY,
                'insufficient observed CPU/RAM capacity')
        busy = float(sample['busy_cpus'])
        require(math.isfinite(busy) and 0 <= busy < 64, 'batch CPU workload exceeds permitted capacity')
        times.append(observed)
    require(times == sorted(set(times)) and times[-1] - times[0] >= 1, 'observations must be distinct and separated')
    return proof


def clean_environment(environ=None):
    source = dict(os.environ if environ is None else environ)
    prefixes = ('SLURM_', 'SBATCH_', 'SRUN_', 'JOB_BUDGET_', 'T6_AG_', 'PMI_', 'PMIX_', 'OMPI_')
    env = {k: v for k, v in source.items() if not k.startswith(prefixes)}
    fastai = ROOT / 'stage/limix2m_table6_deep_cpu_augmented_fastai_20260827_v2/fastai_overlay'
    packages = ROOT / 'stage/table6_nonfoundation_all_benchmarks_20260823_v1/python_packages'
    env.update(CPU_ENV)
    env.update(PYTHONPATH=f'{BASE}:{BASE}/runtime:{fastai}:{packages}',
               PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               GPU_DEVICE_ORDINAL='-1', OMP_NUM_THREADS='16', MKL_NUM_THREADS='16',
               OPENBLAS_NUM_THREADS='16', NUMEXPR_NUM_THREADS='16')
    return env


def command(args, *, env=None):
    result = subprocess.run(args, capture_output=True, text=True, timeout=20, env=env)
    require(result.returncode == 0, f'command failed: {args[0]}: {result.stderr}')
    return result.stdout.strip()


def memory_bytes(value):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)', value)
    require(match is not None, 'unrecognized Slurm memory unit')
    return int(float(match.group(1)) * 1024 ** {'': 2, 'K': 1, 'M': 2, 'G': 3, 'T': 4}[match.group(2)])


def validate_parent(raw, parent, node, uid=None):
    fields = parse_fields(raw)
    require(fields.get('JobId') == parent and fields.get('JobState') == 'RUNNING', 'parent is not exact running allocation')
    require(fields.get('NumNodes') == '1' and fields.get('NodeList') == node, 'parent must be exactly the authorized node')
    uid = os.getuid() if uid is None else uid
    require(re.fullmatch(r'[^()]+\(' + str(uid) + r'\)', fields.get('UserId', '')) is not None, 'parent is not owned by current user')
    allocated = dict(token.split('=', 1) for token in fields.get('AllocTRES', '').split(',') if '=' in token)
    # --mem=0 means all node memory, not a numeric capacity proof. In that
    # case insist on the actual allocated memory from Slurm rather than guess.
    memory = allocated.get('mem', fields.get('MinMemoryNode', '0'))
    require(int(fields['NumCPUs']) >= 64 and memory_bytes(memory) >= MEMORY,
            'parent allocation lacks 64 CPUs / 512 GiB')
    remaining = parse_duration(fields['TimeLimit']) - parse_duration(fields['RunTime'])
    require(remaining > 600, 'insufficient parent remaining time')
    return fields


def srun_command(parent, node, launch_dir):
    return ['srun', '--jobid=' + parent, '--nodelist=' + node, '--nodes=1',
            '--ntasks=4', '--ntasks-per-node=4', '--cpus-per-task=16', '--mem=512G',
            '--gpus=0', '--gpus-per-task=0', '--gres=none', '--exact', '--exclusive',
            '--immediate=10', '--time=02:00:00', '--cpu-bind=cores', '--kill-on-bad-exit=1',
            '--input=none', '--export=ALL',
            'env', 'CPU_ONLY=1', 'CUDA_VISIBLE_DEVICES=', 'HIP_VISIBLE_DEVICES=-1',
            'ROCR_VISIBLE_DEVICES=-1', 'GPU_DEVICE_ORDINAL=-1',
            PYTHON, '-B', str(Path(__file__).resolve()),
            'rank', '--launch-dir', str(launch_dir)]


def verify_plan(launch_dir):
    launch_dir = Path(launch_dir).resolve()
    require(launch_dir.parent == (STAGE / 'launches').resolve(), 'launch directory outside sidecar namespace')
    plan = read(launch_dir / 'plan.json')
    require(plan['plan_id'] == PLAN and plan['source_hashes'] == source_hashes(), 'sidecar plan/source changed')
    require(plan['command'] == srun_command(plan['parent'], plan['node'], launch_dir), 'srun command changed')
    return plan


def launch(parent, node, proof_path, launch_id):
    require(re.fullmatch(r'\d+', parent) is not None and re.fullmatch(r'[A-Za-z0-9_-]+', node), 'invalid exact target')
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,80}', launch_id) is not None, 'invalid launch id')
    proof_path = Path(proof_path)
    proof_bytes = proof_path.read_bytes()
    proof = validate_proof(json.loads(proof_bytes), parent, node)
    env = clean_environment()
    raw = command(['scontrol', 'show', 'job', '-o', parent], env=dict(env, TZ='UTC'))
    validate_parent(raw, parent, node)
    launch_dir = STAGE / 'launches' / launch_id
    plan = {'plan_id': PLAN, 'parent': parent, 'node': node, 'launch_id': launch_id,
            'created_epoch': time.time(), 'source_hashes': source_hashes(),
            'capacity_proof': proof, 'capacity_proof_path': str(proof_path.resolve()),
            'capacity_proof_sha256': hashlib.sha256(proof_bytes).hexdigest(),
            'parent_snapshot': raw, 'command': srun_command(parent, node, launch_dir),
            'resources': {'ranks': 4, 'cpus_per_rank': 16, 'memory_bytes': MEMORY, 'gpus': 0,
                          'step_time_limit_seconds': 7200, 'immediate_seconds': 10},
            'scope': 'existing allocation only; no batch-resource isolation claim; no parent changes'}
    # Exclusive parent intent is never removed automatically, including on an
    # uncertain spawn. A later attempt requires explicit review, not retry logic.
    publish(STAGE / 'parents' / (parent + '.json'), {'launch_id': launch_id, 'plan': plan})
    launch_dir.mkdir(parents=True, exist_ok=False)
    publish(launch_dir / 'plan.json', plan)
    with (launch_dir / 'supervisor.log').open('x') as log:
        process = subprocess.Popen([PYTHON, '-B', str(Path(__file__).resolve()), 'supervise',
                                    '--launch-dir', str(launch_dir)], stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True, close_fds=True)
    receipt = {'state': 'supervisor_started', 'pid': process.pid, 'launch_id': launch_id,
               'parent': parent, 'node': node, 'epoch': time.time(), 'not_yet_step_verified': True}
    publish(launch_dir / 'launch.json', receipt)
    return receipt


def supervise(launch_dir):
    launch_dir = Path(launch_dir)
    plan = verify_plan(launch_dir)
    env = clean_environment()
    validate_proof(plan['capacity_proof'], plan['parent'], plan['node'])
    raw = command(['scontrol', 'show', 'job', '-o', plan['parent']], env=dict(env, TZ='UTC'))
    validate_parent(raw, plan['parent'], plan['node'])
    # Claim the supervisor attempt before starting srun: no second invocation.
    publish(launch_dir / 'srun-intent.json', {'pid': os.getpid(), 'epoch': time.time(), 'command': plan['command']})
    with (launch_dir / 'srun.log').open('x') as log:
        process = subprocess.Popen(plan['command'], stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, env=env, close_fds=True)
        publish(launch_dir / 'srun.json', {'pid': process.pid, 'epoch': time.time(), 'parent': plan['parent']})
        rc = process.wait()
    publish(launch_dir / 'completion.json', {'returncode': rc, 'epoch': time.time(),
            'state': 'step_command_finished' if rc == 0 else 'step_command_failed',
            'scientific_completion_not_implied': True})
    return rc


def capped_budget(raw, parent, node, started, finished, wall, host):
    fields = validate_parent(raw, parent, node)
    original = derive_deadline(raw, job_id=parent, query_started_monotonic=started,
                               query_finished_monotonic=finished, observed_local_epoch=wall,
                               expected_limit_seconds=parse_duration(fields['TimeLimit']),
                               safety_margin_seconds=300, minimum_remaining_seconds=120, hostname=host)
    remaining = min(original.safe_remaining_seconds, 7200 - (finished - started) - 300)
    require(remaining > 180, 'no safe sidecar budget remains')
    return {'schema': 'table6_ag_sidecar_budget_v1', 'parent_derivation': original.record(),
            'cap_seconds': 7200, 'safety_margin_seconds': 300, 'remaining_seconds': remaining,
            'environment': {'JOB_BUDGET_END_EPOCH': format(wall + remaining, '.9f'),
                            'JOB_BUDGET_END_MONOTONIC': format(finished + remaining, '.9f'),
                            'JOB_BUDGET_MONOTONIC_HOST': host, 'JOB_BUDGET_JOB_ID': parent}}


def rank_entry(launch_dir):
    launch_dir = Path(launch_dir)
    plan = verify_plan(launch_dir)
    parent, node = plan['parent'], plan['node']
    require(os.environ.get('SLURM_JOB_ID') == parent and socket.gethostname() == node, 'wrong actual allocation/node')
    step = os.environ.get('SLURM_STEP_ID', '')
    require(re.fullmatch(r'\d+', step) is not None and os.environ.get('SLURM_NTASKS') == '4'
            and os.environ.get('SLURM_CPUS_PER_TASK') == '16', 'wrong actual numeric CPU step')
    require(all(os.environ.get(k) == v for k, v in CPU_ENV.items()), 'CPU environment lost')
    rank = int(os.environ['SLURM_PROCID'])
    require(rank in range(4) and len(os.sched_getaffinity(0)) == 16, 'invalid rank CPU binding')
    publish(launch_dir / f'rank-{rank}.json', {'rank': rank, 'job_id': parent, 'step_id': step,
            'host': node, 'pid': os.getpid(), 'cpu_affinity': sorted(os.sched_getaffinity(0))})
    budget_path = launch_dir / 'budget.json'
    if rank == 0:
        validate_proof(plan['capacity_proof'], parent, node)
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        require(available >= MEMORY, 'node no longer has 512 GiB available')
        env = dict(os.environ, TZ='UTC')
        env.pop('SLURM_TIME_FORMAT', None)
        started = time.monotonic()
        raw = command(['scontrol', 'show', 'job', '-o', parent], env=env)
        finished, wall = time.monotonic(), time.time()
        budget = capped_budget(raw, parent, node, started, finished, wall, node)
        budget['step_id'] = step
        budget['node_memory_available_bytes'] = available
        publish(budget_path, budget)
    deadline = time.monotonic() + 60
    while not budget_path.exists():
        require(time.monotonic() < deadline, 'rank0 budget unavailable; no fit permitted')
        time.sleep(.2)
    budget = read(budget_path)
    require(budget['step_id'] == step and budget['environment']['JOB_BUDGET_JOB_ID'] == parent
            and budget['environment']['JOB_BUDGET_MONOTONIC_HOST'] == node, 'budget belongs to another step/node')
    os.environ.update(budget['environment'])
    os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    os.execvpe('ionice', ['ionice', '-c', '3', PYTHON, '-B', str(REPO / 'table6_restart_ag.py'), 'launch'], os.environ)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'supervise', 'rank'))
    parser.add_argument('--parent')
    parser.add_argument('--node')
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--launch-id')
    parser.add_argument('--launch-dir', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'launch':
        print(json.dumps(launch(args.parent, args.node, args.proof, args.launch_id), allow_nan=False))
        return 0
    if args.action == 'supervise':
        return supervise(args.launch_dir)
    rank_entry(args.launch_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
