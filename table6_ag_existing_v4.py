"""Reviewed retry of failed v3 preflight: two32-CPU ranks, each fitting on16.

Only this new operational adapter is changed. Scientific plans, frozen fitter,
seed/result validation, claim locks/CAS, and v3 memory/cleanup guards are reused.
The real Slurm reservation remains32 in the environment; a module-local fitter
getenv adapter supplies the separately verified effective16 CPU count.
"""
from __future__ import annotations

import argparse
import datetime as dt
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

import table6_ag_existing_v3 as v3

ag, old = v3.ag, v3.old
require, read, publish = v3.require, v3.read, v3.publish
STAGE = ag.ROOT / 'stage/table6_autogluon_existing_20260922_v4'
CAMPAIGN = STAGE.name
V3_STAGE = v3.STAGE
V3_SHA = '109055f3213fc121c40a9ccb33da2ab3216f61bbfd9e8e7dbec484aab9f3a106'
PREVIOUS_STEPS = {'206116': '120', '206117': '190'}
RESERVED = 32
EFFECTIVE = 16
RANKS = 2
FILES = ('table6_ag_existing_v4.py', *v3.FILES)
ORIGINAL_PARENT = v3.validate_parent
ORIGINAL_GUARDED = v3.guarded_run_seed
ORIGINAL_LOAD = ag.load_original
ORIGINAL_EXACT = v3.exact_step_evidence


def sources():
    value = {name: hashlib.sha256((v3.REPO / name).read_bytes()).hexdigest() for name in FILES}
    require(value['table6_ag_existing_v3.py'] == V3_SHA, 'deployed v3 changed')
    require(value['table6_restart_ag.py'] == old.WORKER_SHA, 'frozen AG worker changed')
    return value


def validate_parent(raw, parent, node, ranks):
    require(ranks == RANKS and type(ranks) is int, 'reviewed v4 rollout permits exactly two ranks')
    fields = ORIGINAL_PARENT(raw, parent, node, ranks)
    require(int(fields['NumCPUs']) >= ranks * RESERVED, 'parent lacks reserved32CPU per rank')
    return fields


def select_cpus(reserved, samples):
    require(len(reserved) == RESERVED and len(set(reserved)) == RESERVED and
            all(type(c) is int and c >= 0 for c in reserved), 'real reserved affinity must contain32 distinct CPUs')
    require(len(samples) == 2 and all(isinstance(row, list) for row in samples), 'two CPU samples required')
    for cpu in reserved:
        require(all(cpu < len(row) and type(row[cpu]) in (int, float) and math.isfinite(row[cpu])
                    and 0 <= row[cpu] <= 100 for row in samples), 'CPU observation invalid/incomplete')
    idle = [cpu for cpu in reserved if all(row[cpu] < 25 for row in samples)]
    require(len(idle) >= EFFECTIVE, 'fewer than16 repeatedly idle CPUs inside real32CPU affinity')
    return sorted(sorted(idle, key=lambda cpu: (max(row[cpu] for row in samples),
                                               sum(row[cpu] for row in samples), cpu))[:EFFECTIVE])


def terminal_step(parent, step, *, run=subprocess.run, now=None):
    require(parent in PREVIOUS_STEPS and step == PREVIOUS_STEPS[parent], 'unreviewed previous exact step')
    identity = parent + '.' + step
    env = dict(os.environ, TZ='UTC'); env.pop('SLURM_TIME_FORMAT', None)
    def command(args):
        result = run(args, text=True, capture_output=True, timeout=20, env=env)
        require(result.returncode == 0, 'previous-step scheduler query failed: ' + result.stderr)
        return result.stdout
    rows = [line.split('|') for line in command(['sacct', '-n', '-P', '-j', identity,
                                                '-o', 'JobIDRaw,State%30,End']).splitlines()]
    rows = [row for row in rows if len(row) >= 3 and row[0] == identity]
    require(len(rows) == 1, 'missing/duplicate exact previous-step accounting')
    _, state, ended = rows[0][:3]
    require(state.split() and state.split()[0].rstrip('+') in ag.TERMINAL, 'previous step not terminal')
    end = dt.datetime.fromisoformat(ended)
    end = end.replace(tzinfo=dt.timezone.utc) if end.tzinfo is None else end
    now = time.time() if now is None else now
    require(now - end.timestamp() >= 120, 'previous-step termination grace not reached')
    queue = command(['squeue', '--steps', '-h', '-j', parent, '-o', '%i'])
    ids = [line.split('|', 1)[0].strip() for line in queue.splitlines() if line.strip()]
    require(all(re.fullmatch(re.escape(parent) + r'\.(?:\d+|batch|extern)', item) for item in ids),
            'unknown/truncated previous-step queue identity')
    require(identity not in ids, 'previous exact step still queued')
    return {'owner': identity, 'state': state, 'end': ended, 'checked_epoch': now,
            'age_seconds': now - end.timestamp(), 'exact_step_absent_from_queue': True}


def file_record(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'unsafe prior audit file')
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'value': json.loads(data)}


def previous_attempt(directory, parent, node, *, run=subprocess.run, now=None):
    directory = Path(directory).resolve()
    require(directory.parent == (V3_STAGE / 'launches').resolve(), 'not the immutable v3 launch namespace')
    records = [file_record(directory / name) for name in ('plan.json', 'completion.json', 'srun-intent.json')]
    plan, completion, _ = [row['value'] for row in records]
    require(plan['parent'] == parent and plan['node'] == node and plan['plan_id'] == ag.PLAN and
            plan['source_hashes']['table6_ag_existing_v3.py'] == V3_SHA, 'previous attempt identity/source differs')
    require(plan['operational_digest'] == ag.digest({k: v for k, v in plan.items() if k != 'operational_digest'}),
            'previous operational plan digest differs')
    require(type(completion.get('returncode')) is int and completion['returncode'] != 0,
            'only the reviewed failed preflight attempt may be retried')
    rank_records = [file_record(path) for path in sorted(directory.glob('rank-*.json'))]
    require(rank_records and all(row['value']['job_id'] == parent and row['value']['host'] == node and
            row['value']['step_id'] == PREVIOUS_STEPS[parent] for row in rank_records), 'previous rank/step evidence missing')
    proof = terminal_step(parent, PREVIOUS_STEPS[parent], run=run, now=now)
    matching = []
    require((ag.OUT / 'claims').is_dir() and not (ag.OUT / 'claims').is_symlink(), 'canonical claim directory unavailable')
    for path in (ag.OUT / 'claims').glob('*.json'):
        claim = read(path)
        if str(claim.get('job_id')) == parent and str(claim.get('step_id')) == PREVIOUS_STEPS[parent]:
            matching.append(str(path))
    require(not matching, 'previous failed step created canonical claims; separate recovery review required')
    audit = ag.OUT / 'auxiliary' / V3_STAGE.name / f"j{parent}-s{PREVIOUS_STEPS[parent]}"
    require(not list(audit.glob('new-owner-*.json')) and not list(audit.glob('fit-*.log')),
            'previous attempt entered fitting/claim ownership; separate review required')
    return {'directory': str(directory), 'terminal': proof, 'source_records': records + rank_records,
            'matching_canonical_claims': matching, 'no_new_owner_or_fit_audits': True,
            'checked_epoch': time.time() if now is None else now}


def srun_command(parent, node, ranks, directory):
    require(ranks == RANKS, 'only reviewed2rank rollout supported')
    command = v3_srun(parent, node, ranks, directory)
    command[command.index('--cpus-per-task=16')] = '--cpus-per-task=32'
    command[command.index('--cpu-bind=cores')] = '--cpu-bind=threads'
    command[command.index(str(v3.REPO/'table6_ag_existing_v3.py'))] = str(Path(__file__).resolve())
    return command


v3_srun = v3.srun_command


def verify_plan(directory):
    directory = Path(directory).resolve()
    require(directory.parent == (STAGE/'launches').resolve(), 'unexpected v4 namespace')
    plan = read(directory/'plan.json')
    require(plan['ranks'] == RANKS and plan['reserved_cpus_per_rank'] == RESERVED and
            plan['effective_cpus_per_rank'] == EFFECTIVE, 'v4 CPU contract changed')
    v3.target(plan['parent'], plan['node'], plan['ranks'])
    require(plan['plan_id'] == ag.PLAN and plan['source_hashes'] == sources(), 'v4 source/scientific plan changed')
    require(plan['operational_digest'] == ag.digest({k: v for k, v in plan.items() if k != 'operational_digest'}),
            'v4 operational plan changed')
    require(plan['command'] == srun_command(plan['parent'], plan['node'], plan['ranks'], directory), 'v4 srun changed')
    return plan


def launch(parent, node, proof_path, launch_id, previous):
    v3.target(parent, node, RANKS)
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,80}', launch_id or ''), 'invalid unique launch id')
    prior = previous_attempt(previous, parent, node)
    proof_file = file_record(proof_path)
    proof = v3.validate_capacity(proof_file['value'], parent, node, RANKS)
    raw = old.command(['scontrol', 'show', 'job', '-o', parent], env=dict(v3.clean_environment(), TZ='UTC'))
    fields = validate_parent(raw, parent, node, RANKS)
    allocated = dict(t.split('=', 1) for t in fields.get('AllocTRES', '').split(',') if '=' in t)
    memory = old.memory_bytes(allocated.get('mem', fields.get('MinMemoryNode', '0')))
    require(all(row['parent_memory_current_bytes'] + RANKS*v3.RANK_MEMORY + v3.PARENT_MARGIN <= memory
                and len(row['parent_cpu_ids']) >= RANKS*RESERVED for row in proof['observations']),
            'capacity exceeds whole-parent Slurm memory/CPU reservation')
    directory = STAGE/'launches'/launch_id
    plan = {'plan_id': ag.PLAN, 'parent': parent, 'node': node, 'ranks': RANKS,
            'reserved_cpus_per_rank': RESERVED, 'effective_cpus_per_rank': EFFECTIVE,
            'launch_id': launch_id, 'created_epoch': time.time(), 'source_hashes': sources(),
            'capacity_proof': proof, 'capacity_proof_path': proof_file['path'],
            'capacity_proof_sha256': proof_file['sha256'], 'previous_attempt': prior,
            'parent_snapshot': raw, 'allocated_memory_bytes': memory,
            'command': srun_command(parent, node, RANKS, directory),
            'resources': {'memory_per_rank_bytes': v3.RANK_MEMORY, 'gpus': 0, 'max_step_seconds': 7200}}
    plan['operational_digest'] = ag.digest(plan)
    publish(STAGE/'parents'/(parent+'.json'), {'launch_id': launch_id, 'plan': plan})
    directory.mkdir(parents=True, exist_ok=False)
    publish(directory/'plan.json', plan)
    with (directory/'supervisor.log').open('x') as log:
        child = subprocess.Popen([old.PYTHON, '-B', str(Path(__file__).resolve()), 'supervise', '--launch-dir', str(directory)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=v3.clean_environment(), start_new_session=True, close_fds=True)
    receipt = {'state': 'supervisor_started_not_yet_step_verified', 'pid': child.pid, 'parent': parent,
               'launch_id': launch_id, 'ranks': RANKS, 'epoch': time.time()}
    publish(directory/'launch.json', receipt)
    return receipt


def rank_entry(directory):
    directory = Path(directory)
    plan = verify_plan(directory)
    parent, node = plan['parent'], plan['node']
    step, rank = os.environ.get('SLURM_STEP_ID', ''), int(os.environ['SLURM_PROCID'])
    require(os.environ.get('SLURM_JOB_ID') == parent and socket.gethostname() == node and step.isdigit(), 'wrong actual parent/step/node')
    require(os.environ.get('SLURM_NTASKS') == str(RANKS) and os.environ.get('SLURM_NNODES') == '1' and
            os.environ.get('SLURM_CPUS_PER_TASK') == str(RESERVED) and rank in range(RANKS), 'wrong actual Slurm reservation')
    require(all(os.environ.get(k) == val for k, val in ag.CPU_ENV.items()) and os.environ.get('GPU_DEVICE_ORDINAL') == '-1',
            'CPU-only visibility lost')
    reserved = sorted(os.sched_getaffinity(0))
    import psutil
    samples = [psutil.cpu_percent(interval=1, percpu=True) for _ in range(2)]
    selected = select_cpus(reserved, samples)
    os.sched_setaffinity(0, set(selected))
    require(sorted(os.sched_getaffinity(0)) == selected, 'could not constrain own affinity to effective16')
    publish(directory/f'rank-{rank}.json', {'job_id': parent, 'step_id': step, 'rank': rank, 'host': node,
            'pid': os.getpid(), 'reserved_cpu_ids': reserved, 'selected_cpu_ids': selected,
            'cpu_percent_samples': samples, 'slurm_cpus_per_task': os.environ['SLURM_CPUS_PER_TASK'],
            'effective_fit_cpus': EFFECTIVE, 'only_own_process_affinity_changed': True})
    budget_path = directory/'budget.json'
    if rank == 0:
        v3.validate_capacity(plan['capacity_proof'], parent, node, RANKS)
        previous_attempt(plan['previous_attempt']['directory'], parent, node)
        memory = v3.memory_snapshot(parent, plan['allocated_memory_bytes'])
        v3.check_memory(memory, RANKS, startup=True)
        started = time.monotonic()
        raw = old.command(['scontrol', 'show', 'job', '-o', parent], env=dict(os.environ, TZ='UTC'))
        finished, wall = time.monotonic(), time.time()
        validate_parent(raw, parent, node, RANKS)
        budget = old.capped_budget(raw, parent, node, started, finished, wall, node)
        budget.update(step_id=step, memory=memory)
        publish(budget_path, budget)
    end = time.monotonic() + 90
    while not budget_path.exists():
        require(time.monotonic() < end, 'rank0 budget unavailable')
        time.sleep(.2)
    budget = read(budget_path)
    require(budget['step_id'] == step and budget['environment']['JOB_BUDGET_JOB_ID'] == parent and
            budget['environment']['JOB_BUDGET_MONOTONIC_HOST'] == node, 'wrong shared budget identity')
    os.environ.update(budget['environment'])
    os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    os.execvpe('ionice', ['ionice', '-c', '3', old.PYTHON, '-B', str(Path(__file__).resolve()),
                        'work', '--launch-dir', str(directory)], os.environ)


def preflight(original, budget, audit, stop, ranks, directory):
    require(ranks == RANKS and os.environ.get('SLURM_CPUS_PER_TASK') == str(RESERVED) and
            os.environ.get('SLURM_NTASKS') == str(RANKS), 'actual reservation must stay2x32')
    require(all(os.environ.get(k) == val for k, val in ag.CPU_ENV.items()) and os.environ.get('GPU_DEVICE_ORDINAL') == '-1', 'CPU-only mask required')
    require(budget.monotonic_end is not None and 0 < budget.remaining() <= 6901, 'invalid shared monotonic budget')
    rank = int(os.environ['SLURM_PROCID'])
    chosen = read(Path(directory)/f'rank-{rank}.json')
    require(chosen['job_id'] == os.environ['SLURM_JOB_ID'] and chosen['step_id'] == os.environ['SLURM_STEP_ID'] and
            chosen['host'] == socket.gethostname() and chosen['selected_cpu_ids'] == sorted(os.sched_getaffinity(0)) and
            select_cpus(chosen['reserved_cpu_ids'], chosen['cpu_percent_samples']) == chosen['selected_cpu_ids'], 'effective CPU receipt differs')
    record = {'pass': True, 'plan_id': ag.PLAN, 'rank': rank, 'job_id': chosen['job_id'],
              'step_id': chosen['step_id'], 'host': chosen['host'], 'pid': os.getpid(),
              'cpu_affinity': chosen['selected_cpu_ids'], 'reserved_cpu_ids': chosen['reserved_cpu_ids'],
              'nice': os.getpriority(os.PRIO_PROCESS, 0), 'environment': dict(ag.CPU_ENV),
              'monotonic_deadline': budget.monotonic_end, 'worker_sha256': old.WORKER_SHA,
              'adapter_sources': sources(), **original.cpu_check()}
    publish(audit/f'preflight-{rank}.json', record)
    deadline = min(time.monotonic() + 180, budget.monotonic_end-ag.NEW_FIT_GUARD)
    while not all((audit/f'preflight-{i}.json').exists() for i in range(RANKS)):
        require(not stop.reason and time.monotonic() < deadline, 'two-rank preflight stopped/timed out')
        time.sleep(.5)
    rows = [read(audit/f'preflight-{i}.json') for i in range(RANKS)]
    v3.validate_gate(rows, record['job_id'], record['step_id'], record['host'], RANKS)
    require(len(set(rows[0]['reserved_cpu_ids']) | set(rows[1]['reserved_cpu_ids'])) == RANKS*RESERVED,
            'Slurm reserved32CPU sets overlap')
    require(all(row['adapter_sources'] == record['adapter_sources'] and row['monotonic_deadline'] == budget.monotonic_end for row in rows), 'rank source/budget differs')
    return record


class EffectiveFitOS:
    """Only frozen fitter's one CPU-count lookup changes; global os stays real."""
    def __getattr__(self, name):
        return getattr(os, name)

    def getenv(self, key, default=None):
        if key == 'SLURM_CPUS_PER_TASK':
            require(os.environ.get(key) == str(RESERVED) and len(os.sched_getaffinity(0)) == EFFECTIVE,
                    'native16 adapter lacks real32/effective16 proof')
            return str(EFFECTIVE)
        return os.getenv(key, default)


def seed_entry(directory, key, seed):
    plan = verify_plan(directory)
    require(os.environ.get('SLURM_JOB_ID') == plan['parent'] and socket.gethostname() == plan['node'], 'wrong seed parent/node')
    rank = int(os.environ['SLURM_PROCID'])
    chosen = read(Path(directory)/f'rank-{rank}.json')
    require(chosen['step_id'] == os.environ.get('SLURM_STEP_ID') and chosen['selected_cpu_ids'] == sorted(os.sched_getaffinity(0)), 'seed effective affinity changed')
    EffectiveFitOS().getenv('SLURM_CPUS_PER_TASK')
    require(re.fullmatch(r'[0-9a-f]{24}', key or '') and seed in ag.SEEDS, 'invalid seed identity')
    def load():
        original, scientific, pairs = ORIGINAL_LOAD()
        original.os = EffectiveFitOS()
        return original, scientific, pairs
    ag.load_original = load
    audit = ag.OUT/'auxiliary'/CAMPAIGN/f"j{plan['parent']}-s{os.environ['SLURM_STEP_ID']}"
    publish(audit/f'native-cpu-{key}-{seed:02d}-r{rank}.json', {'reserved_cpus_per_task': RESERVED,
            'actual_slurm_env_cpus': os.environ['SLURM_CPUS_PER_TASK'], 'effective_native_cpus': EFFECTIVE,
            'actual_cpu_affinity': sorted(os.sched_getaffinity(0)), 'fitter_source_hashes': ag.SOURCE_HASHES,
            'global_slurm_environment_unchanged': True, 'model_configuration_unchanged': True})
    ag.seed_entry(key, seed)


def configure(directory):
    """New process-local operational bindings; no frozen files or model edits."""
    v3.STAGE, v3.CAMPAIGN = STAGE, CAMPAIGN
    v3.verify_plan, v3.sources, v3.srun_command, v3.validate_parent = verify_plan, sources, srun_command, validate_parent
    v3.exact_step_evidence = exact_step_evidence
    v3.dynamic_preflight = lambda original, budget, audit, stop, ranks: preflight(original, budget, audit, stop, ranks, directory)
    def guarded(command, log, fd, budget, stop, plan):
        require(command[:4] == [sys.executable, '-B', str(v3.REPO/'table6_restart_ag.py'), 'seed'], 'unexpected frozen seed subprocess')
        rewritten = [*command[:2], str(Path(__file__).resolve()), *command[3:], '--launch-dir', str(directory)]
        return ORIGINAL_GUARDED(rewritten, log, fd, budget, stop, plan)
    v3.guarded_run_seed = guarded


def exact_step_evidence(claim, *, run=subprocess.run, now=None):
    # This site's squeue step formatter supports%i but not job-state%T.
    def supported_query(command, **kwargs):
        if command[:2] == ['squeue', '--steps']:
            command = ['%i' if item == '%i|%T' else item for item in command]
        return run(command, **kwargs)
    return ORIGINAL_EXACT(claim, run=supported_query, now=now)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'supervise', 'rank', 'work', 'seed'))
    parser.add_argument('--parent'); parser.add_argument('--node'); parser.add_argument('--proof', type=Path)
    parser.add_argument('--launch-id'); parser.add_argument('--previous-launch-dir', type=Path)
    parser.add_argument('--launch-dir', type=Path); parser.add_argument('--key'); parser.add_argument('--seed', type=int)
    args = parser.parse_args(argv)
    if args.action == 'launch':
        print(json.dumps(launch(args.parent, args.node, args.proof, args.launch_id, args.previous_launch_dir)), flush=True)
        return 0
    configure(args.launch_dir)
    if args.action == 'supervise':
        plan = verify_plan(args.launch_dir)
        previous_attempt(plan['previous_attempt']['directory'], plan['parent'], plan['node'])
        return v3.supervise(args.launch_dir)
    if args.action == 'rank':
        rank_entry(args.launch_dir)
        return 0
    if args.action == 'seed':
        seed_entry(args.launch_dir, args.key, args.seed)
        return 0
    return v3.work_entry(args.launch_dir)


if __name__ == '__main__':
    sys.exit(main())
