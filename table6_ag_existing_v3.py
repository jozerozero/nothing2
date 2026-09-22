"""CPU-only, proof-gated continuation of the immutable AutoGluon 457x15 plan.

Only the two explicitly approved 2-TiB allocations are eligible. Each actual
Slurm rank has 16 CPUs and a 128-GiB reservation. No sbatch, GPU use, original
source edits, deletion of claims, model configuration changes, or fresh seeds.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import table6_restart_ag as ag
import table6_restart_ag_sidecar as old
from table6_restart_deadline import EnvironmentBudget, parse_fields, parse_duration

REPO = Path(__file__).resolve().parent
STAGE = ag.ROOT / 'stage/table6_autogluon_existing_20260922_v3'
CAMPAIGN = 'table6_autogluon_existing_20260922_v3'
ALLOWED = {'206116': 'auh7-1b-gpu-308', '206117': 'auh7-1b-gpu-306'}
GIB = 1024 ** 3
RANK_MEMORY = 128 * GIB
PARENT_MARGIN = 64 * GIB
NODE_MARGIN = 32 * GIB
FILES = ('table6_ag_existing_v3.py', 'table6_restart_ag.py',
         'table6_restart_ag_sidecar.py', 'table6_restart_deadline.py')
require, read, publish = ag.require, ag.read, ag.publish


def sources():
    result = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in FILES}
    require(result['table6_restart_ag.py'] == old.WORKER_SHA, 'frozen AG worker changed')
    return result


def target(parent, node, ranks):
    require(type(ranks) is int and ranks in (1, 2, 4), 'ranks must be 1, 2 or 4')
    require(ALLOWED.get(parent) == node, 'unapproved exact parent/node')


def validate_parent(raw, parent, node, ranks):
    target(parent, node, ranks)
    fields = old.validate_parent(raw, parent, node)
    allocated = dict(t.split('=', 1) for t in fields.get('AllocTRES', '').split(',') if '=' in t)
    capacity = old.memory_bytes(allocated.get('mem', fields.get('MinMemoryNode', '0')))
    require(capacity >= ranks * RANK_MEMORY + PARENT_MARGIN, 'parent memory reservation lacks safety headroom')
    require(int(fields['NumCPUs']) >= 16 * ranks, 'insufficient allocated CPUs')
    require(parse_duration(fields['TimeLimit']) - parse_duration(fields['RunTime']) > 7500,
            'parent must accommodate the complete bounded two-hour step')
    return fields


def integer(value, label):
    require(type(value) is int and value >= 0, 'invalid nonnegative integer: ' + label)
    return value


def validate_capacity(proof, parent, node, ranks, now=None):
    target(parent, node, ranks)
    now = time.time() if now is None else now
    require(proof.get('allow_cpu_sidecar') is True and proof.get('parent_job_id') == parent
            and proof.get('node') == node, 'capacity proof target/authorization differs')
    observations = proof.get('observations', [])
    require(len(observations) == 2, 'exactly two capacity observations required')
    times, idle_sets = [], []
    for sample in observations:
        require(sample.get('parent_job_id') == parent and sample.get('node') == node,
                'capacity observation came from another parent/node')
        when = float(sample['observed_epoch'])
        require(math.isfinite(when) and 0 <= now - when <= 300, 'stale/future capacity proof')
        times.append(when)
        ids = sample['idle_cpu_ids']
        require(isinstance(ids, list) and len(ids) == len(set(ids)) and
                all(type(cpu) is int and cpu >= 0 for cpu in ids), 'invalid idle CPU ids')
        parent_ids = sample['parent_cpu_ids']
        require(isinstance(parent_ids, list) and all(type(c) is int and c >= 0 for c in parent_ids)
                and set(ids) <= set(parent_ids), 'idle CPUs outside parent allocation')
        idle_sets.append(set(ids))
        memory = integer(sample['parent_memory_current_bytes'], 'parent memory current')
        limit = integer(sample['parent_memory_limit_bytes'], 'parent memory limit')
        available = integer(sample['available_memory_bytes'], 'node memory available')
        require(memory + ranks * RANK_MEMORY + PARENT_MARGIN <= limit,
                'whole-parent memory headroom insufficient')
        require(available >= ranks * RANK_MEMORY + NODE_MARGIN, 'node memory headroom insufficient')
        require(sample.get('parent_memory_source') in ('cgroup_v1', 'cgroup_v2'),
                'whole-parent cgroup measurement is required')
    require(times[1] - times[0] >= 15, 'independent observations must be at least15s apart')
    require(len(idle_sets[0] & idle_sets[1]) >= ranks * 16, 'not enough repeatedly idle parent CPUs')
    return proof


def clean_environment(environ=None):
    env = old.clean_environment(environ)
    env.pop('PYTHONHOME', None)
    for key in list(env):
        if key.startswith('T6_AG_V3_'):
            del env[key]
    return env


def srun_command(parent, node, ranks, directory):
    target(parent, node, ranks)
    return ['srun', '--jobid=' + parent, '--nodelist=' + node, '--nodes=1',
            '--ntasks=' + str(ranks), '--ntasks-per-node=' + str(ranks),
            '--cpus-per-task=16', '--mem=' + str(128 * ranks) + 'G',
            '--gpus=0', '--gpus-per-task=0', '--gres=none', '--exact', '--exclusive',
            '--immediate=10', '--time=02:00:00', '--cpu-bind=cores',
            '--kill-on-bad-exit=1', '--input=none', '--export=ALL',
            'env', 'CPU_ONLY=1', 'CUDA_VISIBLE_DEVICES=', 'HIP_VISIBLE_DEVICES=-1',
            'ROCR_VISIBLE_DEVICES=-1', 'GPU_DEVICE_ORDINAL=-1', old.PYTHON, '-B',
            str(Path(__file__).resolve()), 'rank', '--launch-dir', str(directory)]


def verify_plan(directory):
    directory = Path(directory).resolve()
    require(directory.parent == (STAGE / 'launches').resolve(), 'unexpected launch namespace')
    plan = read(directory / 'plan.json')
    target(plan['parent'], plan['node'], plan['ranks'])
    require(plan['plan_id'] == ag.PLAN and plan['source_hashes'] == sources(), 'plan/source changed')
    require(plan['operational_digest'] == ag.digest({k: v for k, v in plan.items() if k != 'operational_digest'}),
            'operational plan changed')
    require(plan['command'] == srun_command(plan['parent'], plan['node'], plan['ranks'], directory),
            'srun command changed')
    return plan


def launch(parent, node, ranks, proof_path, launch_id):
    target(parent, node, ranks)
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,80}', launch_id or ''), 'invalid unique launch id')
    proof_path = Path(proof_path)
    require(not proof_path.is_symlink(), 'symlink proof refused')
    data = proof_path.read_bytes()
    proof = validate_capacity(json.loads(data), parent, node, ranks)
    raw = old.command(['scontrol', 'show', 'job', '-o', parent], env=dict(clean_environment(), TZ='UTC'))
    fields = validate_parent(raw, parent, node, ranks)
    allocated = dict(t.split('=', 1) for t in fields.get('AllocTRES', '').split(',') if '=' in t)
    allocated_memory = old.memory_bytes(allocated.get('mem', fields.get('MinMemoryNode', '0')))
    for row in proof['observations']:
        require(row['parent_memory_current_bytes'] + ranks * RANK_MEMORY + PARENT_MARGIN <= allocated_memory,
                'observed usage plus reservation exceeds Slurm parent memory')
    directory = STAGE / 'launches' / launch_id
    plan = {'plan_id': ag.PLAN, 'parent': parent, 'node': node, 'ranks': ranks,
            'launch_id': launch_id, 'created_epoch': time.time(), 'source_hashes': sources(),
            'capacity_proof': proof, 'capacity_proof_path': str(proof_path.resolve()),
            'capacity_proof_sha256': hashlib.sha256(data).hexdigest(),
            'parent_snapshot': raw, 'allocated_memory_bytes': allocated_memory,
            'command': srun_command(parent, node, ranks, directory),
            'resources': {'cpus_per_rank': 16, 'memory_per_rank_bytes': RANK_MEMORY,
                          'gpus': 0, 'max_step_seconds': 7200},
            'scope': 'existing approved CPU/RAM capacity only; fixedbest original15seeds'}
    plan['operational_digest'] = ag.digest(plan)
    # New namespace; no erasing old attempts, and uncertain submissions remain
    # blocked. Another attempt needs explicit review, not automated retry.
    publish(STAGE / 'parents' / (parent + '.json'), {'launch_id': launch_id, 'plan': plan})
    directory.mkdir(parents=True, exist_ok=False)
    publish(directory / 'plan.json', plan)
    with (directory / 'supervisor.log').open('x') as log:
        child = subprocess.Popen([old.PYTHON, '-B', str(Path(__file__).resolve()),
                                  'supervise', '--launch-dir', str(directory)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=clean_environment(), start_new_session=True, close_fds=True)
    receipt = {'state': 'supervisor_started_not_yet_step_verified', 'pid': child.pid,
               'parent': parent, 'launch_id': launch_id, 'ranks': ranks, 'epoch': time.time()}
    publish(directory / 'launch.json', receipt)
    return receipt


def supervise(directory):
    directory = Path(directory)
    plan = verify_plan(directory)
    validate_capacity(plan['capacity_proof'], plan['parent'], plan['node'], plan['ranks'])
    raw = old.command(['scontrol', 'show', 'job', '-o', plan['parent']], env=dict(clean_environment(), TZ='UTC'))
    validate_parent(raw, plan['parent'], plan['node'], plan['ranks'])
    publish(directory / 'srun-intent.json', {'epoch': time.time(), 'command': plan['command']})
    stopped, active = [], [None]
    def forward(signum, frame):
        stopped.append(signum)
        if active[0] is not None and active[0].poll() is None:
            active[0].send_signal(signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, forward)
    with (directory / 'srun.log').open('x') as log:
        if stopped:
            return 75
        child = subprocess.Popen(plan['command'], stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, env=clean_environment(), close_fds=True)
        active[0] = child
        if stopped:
            child.send_signal(stopped[-1])
        publish(directory / 'srun.json', {'pid': child.pid, 'epoch': time.time()})
        code = child.wait()
    publish(directory / 'completion.json', {'returncode': code, 'epoch': time.time(),
            'scientific_completion_not_implied': True, 'state': 'step_command_finished'})
    return code


def parent_memory_files(parent, text=None, root=Path('/sys/fs/cgroup')):
    """Find this task's *parent allocation*, never its individual step cgroup."""
    text = Path('/proc/self/cgroup').read_text() if text is None else text
    candidates = []
    for line in text.splitlines():
        hierarchy, controllers, relative = line.split(':', 2)
        if hierarchy == '0' and controllers == '':
            prefix, used, limit, source = root, 'memory.current', 'memory.max', 'cgroup_v2'
        elif 'memory' in controllers.split(','):
            prefix, used, limit, source = root / 'memory', 'memory.usage_in_bytes', 'memory.limit_in_bytes', 'cgroup_v1'
        else:
            continue
        parts = Path(relative).parts
        hits = [i for i, part in enumerate(parts) if re.fullmatch('job_' + re.escape(parent) + r'(?:\.scope)?', part)]
        require(len(hits) == 1 and '..' not in parts, 'actual cgroup lacks unique exact parent')
        path = prefix.joinpath(*[p for p in parts[:hits[0] + 1] if p != '/'])
        candidates.append((path / used, path / limit, source))
    require(len(candidates) == 1, 'whole-parent memory controller unknown/ambiguous')
    return candidates[0]


def memory_snapshot(parent, allocated_memory):
    used, limit, source = parent_memory_files(parent)
    value, cap = used.read_text().strip(), limit.read_text().strip()
    require(value.isdigit() and (cap.isdigit() or cap == 'max'), 'invalid parent memory controller')
    effective = allocated_memory if cap == 'max' else min(allocated_memory, int(cap))
    available = next(int(row.split()[1]) * 1024 for row in Path('/proc/meminfo').read_text().splitlines()
                     if row.startswith('MemAvailable:'))
    return {'parent_memory_current_bytes': int(value), 'parent_memory_limit_bytes': effective,
            'parent_memory_source': source, 'parent_memory_path': str(used),
            'available_memory_bytes': available}


def check_memory(snapshot, ranks, *, startup):
    current, limit = snapshot['parent_memory_current_bytes'], snapshot['parent_memory_limit_bytes']
    additional = ranks * RANK_MEMORY if startup else 0
    require(current + additional + PARENT_MARGIN <= limit, 'whole_parent_memory_guard')
    require(snapshot['available_memory_bytes'] >= additional + NODE_MARGIN, 'node_memory_guard')


def rank_entry(directory):
    directory = Path(directory)
    plan = verify_plan(directory)
    parent, node, ranks = plan['parent'], plan['node'], plan['ranks']
    step = os.environ.get('SLURM_STEP_ID', '')
    require(os.environ.get('SLURM_JOB_ID') == parent and socket.gethostname() == node,
            'wrong actual parent/node')
    require(step.isdigit() and os.environ.get('SLURM_NTASKS') == str(ranks) and
            os.environ.get('SLURM_CPUS_PER_TASK') == '16' and os.environ.get('SLURM_NNODES') == '1',
            'wrong actual Slurm rank resources')
    require(all(os.environ.get(k) == v for k, v in ag.CPU_ENV.items()) and
            os.environ.get('GPU_DEVICE_ORDINAL') == '-1', 'CPU-only mask lost')
    rank = int(os.environ['SLURM_PROCID'])
    require(rank in range(ranks) and len(os.sched_getaffinity(0)) == 16, 'invalid rank/CPU binding')
    import psutil
    cpu_samples = [psutil.cpu_percent(interval=1, percpu=True) for _ in range(2)]
    require(all(cpu < len(row) and 0 <= row[cpu] < 25 for cpu in os.sched_getaffinity(0)
                for row in cpu_samples), 'Slurm assigned CPUs are not currently idle')
    publish(directory / f'rank-{rank}.json', {'job_id': parent, 'step_id': step, 'rank': rank,
            'host': node, 'pid': os.getpid(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'cpu_percent_samples': cpu_samples})
    budget_path = directory / 'budget.json'
    if rank == 0:
        validate_capacity(plan['capacity_proof'], parent, node, ranks)
        memory = memory_snapshot(parent, plan['allocated_memory_bytes'])
        check_memory(memory, ranks, startup=True)
        started = time.monotonic()
        raw = old.command(['scontrol', 'show', 'job', '-o', parent], env=dict(os.environ, TZ='UTC'))
        finished, wall = time.monotonic(), time.time()
        validate_parent(raw, parent, node, ranks)
        budget = old.capped_budget(raw, parent, node, started, finished, wall, node)
        budget.update(step_id=step, memory=memory)
        publish(budget_path, budget)
    timeout = time.monotonic() + 60
    while not budget_path.exists():
        require(time.monotonic() < timeout, 'rank0 budget missing; no fit permitted')
        time.sleep(.2)
    budget = read(budget_path)
    require(budget['step_id'] == step and budget['environment']['JOB_BUDGET_JOB_ID'] == parent and
            budget['environment']['JOB_BUDGET_MONOTONIC_HOST'] == node, 'budget identity mismatch')
    os.environ.update(budget['environment'])
    os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    os.execvpe('ionice', ['ionice', '-c', '3', old.PYTHON, '-B', str(Path(__file__).resolve()),
                        'work', '--launch-dir', str(directory)], os.environ)


def exact_step_evidence(claim, *, run=subprocess.run, now=None):
    """Used only inside frozen work_pair's acquired pair flock and byte CAS.

    An exact terminal Slurm step plus the previous wrapper's durable cleanup
    declaration replaces the old, overstrict requirement that its parent also
    finish. Old claims lacking a numeric step retain the old strict policy.
    """
    job, step = str(claim.get('job_id', '')), str(claim.get('step_id') or '')
    require(re.fullmatch(r'\d+', job), 'old claim has unknown parent')
    if not step.isdigit():
        return ag_original_terminal(claim, run=run, now=now)
    require(claim.get('state') == 'paused' and claim.get('descendants_reaped') is True,
            'old numeric-step claim lacks durable paused/descendants_reaped proof')
    require(claim.get('worker_sha256') == old.WORKER_SHA and claim.get('plan_id') == ag.PLAN,
            'old cleanup declaration comes from unknown worker/plan')
    require(claim.get('host') and claim.get('owner_token'), 'old owner identity incomplete')
    identity = job + '.' + step
    env = dict(os.environ, TZ='UTC')
    env.pop('SLURM_TIME_FORMAT', None)
    def command(args):
        response = run(args, capture_output=True, text=True, timeout=20, env=env)
        require(response.returncode == 0, 'owner scheduler query failed: ' + response.stderr)
        return response.stdout
    raw = command(['sacct', '-n', '-P', '-j', identity, '-o', 'JobIDRaw,State%30,End'])
    rows = [row.split('|') for row in raw.splitlines() if row.strip()]
    exact = [row for row in rows if len(row) >= 3 and row[0] == identity]
    require(len(exact) == 1, 'missing/duplicate exact old step accounting')
    _, state, end = exact[0][:3]
    require(state.split() and state.split()[0].rstrip('+') in ag.TERMINAL, 'old exact step remains nonterminal')
    ended = dt.datetime.fromisoformat(end)
    ended = ended.replace(tzinfo=dt.timezone.utc) if ended.tzinfo is None else ended
    now = time.time() if now is None else now
    age = now - ended.timestamp()
    require(age >= 120, 'old step termination grace not reached')
    queue = command(['squeue', '--steps', '-h', '-j', job, '-o', '%i|%T'])
    live = [line.split('|', 1)[0].strip() for line in queue.splitlines() if line.strip()]
    require(all(re.fullmatch(re.escape(job) + r'\.(?:\d+|batch|extern)', item) for item in live),
            'old-step queue response has unknown/truncated identity')
    require(identity not in live, 'old exact step still queued')
    return {'policy': 'exact_terminal_step_plus_durable_reaped_claim_and_held_pair_flock',
            'owner': identity, 'state': state, 'end': end, 'age_seconds': age,
            'checked_epoch': now, 'exact_step_absent_from_queue': True,
            'parent_may_still_be_running': True, 'descendants_reaped': True,
            'old_claim_sha256': ag.digest(claim)}


ag_original_terminal = ag.terminal_evidence


def validate_gate(records, job, step, host, ranks):
    require(len(records) == ranks and {r['rank'] for r in records} == set(range(ranks)), 'all ranks required')
    used = set()
    for record in records:
        require(record.get('pass') is True and record['job_id'] == job and record['step_id'] == step and
                record['host'] == host and record['plan_id'] == ag.PLAN and record['devices'] == 0 and
                record['nice'] == 19 and record['environment'] == ag.CPU_ENV and record.get('fastai'),
                'dynamic CPU gate/fastai identity mismatch')
        cpus = set(record['cpu_affinity'])
        require(len(cpus) == 16 and not cpus & used, 'CPU ranks overlap or lack16CPUs')
        used |= cpus
    return sorted(used)


def dynamic_preflight(original, budget, audit, stop, ranks):
    require(os.environ.get('SLURM_NTASKS') == str(ranks) and os.environ.get('SLURM_CPUS_PER_TASK') == '16'
            and os.environ.get('SLURM_NNODES') == '1', 'wrong dynamic rank shape')
    require(all(os.environ.get(k) == v for k, v in ag.CPU_ENV.items()) and
            os.environ.get('GPU_DEVICE_ORDINAL') == '-1', 'GPU visibility not fully masked')
    require(budget.monotonic_end is not None and 0 < budget.remaining() <= 6901, 'invalid monotonic budget')
    os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    rank = int(os.environ['SLURM_PROCID'])
    require(rank in range(ranks), 'unexpected rank')
    record = {'pass': True, 'plan_id': ag.PLAN, 'rank': rank, 'job_id': os.environ['SLURM_JOB_ID'],
              'step_id': os.environ['SLURM_STEP_ID'], 'host': socket.gethostname(), 'pid': os.getpid(),
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'nice': os.getpriority(os.PRIO_PROCESS, 0),
              'environment': dict(ag.CPU_ENV), 'epoch': time.time(),
              'monotonic_deadline': budget.monotonic_end, 'worker_sha256': old.WORKER_SHA,
              'adapter_sources': sources(), **original.cpu_check()}
    publish(audit / f'preflight-{rank}.json', record)
    end = min(time.monotonic() + 180, budget.monotonic_end - ag.NEW_FIT_GUARD)
    while not all((audit / f'preflight-{i}.json').exists() for i in range(ranks)):
        require(not stop.reason and time.monotonic() < end, 'dynamic preflight stopped/timed out')
        time.sleep(.5)
    records = [read(audit / f'preflight-{i}.json') for i in range(ranks)]
    validate_gate(records, record['job_id'], record['step_id'], record['host'], ranks)
    require(all(r['adapter_sources'] == record['adapter_sources'] and
                r['monotonic_deadline'] == budget.monotonic_end for r in records), 'rank source/budget mismatch')
    return record


def guarded_run_seed(command, log, lock_fd, budget, stop, plan):
    if stop.reason or budget.remaining() <= ag.NEW_FIT_GUARD:
        return None, stop.reason or 'allocation_budget'
    try:
        check_memory(memory_snapshot(plan['parent'], plan['allocated_memory_bytes']), plan['ranks'], startup=False)
    except (RuntimeError, OSError, ValueError) as exc:
        return None, 'memory_proof_unavailable_or_guard: ' + str(exc)
    env = dict(os.environ, T6_AG_PARENT_PID=str(os.getpid()), T6_AG_LOCK_FD=str(lock_fd))
    with Path(log).open('x') as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True, pass_fds=(lock_fd,))
        reason = None
        try:
            while child.poll() is None:
                own = ag.descendants(ag.process_snapshot(), os.getpid())
                rss = sum(row['rss'] for row in own.values())
                try:
                    check_memory(memory_snapshot(plan['parent'], plan['allocated_memory_bytes']), plan['ranks'], startup=False)
                except (RuntimeError, OSError, ValueError) as exc:
                    reason = 'memory_proof_unavailable_or_guard: ' + str(exc)
                    break
                if stop.reason or budget.remaining() <= ag.STOP_GUARD or rss > ag.RSS_LIMIT:
                    reason = stop.reason or ('allocation_budget' if budget.remaining() <= ag.STOP_GUARD else 'rss_guard')
                    break
                time.sleep(.5)
            code = child.poll()
        finally:
            try:
                ag.clean_children(child)
            except Exception as exc:
                raise ag.UnsafeCleanup('v3 cleanup unproven; retain inherited pair lock') from exc
        return code, reason or stop.reason or ('allocation_budget' if budget.remaining() <= ag.STOP_GUARD else None)


def work_entry(directory):
    directory = Path(directory)
    plan = verify_plan(directory)
    require(os.environ.get('SLURM_JOB_ID') == plan['parent'] and socket.gethostname() == plan['node'],
            'wrong actual work parent/node')
    require(os.environ.get('SLURM_STEP_ID', '').isdigit(), 'numeric actual step required')
    EnvironmentBudget.from_environment()  # Fail before loading any ML code.
    ag.CAMPAIGN = CAMPAIGN
    ag.terminal_evidence = exact_step_evidence
    ag.preflight = lambda original, budget, audit, stop: dynamic_preflight(original, budget, audit, stop, plan['ranks'])
    ag.run_seed = lambda command, log, fd, budget, stop: guarded_run_seed(command, log, fd, budget, stop, plan)
    # The inherited worker already does strict old-byte CAS under the pair flock,
    # saves prior claim+hash+proof, and only changes its own owner_token. Add an
    # immediate new-owner receipt before any fit, without altering its fitter.
    audit = ag.OUT / 'auxiliary' / CAMPAIGN / f"j{plan['parent']}-s{os.environ['SLURM_STEP_ID']}"
    seen = set()
    def audited_publish(path, value, *, replace=False):
        path = Path(path)
        if path.parent == ag.OUT / 'claims' and isinstance(value, dict) and value.get('state') == 'paused' and \
                value.get('job_id') == plan['parent'] and value.get('step_id') == os.environ['SLURM_STEP_ID']:
            require(not ag.descendants(ag.process_snapshot(), os.getpid()), 'paused claim still has descendants')
            value = dict(value, descendants_reaped=True)
        token = value.get('owner_token') if isinstance(value, dict) else None
        takeover = path.parent == ag.OUT / 'claims' and token and token not in seen
        if takeover:
            require(value.get('job_id') == plan['parent'] and value.get('step_id') == os.environ['SLURM_STEP_ID'],
                    'new claim owner differs from genuine step')
        publish(path, value, replace=replace)
        if takeover:
            seen.add(token)
            publish(audit / f"new-owner-{value['key']}-r{value['rank']}.json",
                    {'claim': value, 'claim_path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'operational_digest': plan['operational_digest'], 'under_pair_flock': True})
    ag.publish = audited_publish
    return ag.main(['launch'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'supervise', 'rank', 'work'))
    parser.add_argument('--parent'); parser.add_argument('--node')
    parser.add_argument('--ranks', type=int, choices=(1, 2, 4), default=4)
    parser.add_argument('--proof', type=Path); parser.add_argument('--launch-id')
    parser.add_argument('--launch-dir', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'launch':
        print(json.dumps(launch(args.parent, args.node, args.ranks, args.proof, args.launch_id), allow_nan=False))
        return 0
    if args.action == 'supervise':
        return supervise(args.launch_dir)
    if args.action == 'rank':
        rank_entry(args.launch_dir)
        return 0
    return work_entry(args.launch_dir)


if __name__ == '__main__':
    sys.exit(main())
