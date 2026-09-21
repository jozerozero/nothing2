"""Bounded CPU-only continuation of the frozen 15-seed AutoGluon campaign.

Run four Slurm ranks with ``launch``.  The batch shell must export a single
same-node EnvironmentBudget before srun.  No submission or parent mutation is
performed here.  Model fitting remains frozen run_fixed.one_seed; only its
publication function is replaced with exclusive, atomic publication.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import fcntl
import hashlib
import importlib.util
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
import uuid

from table6_restart_deadline import EnvironmentBudget

ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
BASE = ROOT / 'stage/table6_standard457_fixedbest_bg1_gpu_recovery_20260909_v2'
OUT = ROOT / 'evaluation/table6_standard457_fixedbest_gt1_20260909_v1'
PLAN = '95ed50cd7348ed167f71ff159f6af14cc351e67959d6d227f9201bdc024fbb85'
CAMPAIGN = 'table6_autogluon_restart_20260922_v1'
SOURCE_HASHES = {
    'run_fixed.py': '1ca5cfcb611b84cc0243fb1b8cd173e0fa5b02ff3450b51aafae2ebf3b94e076',
    'common_fixed.py': '23266c9926c66eb42af9d78923200a46e5eef4db774b5d4a33ab1fd1386f7837',
    'standard_data.py': 'ab6cff0ea3f049154c1bfe5f81e06de2177a7ed0f83ebe4c568d91735edfbd83',
}
SEEDS = list(range(15))
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
CPU_ENV = {'CPU_ONLY': '1', 'CUDA_VISIBLE_DEVICES': '',
           'HIP_VISIBLE_DEVICES': '-1', 'ROCR_VISIBLE_DEVICES': '-1'}
NEW_FIT_GUARD = 180
STOP_GUARD = 60
RSS_LIMIT = 120 * 1024 ** 3


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    require(not Path(path).is_symlink(), f'symlink refused: {path}')
    return json.loads(Path(path).read_text())


def publish(path, value, *, replace=False):
    """Hard-link publication cannot replace an existing seed/result/receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f'symlink refused: {path}')
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with tmp.open('x') as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(tmp, path)  # Only current-owned mutable claim state.
        else:
            os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_original():
    for name, expected in SOURCE_HASHES.items():
        require(hashlib.sha256((BASE / name).read_bytes()).hexdigest() == expected,
                f'frozen source changed: {name}')
    sys.path.insert(0, str(BASE))
    spec = importlib.util.spec_from_file_location('_table6_frozen_ag', BASE / 'run_fixed.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    require(Path(original.OUT) == OUT and Path(original.STAGE) == BASE, 'frozen paths changed')
    plan = original.load_plan()
    require(plan.get('plan_id') == PLAN and digest({k: v for k, v in plan.items() if k != 'plan_id'}) == PLAN,
            'frozen plan identity changed')
    require(plan.get('seeds') == SEEDS and plan.get('hpo_trials') == 0, 'seed/HPO contract changed')
    pairs = [p for p in plan['pairs'] if p['method'] == 'AutoGluon']
    require(len(pairs) == 457 and len({p['dataset'] for p in pairs}) == 457, 'AG membership changed')
    for pair in pairs:
        require(digest(pair['config']) == pair['config_sha256'], 'configuration identity changed')
        require(re.fullmatch('[a-f0-9]{24}', pair['key']) is not None, 'invalid pair key')
        require('/' not in pair['dataset'] and pair['dataset'] not in ('.', '..'), 'invalid dataset path')
    # No fitter/preprocessing/seed change, only prohibit overwriting publication.
    original.atomic = publish
    return original, plan, pairs


def validate_seed(record, pair, seed, plan_id=PLAN):
    required = {'complete': True, 'plan_id': plan_id, 'method': 'AutoGluon',
                'dataset': pair['dataset'], 'suite': pair['suite'], 'seed': seed,
                'selected_config_sha256': pair['config_sha256'], 'hpo_trials': 0}
    require(all(record.get(k) == v for k, v in required.items()) and not record.get('error'),
            'seed identity/complete mismatch')
    require(record.get('selected_config_source') == pair['source'], 'seed configuration source mismatch')
    require(record['data_audit'].get('exact_frozen_cache_match') is True, 'seed lacks exact split audit')
    for metric in ('ACC', 'AUC', 'F1'):
        value = float(record['metrics'][metric])
        require(math.isfinite(value) and 0 <= value <= 1, f'invalid seed metric {metric}')
    return record


def inspect_pair(pair, output=None):
    """Read every existing seed; malformed partials are errors, never recomputed."""
    output = OUT if output is None else output
    records = {}
    for seed in SEEDS:
        path = output / 'seeds' / pair['key'] / f'{seed:02d}.json'
        if path.exists():
            records[seed] = validate_seed(read(path), pair, seed)
    audits = [r['data_audit'] for r in records.values()]
    require(not audits or all(x == audits[0] for x in audits), 'partial seed split audits differ')
    result = output / 'results/AutoGluon' / (pair['dataset'] + '.json')
    if result.exists():
        value = read(result)
        required = {'complete': True, 'plan_id': PLAN, 'method': 'AutoGluon',
                    'dataset': pair['dataset'], 'suite': pair['suite'], 'hpo_trials': 0,
                    'seed_count': 15, 'seeds': SEEDS, 'selected_config': pair['config'],
                    'selected_config_sha256': pair['config_sha256'], 'selected_config_source': pair['source']}
        require(all(value.get(k) == v for k, v in required.items()) and not value.get('error'),
                'aggregate identity mismatch')
        require(len(records) == 15 and value['data_audit'] == audits[0], 'aggregate lacks 15 matching seeds')
        for metric in ('ACC', 'AUC', 'F1'):
            average = sum(records[s]['metrics'][metric] for s in SEEDS) / 15
            require(math.isfinite(float(value['metrics_mean'][metric])) and
                    abs(value['metrics_mean'][metric] - average) < 1e-12, 'aggregate mean mismatch')
        return records, True
    return records, False


def terminal_evidence(claim, *, run=subprocess.run, now=None):
    """Free flock is necessary, NOT sufficient: require terminal owner plus grace."""
    job = str(claim.get('job_id', ''))
    step = str(claim.get('step_id') or '')
    require(re.fullmatch(r'\d+', job) is not None, 'claim owner job unknown')
    require(not step or re.fullmatch(r'\d+|batch|extern', step) is not None, 'claim owner step malformed')
    env = dict(os.environ, TZ='UTC')
    env.pop('SLURM_TIME_FORMAT', None)
    def command(args, allow_gone=False):
        result = run(args, text=True, capture_output=True, timeout=20, env=env)
        if allow_gone and result.returncode != 0 and not result.stdout.strip():
            # Only after sacct established terminal identity and grace. Do not
            # confuse scheduler outages/permission failures with an empty queue.
            if re.fullmatch(r'squeue: error: Invalid job id specified(?:\s*[:=]\s*' + re.escape(job) + r')?',
                            result.stderr.strip()):
                return ''
        require(result.returncode == 0, f'owner query failed: {args[0]} {result.stderr}')
        return result.stdout.strip()
    raw = command(['sacct', '-n', '-P', '-j', job, '-o', 'JobIDRaw,State%30,End'])
    rows = {}
    for line in raw.splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 3:
            require(fields[0] not in rows, 'duplicate accounting identity')
            rows[fields[0]] = fields[1:3]
    now = time.time() if now is None else now
    proof = []
    for owner in [job] + ([job + '.' + step] if step else []):
        require(owner in rows, f'missing terminal accounting: {owner}')
        state, end = rows[owner]
        require(state.split()[0].rstrip('+') in TERMINAL, f'owner nonterminal: {owner}: {state}')
        ended = dt.datetime.fromisoformat(end)
        if ended.tzinfo is None:
            ended = ended.replace(tzinfo=dt.timezone.utc)
        age = now - ended.timestamp()
        require(age >= 120, f'owner termination grace not reached: {owner}')
        proof.append({'owner': owner, 'state': state, 'end': end, 'age_seconds': age})
    require(not command(['squeue', '-h', '-j', job, '-o', '%i|%T'], allow_gone=True), 'owner job remains queued')
    require(not command(['squeue', '--steps', '-h', '-j', job, '-o', '%i|%T'], allow_gone=True), 'owner step remains queued')
    return {'checked_epoch': now, 'terminal': proof, 'queue_empty': True, 'steps_empty': True}


class Stop:
    reason = None

    def signal(self, signum, _frame):
        self.reason = f'signal_{signal.Signals(signum).name}'


class UnsafeCleanup(RuntimeError):
    """Must retain pair flock until scheduler cgroup cleanup."""


def process_snapshot():
    """Linux identities include starttime, preventing accidental PID-reuse kills."""
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            result[int(path.name)] = {'state': fields[0], 'ppid': int(fields[1]),
                                     'group': int(fields[2]), 'start': int(fields[19]),
                                     'rss': int(fields[21]) * os.sysconf('SC_PAGE_SIZE')}
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def descendants(snapshot, parent):
    owned = {parent}
    while True:
        expanded = owned | {pid for pid, row in snapshot.items() if row['ppid'] in owned}
        if expanded == owned:
            return {pid: snapshot[pid] for pid in owned if pid != parent and pid in snapshot}
        owned = expanded


def enable_subreaper():
    require(sys.platform == 'linux', 'Linux subreaper/proc ownership required')
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(36, 1, 0, 0, 0) == 0, 'could not enable child subreaper')


def clean_children(child, *, grace=20):
    """Reap the seed and any adopted/detached descendants before releasing flock."""
    deadline = time.monotonic() + grace
    while True:
        child.poll()
        owned = descendants(process_snapshot(), os.getpid())
        for pid, record in owned.items():
            if record['state'] == 'Z':
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
                continue
            current = process_snapshot().get(pid)
            if current and current['start'] == record['start']:
                try:
                    os.kill(pid, signal.SIGTERM if time.monotonic() < deadline else signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if not descendants(process_snapshot(), os.getpid()):
            child.wait()
            return
        if time.monotonic() > deadline + 20:
            raise RuntimeError('owned descendants could not be reaped; retain claim lock until Slurm cleanup')
        time.sleep(.2)


def run_seed(command, log, lock_fd, budget, stop):
    if stop.reason or budget.remaining() <= NEW_FIT_GUARD:
        return None, stop.reason or 'allocation_budget'
    env = dict(os.environ, T6_AG_PARENT_PID=str(os.getpid()), T6_AG_LOCK_FD=str(lock_fd))
    with log.open('x') as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True, pass_fds=(lock_fd,))
        reason = None
        try:
            while child.poll() is None:
                own_rss = sum(p['rss'] for p in descendants(process_snapshot(), os.getpid()).values())
                if stop.reason or budget.remaining() <= STOP_GUARD or own_rss > RSS_LIMIT:
                    reason = stop.reason or ('allocation_budget' if budget.remaining() <= STOP_GUARD else 'rss_guard')
                    break
                time.sleep(.5)
            rc = child.poll()
        finally:
            try:
                clean_children(child)
            except Exception as exc:
                raise UnsafeCleanup('cannot verify owned descendants reaped') from exc
        # Signals can terminate the child before the first poll, or race its
        # exit. They remain operational pauses, not new model failures.
        reason = reason or stop.reason
        if reason is None and budget.remaining() <= STOP_GUARD:
            reason = 'allocation_budget'
        return rc, reason


def validate_gate(records, job, step, host):
    require(len(records) == 4 and {r['rank'] for r in records} == set(range(4)), 'four ranks required')
    used = set()
    for row in records:
        require(row.get('pass') is True and row['job_id'] == job and row['step_id'] == step and
                row['host'] == host and row['plan_id'] == PLAN and row['devices'] == 0 and
                row['nice'] == 19 and row['environment'] == CPU_ENV,
                'CPU gate identity/visibility mismatch')
        cpus = set(row['cpu_affinity'])
        require(len(cpus) == 16 and not (cpus & used), 'CPU ranks overlap or lack 16 CPUs')
        used |= cpus
    return sorted(used)


def preflight(original, budget, audit, stop):
    require(all(os.environ.get(k) == v for k, v in CPU_ENV.items()), 'CPU-only environment required')
    require(os.environ.get('SLURM_NTASKS') == '4' and os.environ.get('SLURM_CPUS_PER_TASK') == '16'
            and os.environ.get('SLURM_NNODES') == '1', 'expected one node / four 16-CPU ranks')
    require(budget.monotonic_end is not None and 0 < budget.remaining() <= 6901, 'same-node bounded budget required')
    os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    rank = int(os.environ['SLURM_PROCID'])
    require(rank in range(4), 'unexpected rank')
    record = {'pass': True, 'plan_id': PLAN, 'rank': rank, 'job_id': os.environ['SLURM_JOB_ID'],
              'step_id': os.environ['SLURM_STEP_ID'], 'host': socket.gethostname(),
              'pid': os.getpid(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'nice': os.getpriority(os.PRIO_PROCESS, 0), 'environment': dict(CPU_ENV),
              'epoch': time.time(), 'monotonic_deadline': budget.monotonic_end,
              'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              **original.cpu_check()}
    require(len(record['cpu_affinity']) == 16, 'Slurm CPU binding missing')
    publish(audit / f'preflight-{rank}.json', record)
    deadline = min(time.monotonic() + 180, budget.monotonic_end - NEW_FIT_GUARD)
    while not all((audit / f'preflight-{i}.json').exists() for i in range(4)):
        if stop.reason or time.monotonic() >= deadline:
            raise RuntimeError('CPU all-rank preflight gate stopped/timed out')
        time.sleep(.5)
    records = [read(audit / f'preflight-{i}.json') for i in range(4)]
    validate_gate(records, record['job_id'], record['step_id'], record['host'])
    require(all(r['worker_sha256'] == record['worker_sha256'] and
                r['monotonic_deadline'] == budget.monotonic_end for r in records), 'rank source/budget mismatch')
    return record


def claim_update(path, state):
    current = read(path)
    require(current.get('owner_token') == state['owner_token'], 'claim ownership changed')
    publish(path, state, replace=True)


def work_pair(original, plan, pair, audit, owner, budget, stop):
    lock_path = OUT / 'claims' / (pair['key'] + '.lock')
    require(not lock_path.is_symlink(), 'symlink claim lock refused')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 'locked'
        records, complete = inspect_pair(pair)
        if complete:
            return 'complete'
        require(not any((OUT / 'errors').glob(pair['key'] + '-*.json')), 'existing fit error; no automatic retry')
        if stop.reason or budget.remaining() <= NEW_FIT_GUARD:
            return 'paused'
        state_path = OUT / 'claims' / (pair['key'] + '.json')
        if state_path.exists():
            previous_bytes = state_path.read_bytes()
            previous = read(state_path)
            require(previous.get('method') == 'AutoGluon' and previous.get('dataset') == pair['dataset'], 'old claim identity mismatch')
            try:
                proof = terminal_evidence(previous)
            except (RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                publish(audit / f'skip-{pair["key"]}-r{owner["rank"]}.json', {'reason': str(exc), 'previous_claim': previous})
                return 'owner_unproven'
            publish(audit / f'prior-claim-{pair["key"]}-r{owner["rank"]}.json',
                    {'source_path': str(state_path), 'sha256': hashlib.sha256(previous_bytes).hexdigest(),
                     'original_text': previous_bytes.decode(), 'terminal_evidence': proof})
            require(state_path.read_bytes() == previous_bytes, 'old claim changed during owner checks')
        state = {'method': 'AutoGluon', 'dataset': pair['dataset'], 'key': pair['key'],
                 'job_id': owner['job_id'], 'step_id': owner['step_id'], 'rank': owner['rank'],
                 'pid': os.getpid(), 'host': owner['host'], 'owner_token': uuid.uuid4().hex,
                 'started': time.time(), 'state': 'running', 'auxiliary_campaign': CAMPAIGN,
                 'plan_id': PLAN, 'worker_sha256': owner['worker_sha256']}
        publish(state_path, state, replace=state_path.exists())
        try:
            for seed in SEEDS:
                if seed in records:
                    continue
                if stop.reason or budget.remaining() <= NEW_FIT_GUARD:
                    state.update(state='paused', reason=stop.reason or 'allocation_budget', finished=time.time())
                    claim_update(state_path, state)
                    return 'paused'
                state.update(seed=seed, heartbeat=time.time())
                claim_update(state_path, state)
                log = audit / f'fit-{pair["key"]}-{seed:02d}-r{owner["rank"]}.log'
                command = [sys.executable, '-B', str(Path(__file__).resolve()), 'seed', '--key', pair['key'], '--seed', str(seed)]
                rc, pause = run_seed(command, log, lock.fileno(), budget, stop)
                if pause:
                    # A just-published valid seed is retained; incomplete work is resumable.
                    inspect_pair(pair)
                    state.update(state='paused', reason=pause, finished=time.time(), descendants_reaped=True)
                    claim_update(state_path, state)
                    return 'paused'
                result_path = OUT / 'seeds' / pair['key'] / f'{seed:02d}.json'
                try:
                    require(rc == 0, f'frozen fitter exited {rc}')
                    validate_seed(read(result_path), pair, seed)
                except Exception as exc:
                    error = {'method': 'AutoGluon', 'dataset': pair['dataset'], 'seed': seed,
                             'exit_code': rc, 'error': str(exc), 'log': str(log), 'job_id': owner['job_id'],
                             'retry': False, 'plan_id': PLAN, 'campaign': CAMPAIGN}
                    publish(OUT / 'errors' / f'{pair["key"]}-{seed:02d}.json', error)
                    state.update(state='failed', finished=time.time(), descendants_reaped=True)
                    claim_update(state_path, state)
                    raise RuntimeError('fit failed; stopping this rank') from exc
            inspect_pair(pair)
            require(original.combine_pair(pair, plan), 'all seeds did not combine')
            require(inspect_pair(pair)[1], 'new aggregate failed strict validation')
            state.update(state='complete', finished=time.time(), descendants_reaped=True)
            claim_update(state_path, state)
            return 'complete'
        except BaseException as exc:
            # An unkillable descendant must not outlive an unlocked pair. Slurm
            # owns eventual cgroup teardown; never release this flock ourselves.
            try:
                unsafe = isinstance(exc, UnsafeCleanup) or bool(descendants(process_snapshot(), os.getpid()))
            except Exception:
                unsafe = True
            if unsafe:
                publish(audit / f'cleanup-fatal-{pair["key"]}-r{owner["rank"]}.json',
                        {'state': 'holding_lock_until_slurm_cleanup', 'claim': state})
                while True:
                    time.sleep(1)
            raise


def seed_entry(key, seed):
    require(os.getppid() == int(os.environ['T6_AG_PARENT_PID']), 'seed must be launched by owning wrapper')
    fd = int(os.environ['T6_AG_LOCK_FD'])
    expected = OUT / 'claims' / (key + '.lock')
    require(os.fstat(fd).st_ino == expected.stat().st_ino and os.fstat(fd).st_dev == expected.stat().st_dev,
            'seed inherited wrong pair lock')
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original, plan, pairs = load_original()
    pair = next(p for p in pairs if p['key'] == key)
    require(seed in SEEDS, 'invalid seed')
    original.cpu_check()
    original.one_seed(key, seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'seed'))
    parser.add_argument('--key')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args(argv)
    if args.action == 'seed':
        seed_entry(args.key, args.seed)
        return 0
    stop = Stop()
    for signum in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(signum, stop.signal)
    budget = EnvironmentBudget.from_environment()
    original, plan, pairs = load_original()
    job, step = os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID']
    require(re.fullmatch(r'\d+', job) and re.fullmatch(r'\d+', step), 'numeric job/step required')
    audit = OUT / 'auxiliary' / CAMPAIGN / f'j{job}-s{step}'
    audit.mkdir(parents=True, exist_ok=True)
    enable_subreaper()
    try:
        owner = preflight(original, budget, audit, stop)
    except Exception:
        if stop.reason or budget.remaining() <= NEW_FIT_GUARD:
            publish(audit / f'paused-preflight-{os.environ["SLURM_PROCID"]}.json',
                    {'state': 'paused', 'reason': stop.reason or 'allocation_budget'})
            return 0
        raise
    counts = {}
    try:
        for pair in sorted(pairs, key=lambda p: (p['work_size'], p['dataset'])):
            if stop.reason or budget.remaining() <= NEW_FIT_GUARD:
                break
            state = work_pair(original, plan, pair, audit, owner, budget, stop)
            counts[state] = counts.get(state, 0) + 1
            if state == 'paused':
                break
        publish(audit / f'finished-{owner["rank"]}.json',
                {'state': 'paused' if stop.reason or budget.remaining() <= NEW_FIT_GUARD or counts.get('paused') else 'queue_exhausted',
                 'counts': counts, 'reason': stop.reason, 'remaining_seconds': budget.remaining(), 'owner': owner})
        return 0
    except Exception as exc:
        publish(audit / f'failed-{owner["rank"]}.json', {'error': str(exc), 'counts': counts, 'owner': owner})
        return 1


if __name__ == '__main__':
    sys.exit(main())
