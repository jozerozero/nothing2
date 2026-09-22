#!/usr/bin/env python3
"""One audited existing-allocation lane; frozen missing190 science is unchanged.

The external launcher proves allocation ownership/resources and exports a
same-node, parent-derived monotonic budget capped at two hours less 300s.
This wrapper never submits/cancels allocations, changes a plan/model, steals a
pair lock, or changes the frozen Runtime.sources identity used by other workers.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
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

GIB = 1024**3
SCIENTIFIC_PLAN_ID = 'f35686cd72710801e14ccf00e1d44f108d87627ee404927a178efc9c32c86066'
REQUIRED_SOURCES = ('table6_existing_gap190.py', 'table6_missing190_worker.py',
                    'table6_missing190_fit.py', 'table6_restart_deadline.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_file(record):
    path = Path(record['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'invalid pinned source')
    before = path.stat()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'source changed while reading')
    require(actual == record['sha256'], 'pinned source SHA changed: '+str(path))
    for key, value in (('size_bytes', after.st_size), ('mtime_ns', after.st_mtime_ns)):
        require(key not in record or record[key] == value, 'pinned source metadata changed')
    return path.resolve()


def load_plan(path):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink(), 'runtime plan must be absolute and regular')
    plan = json.loads(path.read_text())
    require(plan.get('plan_id') == digest({k:v for k,v in plan.items() if k != 'plan_id'}),
            'runtime plan digest mismatch')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', plan['sidecar_id']) and
            re.fullmatch(r'[0-9]+', str(plan['parent_job_id'])), 'invalid sidecar/parent identity')
    require(plan['cpus'] == 16 and plan['mem_gib'] == 40 and plan['parent_mem_gib'] == 64 and
            plan['max_step_seconds'] == 7200, 'requires the audited16CPU/40GiB/64GiB/2h contract')
    require(isinstance(plan.get('proof'), dict) and plan['proof'], 'external ownership/resource proof absent')
    require(plan['node'] and re.fullmatch(r'[0-9a-fA-F]+', plan['gpu']['uuid']) and
            re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', plan['gpu']['pci']),
            'invalid physical GPU identity')
    records = plan['source_records']
    verified = [verify_file(record) for record in records]
    require(len(verified) == len(set(verified)), 'duplicate source pins')
    here = Path(__file__).resolve().parent
    require({here/name for name in REQUIRED_SOURCES} <= set(verified), 'missing runtime/fit/deadline pins')
    scientific_path = verify_file(plan['scientific_plan'])
    scientific = json.loads(scientific_path.read_text())
    require(scientific['plan_id'] == SCIENTIFIC_PLAN_ID and
            scientific['plan_id'] == digest({k:v for k,v in scientific.items() if k != 'plan_id'}),
            'wrong frozen missing190 scientific plan')
    require(Path(scientific['output_root']).is_absolute(), 'scientific output root must be absolute')
    return plan, scientific_path, scientific


def verify_node(plan, *, environ=None, hostname=None, affinity=None, gpu_reader=None):
    env = os.environ if environ is None else environ
    host = socket.gethostname() if hostname is None else hostname
    require(env.get('SLURM_JOB_ID') == str(plan['parent_job_id']) and
            env.get('SLURM_NTASKS') == '1' and env.get('SLURM_PROCID') == env.get('SLURM_LOCALID') == '0' and
            re.fullmatch(r'[0-9]+', env.get('SLURM_STEP_ID', '')), 'requires a genuine one-rank parent job step')
    require(host == plan['node'], 'wrong allocation node')
    cores = set(os.sched_getaffinity(0) if affinity is None else affinity)
    require(len(cores) == 16, 'sidecar must be bound to exactly16 proved CPU cores')
    if 'cpu_ids' in plan:
        require(cores == set(plan['cpu_ids']), 'CPU binding differs from ownership proof')
    expected = plan['gpu']['uuid'].lower().removeprefix('gpu-')
    require(env.get('ROCR_VISIBLE_DEVICES', '').lower() == 'gpu-'+expected,
            'single physical ROCr UUID binding is mandatory')
    require(all(env.get(key) is None for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')),
            'remove secondary GPU aliases before this runtime')
    reader = gpu_reader or (lambda pci: (Path('/sys/bus/pci/devices')/pci/'unique_id').read_text().strip())
    require(reader(plan['gpu']['pci']).lower() == expected, 'PCI/UUID ownership mismatch')
    budget = EnvironmentBudget.from_environment(env, hostname=host)
    require(budget.monotonic_end is not None and env.get('JOB_BUDGET_JOB_ID') == env['SLURM_JOB_ID'],
            'verified parent-derived same-node monotonic budget required')
    require(120 < budget.remaining() <= 6901, 'insufficient or excessive two-hour safe budget')
    return budget


def rss_snapshot():
    """Same-UID node RSS conservatively overcounts, never understates parent RSS."""
    import psutil
    parent = psutil.Process(os.getpid())
    own_ids = {parent.pid, *(child.pid for child in parent.children(recursive=True))}
    own = same_uid = 0
    for proc in psutil.process_iter():
        try:
            if proc.uids().real != os.getuid():
                continue
            rss = proc.memory_info().rss
            same_uid += rss
            if proc.pid in own_ids:
                own += rss
        except psutil.NoSuchProcess:
            pass
        # An unreadable same-user process must fail closed, not undercount.
    return {'own_tree_rss_bytes': own, 'same_uid_node_rss_bytes': same_uid,
            'parent_rss_guard_basis': 'conservative whole-node same-UID process RSS', 'epoch': time.time()}


def check_rss(snapshot):
    for key in ('own_tree_rss_bytes', 'same_uid_node_rss_bytes'):
        require(type(snapshot[key]) is int and snapshot[key] >= 0, 'invalid RSS observation')
    require(snapshot['own_tree_rss_bytes'] <= 34*GIB, 'resource_guard: own aggregate RSS exceeds34GiB')
    require(snapshot['same_uid_node_rss_bytes'] <= 60*GIB, 'resource_guard: parent-conservative RSS exceeds60GiB')


def publish(root, name, value):
    root.mkdir(parents=True, exist_ok=True)
    path = root/(name+'.json')
    require(not root.is_symlink() and not path.is_symlink(), 'sidecar audit symlink refused')
    temporary = root/('.'+name+'-'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def audit_root(plan, scientific):
    root = Path(scientific['output_root']).resolve()
    path = root/'sidecars'/plan['sidecar_id']
    require(not path.is_symlink() and path.resolve().is_relative_to(root), 'sidecar audit escaped output')
    return path


def setup_runtime(plan, scientific_path, budget, audit):
    from table6_missing190_worker import Runtime
    runtime = Runtime(scientific_path)
    original_sources = dict(runtime.sources)
    short = runtime.short

    class GuardedBudget(short.Budget):
        def remaining(self):
            return budget.remaining()

        def check(self, new_fit=False):
            resource_flag = audit/('resource-stop-step-'+os.environ['SLURM_STEP_ID']+'.json')
            if resource_flag.exists():
                self.stop_reason = 'sidecar_resource_guard: '+json.loads(resource_flag.read_text())['reason']
            super().check(new_fit)
            try:
                check_rss(rss_snapshot())
            except Exception as exc:
                self.request_stop('sidecar_resource_guard: '+str(exc))
                super().check(new_fit)

    short.BUDGET = GuardedBudget(budget.hard_end_epoch)
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, short.signal_stop)
    original_execute = short.execute

    def execute(*args, **kwargs):
        try:
            return original_execute(*args, **kwargs)
        except short.BudgetStop as exc:
            if str(exc).startswith('sidecar_resource_guard:'):
                # Original execute has already reaped all owned fits/scratch.
                # A smaller sidecar's RAM guard must not permanently defer a
                # valid method for the full-memory standalone208407 campaign.
                publish(audit, 'resource-pause-'+str(time.time_ns()),
                        {'reason': str(exc), 'fit_elapsed_seconds': getattr(exc, 'fit_elapsed_seconds', None),
                         'original_cleanup_completed': short._ACTIVE is None})
                raise short.BudgetStop(str(exc)) from exc
            raise

    short.execute = execute
    require(runtime.sources == original_sources, 'sidecar must not invalidate existing result source identities')
    return runtime


def action(plan, scientific_path, scientific, budget, mode, method=None):
    audit = audit_root(plan, scientific)
    runtime = setup_runtime(plan, scientific_path, budget, audit)
    try:
        runtime.short.budget().check()
        if mode == 'preflight':
            record = runtime.preflight('gpu', 16)
            require(record['physical_gpu'].lower() == plan['gpu']['pci'].lower(), 'actual Torch GPU differs from proof')
            return record
        if mode == 'smoke':
            return runtime.smoke(method)  # Exact native CPU16/GPU8 request/cache.
        if mode == 'gate':
            return runtime.gate(1)
        if mode == 'worker':
            summary = runtime.worker('gpu')  # Same nonblocking pair locks/study/seed outputs.
            return {'state': 'paused' if summary.get('paused') or summary.get('deferred') else 'worker_finished',
                    'summary': summary}
        raise ValueError(mode)
    except runtime.short.BudgetStop as exc:
        runtime.short.lifecycle('paused', phase='existing_sidecar_'+mode, reason=str(exc))
        return {'state': 'paused', 'reason': str(exc)}


def stop_owned(child):
    """The leader handles cleanup first; emergency signals touch owned groups only."""
    if child.poll() is not None:
        child.wait(); return
    child.send_signal(signal.SIGTERM)
    try:
        child.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        import psutil
        descendants = psutil.Process(child.pid).children(recursive=True)
        groups = {child.pid}
        for proc in descendants:
            try:
                groups.add(os.getpgid(proc.pid))
            except ProcessLookupError:
                pass
        require(os.getpgrp() not in groups, 'refuse signalling the owning allocation process group')
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait(timeout=15)


def supervise(path, plan, scientific, budget):
    from table6_missing190_fit import METHODS
    audit = audit_root(plan, scientific)
    root = Path(scientific['output_root'])
    lock = root/'sidecars'/'_locks'/f"parent-{plan['parent_job_id']}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    require(not lock.is_symlink(), 'sidecar owner lock is a symlink')
    # Existing metadata is keyed job/rank, not step. One sidecar per parent
    # prevents same-parent rank0 gates/lifecycle records from overwriting peers.
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopped, active = [], [None]
        def signal_stop(signum, frame):
            stopped.append(signum)
            child = active[0]
            if child is not None and child.poll() is None:
                try: child.send_signal(signum)
                except ProcessLookupError: pass
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal_stop)
        actions = [('preflight', None), *(('smoke', method) for method in METHODS), ('gate', None), ('worker', None)]
        for mode, method in actions:
            if stopped or budget.remaining() <= 180:
                return {'state': 'paused', 'reason': 'signal_or_budget_before_'+mode}
            check_rss(rss_snapshot())
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--plan', str(path), '--action', mode]
            if method is not None: command += ['--method', method]
            child = subprocess.Popen(command, start_new_session=True)
            active[0] = child
            try:
                if stopped: child.send_signal(stopped[-1])
                while child.poll() is None:
                    if stopped or budget.remaining() <= 150:
                        stop_owned(child)
                        return {'state': 'paused', 'reason': 'signal_or_budget_during_'+mode}
                    # Child's own BudgetStop guard is authoritative for safe
                    # HPO pause semantics. Parent observes startup overhead too.
                    try:
                        check_rss(rss_snapshot())
                    except Exception as exc:
                        flag_name = 'resource-stop-step-'+os.environ['SLURM_STEP_ID']
                        if not (audit/(flag_name+'.json')).exists():
                            publish(audit, flag_name, {'reason': str(exc)})
                        child.send_signal(signal.SIGTERM)
                        stop_owned(child)
                        publish(audit, 'supervisor-resource-stop-'+str(time.time_ns()), {'reason': str(exc)})
                        return {'state': 'paused', 'reason': str(exc)}
                    try: child.wait(timeout=1)
                    except subprocess.TimeoutExpired: pass
                child.wait()
                if child.returncode == 75:
                    return {'state': 'paused', 'reason': 'bounded_child_pause_during_'+mode}
                require(child.returncode == 0, 'sidecar action failed: '+mode+'/'+str(method))
            finally:
                if child.poll() is None: stop_owned(child)
                active[0] = None
        return {'state': 'worker_finished', 'note': 'lane exit does not imply all190 pairs complete'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--action', choices=('preflight', 'smoke', 'gate', 'worker'))
    parser.add_argument('--method')
    args = parser.parse_args(argv)
    require(__debug__, 'Python optimization would disable inherited scientific validators')
    plan, scientific_path, scientific = load_plan(args.plan)
    budget = verify_node(plan)
    audit = audit_root(plan, scientific)
    receipt = {'runtime_plan_id': plan['plan_id'], 'scientific_plan_id': scientific['plan_id'],
               'parent_job_id': plan['parent_job_id'], 'step': os.environ['SLURM_STEP_ID'],
               'node': socket.gethostname(), 'pid': os.getpid(), 'action': args.action or 'supervisor',
               'hard_end_epoch': budget.hard_end_epoch, 'monotonic_end': budget.monotonic_end,
               'source_records': plan['source_records'], 'proof_digest': digest(plan['proof'])}
    publish(audit, 'start-'+str(time.time_ns())+'-'+str(os.getpid()), receipt)
    result = (action(plan, scientific_path, scientific, budget, args.action, args.method) if args.action else
              supervise(args.plan, plan, scientific, budget))
    publish(audit, 'finish-'+str(time.time_ns())+'-'+str(os.getpid()), {**receipt, 'result': result})
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    if args.action and result.get('state') == 'paused':
        raise SystemExit(75)


if __name__ == '__main__':
    main()
