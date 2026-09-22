#!/usr/bin/env python3
"""One existing GPU: full-data small tasks, isolated attempts, owned deferrals.

No scientific manifest/worker or native arguments are changed. Canonical claims
exclude the frozen queues; guarded attempts publish only to private namespaces.
All canonical claims remain immutable, including after resource deferral. After
descendants are reaped, a durable audit records the retained claim; recovery
requires separately authorized quiescent queues. No claim is removed here.
The launcher supplies same-node EnvironmentBudget exports and a source-pinned
plan with two recent parent/GPU resource snapshots. This entry never submits.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import uuid

import tabfm_default_dispatch as q
from table6_restart_deadline import EnvironmentBudget, parse_duration, parse_fields

GIB = 1024**3
RESOURCE = {'parent_mem_gib': 64, 'step_mem_gib': 40, 'own_rss_gib': 32,
            'total_uid_rss_gib': 60, 'startup_other_gib': 20, 'threads': 4}
ELIGIBILITY = {'max_total_rows': 2048, 'max_features': 100}
CPU_COUNT = 64
STOP = None


class OperationalDeferral(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_checks(plan):
    records = plan['source_records']
    paths = [q.verify_file(record) for record in records]
    require(len(paths) == len(set(paths)), 'Duplicate runtime source identity')
    directory = Path(__file__).resolve().parent
    names = {'shared_foundation_sidecar.py', 'tabfm_default_dispatch.py',
             'tabfm_local_tmp.py', 'table6_restart_deadline.py', 'pfn_mitra_one.py', 'shared_eval_launch.py'}
    require({directory/name for name in names}.issubset(set(paths)), 'Unpinned runtime dependency')


def parent_fields(raw, plan):
    fields = parse_fields(raw)
    require(fields.get('JobId') == str(plan['parent_job_id']) and fields.get('JobState') == 'RUNNING'
            and fields.get('NumNodes') == '1' and fields.get('NodeList') == plan['node'], 'Parent allocation changed')
    require(fields.get('UserId', '').endswith('('+str(os.getuid())+')'), 'Parent allocation belongs to another uid')
    tres = dict(item.split('=', 1) for item in fields.get('AllocTRES', '').split(',') if '=' in item)
    require(tres.get('mem') == '64G' and int(tres.get('cpu', 0)) == CPU_COUNT
            and int(tres.get('gres/gpu', 0)) == 8, 'Requires the verified64CPU/64GiB/eight-GPU parent')
    require(parse_duration(fields['TimeLimit']) > parse_duration(fields['RunTime']), 'Parent allocation expired')
    return fields


def validate_probe(record, plan):
    require(record.get('complete') is True and str(record['parent']) == str(plan['parent_job_id'])
            and record['node'] == plan['node'], 'Wrong/incomplete probe parent/node')
    fields = record['job_fields']
    parent_fields(' '.join(k+'='+str(v) for k, v in fields.items()), plan)
    matches = [g for g in record['gpus'] if g.get('uuid', '').lower() == plan['gpu']['uuid'].lower()]
    require(len(matches) == 1 and matches[0]['pci'].lower() == plan['gpu']['pci'].lower()
            and matches[0]['hardware_idle'] is True and not matches[0]['foreign_fd_owner_pids']
            and matches[0]['busy_percent'] == 0 and 0 <= matches[0]['vram_used_bytes'] < 128*1024**2,
            'Selected GPU was not genuinely idle')
    rss = sum(p['rss_bytes'] for p in record['owned_processes'])
    require(rss == record['same_uid_rss_bytes'], 'Inconsistent same-uid memory proof')
    require(0 <= rss <= RESOURCE['startup_other_gib']*GIB, 'Parent lacks40GiB step plus4GiB headroom')


def load_plan(path):
    plan = q.read(path)
    require(plan.get('plan_id') == q.digest({k:v for k,v in plan.items() if k != 'plan_id'}), 'Runtime plan digest mismatch')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', plan['sidecar_id']) and str(plan['parent_job_id']).isdigit(), 'Unsafe runtime identity')
    require(plan['resource'] == RESOURCE and plan['eligibility'] == ELIGIBILITY, 'Resource/eligibility contract changed')
    require(re.fullmatch(r'[0-9a-fA-F]{16}', plan['gpu']['uuid']) and
            re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', plan['gpu']['pci']), 'Invalid physical GPU identity')
    source_checks(plan)
    proof = plan['proof']
    require(proof['gpu_idle_verified'] is True and proof['resources_available_verified'] is True, 'Missing verified resource proofs')
    samples = [q.read(q.verify_file(record)) for record in proof['sample_records']]
    require(len(samples) == 2 and 15 <= samples[1]['started_epoch']-samples[0]['epoch'] <= 900
            and proof['controller_finished_epoch'] == samples[1]['epoch'], 'Need two15s-separated controller resource samples')
    # Freshness is checked by the pinned launcher on the LOGIN clock. Never
    # subtract these timestamps from a compute-node clock; recheck GPU/RSS here.
    for sample in samples:
        parents = [record for record in sample['parents'] if str(record['parent']) == str(plan['parent_job_id'])]
        require(len(parents) == 1, 'Ambiguous/missing parent resource observation')
        validate_probe(parents[0], plan)
    campaigns = []
    for path in plan['campaign_paths']:
        man, tasks = q.load_campaign(path)
        campaigns.append((Path(path).resolve(), man, tasks))
    variants = [m.get('protocol', {}).get('variant') for _,m,_ in campaigns]
    require(variants == ['official16', 'budget32x8'] or
            (len(campaigns) == 1 and campaigns[0][1].get('name') == 'tabfm_defaults_standard681_20260922_v1'),
            'Only original dual TabSwift or one original TabFM campaign supported')
    require(len({m['output_root'] for _,m,_ in campaigns}) == len(campaigns), 'Campaign outputs collide')
    require(len({m['worker_python'] for _,m,_ in campaigns}) == 1, 'Worker runtimes differ')
    pinned = {Path(r['path']).resolve() for r in plan['source_records']}
    require({p for p,_,_ in campaigns}.issubset(pinned), 'Scientific manifest files must be runtime-plan pinned')
    if variants == ['official16', 'budget32x8']:
        require(Path(__file__).with_name('tabswift_dispatch.py').resolve() in pinned, 'Strict variant validator unpinned')
        import tabswift_dispatch  # Installs its unchanged strict32/8 validator.
        require(q.valid_result is tabswift_dispatch.validated_result, 'Strict original validator not installed')
    return plan, campaigns


def small_task(task, overlay=None):
    shape = task['row']
    if task['task_kind'] == 'regression':
        shape = (overlay or {}).get(str(task['dataset_index']), shape)
    dimensions = {k:shape.get(k) for k in ('train_rows', 'test_rows', 'features')}
    known = all(type(v) is int and v > 0 for v in dimensions.values())
    eligible = known and dimensions['train_rows']+dimensions['test_rows'] <= 2048 and dimensions['features'] <= 100
    return {'eligible': bool(eligible), 'dimensions': dimensions,
            'shape_source': 'pinned_regression_overlay' if task['task_kind'] == 'regression' and overlay and
                            str(task['dataset_index']) in overlay else 'frozen_manifest',
            'unknown_dimensions_skipped': not known, 'full_rows_unchanged': True}


def snapshot():
    import psutil
    mine = psutil.Process(os.getpid())
    owned = {mine.pid, *(p.pid for p in mine.children(recursive=True))}
    own = total = 0
    for proc in psutil.process_iter():
        try:
            if proc.uids().real != os.getuid():
                continue
            rss = proc.memory_info().rss
            total += rss
            if proc.pid in owned:
                own += rss
        except psutil.NoSuchProcess:
            pass
        # Any AccessDenied/inspection failure propagates, never undercounts.
    return {'own_tree_rss_bytes': own, 'same_uid_rss_bytes': total,
            'other_same_uid_rss_bytes': max(0, total-own), 'epoch': time.time()}


def guard(memory, budget, startup=False):
    if STOP is not None:
        raise OperationalDeferral('signal_'+str(STOP))
    if budget.remaining() <= 60:
        raise OperationalDeferral('allocation_deadline_shutdown_reserve')
    if memory['own_tree_rss_bytes'] > 32*GIB:
        raise OperationalDeferral('own_process_tree_exceeds32GiB')
    if memory['same_uid_rss_bytes'] > 60*GIB:
        raise OperationalDeferral('same_uid_node_RSS_exceeds60GiB')
    if startup and memory['other_same_uid_rss_bytes'] > 20*GIB:
        raise OperationalDeferral('startup_other_RSS_exceeds20GiB')


def signal_stop(number, _frame):
    global STOP
    STOP = number  # Never interrupt a claim or result write.


def subreaper():
    # Adopt orphaned model descendants, including children that create sessions.
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(36, 1, 0, 0, 0) == 0, 'Cannot install own-child subreaper')


def cleanup_children():
    """The single-purpose lane has no other children; reap before terminal audit."""
    import psutil
    own = psutil.Process(os.getpid())
    for sig in (signal.SIGTERM, signal.SIGKILL):
        children = own.children(recursive=True)
        for child in children:
            try:
                child.send_signal(sig)
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(children, timeout=10)
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
            except ChildProcessError:
                break
        if not own.children(recursive=True):
            return {'all_owned_children_reaped': True, 'epoch': time.time()}
    raise RuntimeError('Owned descendant cleanup unproven; retain canonical claim')


@contextmanager
def guarded_runtime(budget, events):
    original_rss, original_subprocess = q.process_rss, q.subprocess
    failed = [False]

    def inspect(_pid=None):
        memory = {}
        try:
            memory = snapshot()
            guard(memory, budget)
        except Exception as exc:
            failed[0] = True
            events.append({'reason': str(exc), 'inspection_type': type(exc).__name__, **memory})
            raise OperationalDeferral(str(exc)) from exc
        return memory['own_tree_rss_bytes']

    class Watched:
        def __init__(self, process):
            self.process = process

        def __getattr__(self, name):
            return getattr(self.process, name)

        def wait(self, timeout=None):
            if failed[0]:
                return self.process.wait(timeout=timeout)
            end = None if timeout is None else time.monotonic()+timeout
            while True:
                inspect()
                interval = 1 if end is None else max(0, min(1, end-time.monotonic()))
                try:
                    return self.process.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    if end is not None and time.monotonic() >= end:
                        raise

    class Proxy:
        def __getattr__(self, name):
            return getattr(original_subprocess, name)

        def Popen(self, *args, **kwargs):
            inspect()
            return Watched(original_subprocess.Popen(*args, **kwargs))

    q.process_rss, q.subprocess = inspect, Proxy()
    try:
        yield
    finally:
        q.process_rss, q.subprocess = original_rss, original_subprocess


def claim_identity(man, task, token):
    path = q.task_path(man, 'claims', task)
    require(not path.is_symlink(), 'Claim symlink forbidden')
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Claim changed while recording ownership')
    value = json.loads(raw)
    require(value.get('reservation_token') == token and value.get('manifest_id') == man['manifest_id'] and
            value.get('task_kind') == task['task_kind'] and value.get('dataset_index') == task['dataset_index'] and
            value.get('dataset') == task['dataset'], 'Not this exact lane reservation')
    return {'path': str(path), 'device': before.st_dev, 'inode': before.st_ino, 'size': before.st_size,
            'mtime_ns': before.st_mtime_ns, 'sha256': hashlib.sha256(raw).hexdigest(), 'token': token}


def attempt(plan, campaign_path, man, task, owner, budget, root, smoke=False):
    owner = dict(owner, manifest_id=man['manifest_id'])
    guard(snapshot(), budget)
    reservation = None
    if not smoke:
        token = uuid.uuid4().hex
        if not q.claim(man, task, dict(owner, reservation_token=token)):
            return {'state': 'already_claimed', 'canonical_touched': False}
        reservation = claim_identity(man, task, token)
    attempt_id = ('smoke' if smoke else 'formal')+'-'+task['task_kind']+'-'+str(task['dataset_index'])+'-'+uuid.uuid4().hex
    attempt_root = root/'attempts'/attempt_id
    temporary_man = dict(man, output_root=str(attempt_root))
    events, failure = [], None
    try:
        with guarded_runtime(budget, events):
            ok = q.launch(temporary_man, campaign_path, task, owner, smoke=smoke)
    except BaseException as exc:
        ok, failure = False, repr(exc)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            events.append({'reason': 'lane_interrupted', 'error': failure})
    # Cleanup failure intentionally leaves the exact canonical reservation held.
    cleanup = cleanup_children()
    output = q.task_path(temporary_man, 'smoke' if smoke else 'results', task)
    value = {'manifest_id': man['manifest_id'], 'plan_id': plan['plan_id'], 'task': {
             k:task[k] for k in ('task_kind', 'dataset_index', 'dataset')}, 'attempt_id': attempt_id,
             'attempt_output': str(output), 'reservation': reservation, 'cleanup': cleanup,
             'owner': owner, 'guard_events': events, 'exception': failure, 'finished_epoch': time.time()}
    if events:
        value.update(state='retained_resource_deferral' if reservation is not None else 'operationally_deferred',
                     reservation_retained=reservation is not None, automatic_retry=False,
                     recovery_requires_quiescent_authorization=reservation is not None)
        if reservation is not None:
            require(claim_identity(man, task, reservation['token']) == reservation, 'Reservation changed during deferral')
        deferred_path = root/'deferrals'/(attempt_id+'.json')
        q.atomic(man, deferred_path, value)
        return value
    if ok:
        result = q.valid_result(output, man, task)
        require(result['physical_gpu']['uuid'] == owner['uuid'] and
                result['physical_gpu']['pci_bus_id'] == owner['pci'], 'Completed result GPU mismatch')
        if not smoke:
            require(claim_identity(man, task, reservation['token']) == reservation, 'Reservation changed before publication')
            q.atomic(man, q.task_path(man, 'results', task), result)
        value['state'] = 'complete'
    else:
        value['state'] = 'model_error'
        if not smoke:
            require(claim_identity(man, task, reservation['token']) == reservation, 'Reservation changed before error publication')
            q.atomic(man, q.task_path(man, 'results', task), {'complete': False, 'status': 'error',
                     'manifest_id': man['manifest_id'], **value['task'], 'reason': 'isolated_model_attempt_failed',
                     'attempt_audit': str(root/'finished'/(attempt_id+'.json')), 'attempt_output': str(output),
                     'reservation_retained': True, 'resource_deferral': False})
    q.atomic(man, root/'finished'/(attempt_id+'.json'), value)
    return value


def bind_cpus(plan, gate_path):
    import psutil
    allowed = set(os.sched_getaffinity(0))
    if len(allowed) == 4:
        gate = q.read(gate_path)
        require(gate['plan_id'] == plan['plan_id'] and gate['job'] == str(plan['parent_job_id'])
                and gate['node'] == plan['node'] and gate['step'] == os.environ['SLURM_STEP_ID']
                and set(gate['actual_cpu_ids']) == allowed and len(gate['inherited_cpu_ids']) >= CPU_COUNT,
                'Four-core mask lacks matching current launcher resource gate')
        return {'selected': sorted(allowed), 'allowed_before': sorted(allowed), 'actual_threads': 4,
                'selection_source': 'source-pinned launcher two-sample idle selection', 'launcher_gate': str(gate_path),
                'launcher_gate_sha256': file_sha(gate_path), 'inherited_parent_access_count': len(gate['inherited_cpu_ids'])}
    require(len(allowed) >= CPU_COUNT, 'Need launcher-verified four-core mask or full parent CPU access')
    frames = [psutil.cpu_percent(interval=.5, percpu=True) for _ in range(2)]
    require(all(all(i < len(f) and isinstance(f[i], (int,float)) and math.isfinite(f[i]) and 0 <= f[i] <= 100
                    for i in allowed) for f in frames), 'Invalid CPU observations')
    eligible = [i for i in allowed if all(f[i] < 25 for f in frames)]
    require(len(eligible) >= 4, 'Fewer than four observed idle CPUs')
    selected = sorted(eligible, key=lambda i:(max(f[i] for f in frames), sum(f[i] for f in frames), i))[:4]
    os.sched_setaffinity(0, selected)
    require(set(os.sched_getaffinity(0)) == set(selected), 'CPU binding failed')
    return {'selected': sorted(selected), 'allowed_before': sorted(allowed), 'frames': frames, 'actual_threads': 4}


def gpu_idle(plan):
    root = Path('/sys/bus/pci/devices')/plan['gpu']['pci']
    record = {'uuid': (root/'unique_id').read_text().strip().lower(),
              'busy': int((root/'gpu_busy_percent').read_text()), 'vram': int((root/'mem_info_vram_used').read_text())}
    require(record['uuid'] == plan['gpu']['uuid'].lower() and record['busy'] == 0 and
            0 <= record['vram'] < 128*1024**2, 'Selected GPU no longer idle')
    return record


def run(plan, campaigns, plan_path):
    global STOP
    STOP = None
    require(os.environ.get('SLURM_JOB_ID') == str(plan['parent_job_id']) and os.environ.get('SLURM_NTASKS') == '1'
            and os.environ.get('SLURM_PROCID') == os.environ.get('SLURM_LOCALID') == '0'
            and os.environ.get('SLURM_STEP_ID', '').isdigit(), 'Actual one-rank child step required')
    require(socket.gethostname().split('.')[0] == plan['node'], 'Wrong physical node')
    budget = EnvironmentBudget.from_environment()
    require(budget.monotonic_end is not None and 120 < budget.remaining() <= 6901,
            'Require inherited same-node controller-derived max2h-minus300s budget')
    parent = subprocess.run(['scontrol', 'show', 'job', '-o', str(plan['parent_job_id'])],
                            capture_output=True, text=True, timeout=30, check=True)
    parent_fields(parent.stdout, plan)
    memory = snapshot(); guard(memory, budget, startup=True)
    os.nice(19)
    cpu = bind_cpus(plan, Path(plan_path).parent/'node-resource-gate.json')
    os.environ['ROCR_VISIBLE_DEVICES'] = 'GPU-'+plan['gpu']['uuid']
    for key in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL','PYTHONPATH','PYTHONHOME'):
        os.environ.pop(key, None)
    os.environ.update(EXPECTED_GPU_UUID=plan['gpu']['uuid'], EXPECTED_GPU_PCI_BUS_ID=plan['gpu']['pci'],
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
                      PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    roots = [Path(man['output_root'])/'sidecars'/plan['sidecar_id'] for _,man,_ in campaigns]
    for root, (_,man,_) in zip(roots, campaigns):
        q.atomic(man, root/'lane_claim.json', {'plan_id': plan['plan_id'], 'pid': os.getpid(), 'epoch': time.time()})
    from tabfm_local_tmp import activate
    runtime_tmp = activate(campaigns[0][1], plan, roots[0]/'runtime_environment')
    idle = gpu_idle(plan)
    import torch
    from pfn_mitra_one import gpu_identity
    gpu_idle(plan)  # Last empty-device assertion BEFORE our own context exists.
    gpu = gpu_identity(torch)
    require(gpu['uuid'] == plan['gpu']['uuid'] and gpu['pci_bus_id'] == plan['gpu']['pci'], 'Actual physical GPU mismatch')
    subreaper()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, signal_stop)
    owner = {'job': str(plan['parent_job_id']), 'step': os.environ['SLURM_STEP_ID'], 'rank': 0,
             'node': plan['node'], 'uuid': gpu['uuid'], 'pci': gpu['pci_bus_id'], 'plan_id': plan['plan_id'],
             'sidecar_id': plan['sidecar_id'], 'one_real_lane': True}
    audits, overlays = [], []
    for root, (_,man,tasks) in zip(roots, campaigns):
        overlay = {}
        raw_overlay = plan.get('regression_shape_overlays', {}).get(man['manifest_id'])
        if raw_overlay is not None:
            require(Path(__file__).with_name('tabfm_regression_shapes.py').resolve() in
                    {Path(r['path']).resolve() for r in plan['source_records']}, 'Regression shape helper unpinned')
            from tabfm_regression_shapes import validate
            overlay = validate(raw_overlay, man, tasks)
        overlays.append(overlay)
        eligibility = [{**{k:t[k] for k in ('task_kind','dataset_index','dataset')}, **small_task(t, overlay)} for t in tasks]
        audits.append(eligibility)
        q.atomic(man, root/'preflight.json', {'plan_id': plan['plan_id'], 'owner': owner, 'memory': memory,
                 'cpu_binding': cpu, 'gpu_idle': idle, 'physical_gpu': gpu, 'TMPDIR': runtime_tmp,
                 'parent_scontrol': parent.stdout, 'resource': RESOURCE,
                 'step_inherited_cpu_access': 64, 'step_inherited_gpu_access': 8, 'actual_model_gpus': 1,
                 'budget_monotonic_end': budget.monotonic_end, 'eligibility': eligibility})
    summary = {'plan_id': plan['plan_id'], 'smokes': [], 'formal': [], 'state': 'starting'}
    try:
        for root, (path,man,tasks) in zip(roots, campaigns):
            checks = []
            for task in q.smoke_tasks(man, tasks):
                result = attempt(plan, path, man, task, owner, budget, root, smoke=True)
                summary['smokes'].append(result)
                require(result['state'] == 'complete', 'Current-lane full-data smoke incomplete; no formal work')
                checks.append(result)
            q.atomic(man, root/'smoke_gate.json', {'plan_id': plan['plan_id'], 'manifest_id': man['manifest_id'],
                     'owner': owner, 'four_current_lane_full_data_smokes': checks})
        queues = [sorted((t for t in tasks if small_task(t, overlay)['eligible']),
                         key=lambda t:(q.work_size(t),t['task_kind'],t['dataset_index']))
                  for (_,_,tasks),overlay in zip(campaigns, overlays)]
        # Alternate protocols while both have tasks; no differing membership truncation.
        for index in range(max(map(len, queues), default=0)):
            for queue, root, (path,man,_) in zip(queues, roots, campaigns):
                if index >= len(queue):
                    continue
                if budget.remaining() <= 120 or STOP is not None:
                    raise OperationalDeferral('no_new_task_window_or_signal')
                task = queue[index]
                output = q.task_path(man, 'results', task)
                if output.exists():
                    if q.read(output).get('complete') is True:
                        q.valid_result(output, man, task)
                    require(q.task_path(man, 'claims', task).exists(), 'Existing canonical result lacks claim')
                    continue
                result = attempt(plan, path, man, task, owner, budget, root)
                summary['formal'].append(result)
                if result['state'] in {'operationally_deferred', 'retained_resource_deferral'}:
                    raise OperationalDeferral(result['state']+'; no automatic retry or claim release')
        summary['state'] = 'no_unclaimed_eligible_work'
    except OperationalDeferral as exc:
        summary.update(state='operationally_deferred', reason=str(exc))
    except BaseException as exc:
        summary.update(state='error', reason=repr(exc))
        raise
    finally:
        summary['cleanup'] = cleanup_children()
        summary['finished_epoch'] = time.time()
        for root, (_,man,_) in zip(roots, campaigns):
            q.atomic(man, root/'lane_finished.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(argv)
    require(__debug__, 'Optimized Python unsupported')
    plan, campaigns = load_plan(args.plan)
    print(json.dumps(run(plan, campaigns, args.plan.resolve()), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
