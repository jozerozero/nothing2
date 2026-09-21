#!/usr/bin/env python3
"""Independent, immutable strict457 Mitra classification / 32-member campaign.

prepare freezes existing caches without regenerating data; plan requires two
fresh capacity observations and explicit existing parents. launch is opt-in and
starts only child steps, one isolated physical GPU per parent. No parent jobs,
historical results, model weights, or running regression dispatchers are changed.
"""
import argparse
from collections import Counter
from datetime import datetime
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
import threading
import time
import uuid

from eval_dispatch import atomic, check_idle, gpu, owned_rss, read, stop_own, GIB, IDLE_MAX


BASE = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
FT = BASE / 'stage/reg_loop3_step22175_finetune50_20260921_v1'
REPO = FT / 'repo'
ROOT = FT / 'mitra_class32_457_20260921_v1'
BENCHMARK = BASE / 'evaluation/benchmark8_seven_suites_20260820_v1'
MITRA = BASE / 'stage/mitra_all_classification_regression_20260823_v3'
PARENTS = {'196092', '196093', '200798', '200797', '204828', '204827',
           '204826', '206117', '206116', '194259', '194181', '194180'}
COUNTS = {'talent': 200, 'BCCO': 106, 'CTR23': 0, 'OpenML-CC18': 62,
          'PFN': 29, 'TabArena': 33, 'TabZilla': 27}
WEIGHTS_MANIFEST_SHA = '228008da42a9bef329872735200175f2b513f945e425abdb7074a94556a8330f'
BENCHMARK_MANIFEST_SHA = '1d19ef254208a3ccf83a7f83e9fe4c171854573d23ff378b48d7e8c9d0aacd0f'
SCRIPTS = ('mitra_class32_dispatch.py', 'mitra_class32_one.py', 'eval_dispatch.py',
           'pfn_mitra_one.py', 'eval_one.py')
STAGE_SCRIPTS = ('mitra_common.py', 'hierarchical_mitra.py')
STAGE_SHA = {'mitra_common.py': '1f1b81e0f757026a2b94ae1b6f205776f2c3d695581f2278b79d092afc3340c7',
             'hierarchical_mitra.py': '12a957d697afb5975d8e826ceebfa420ac85bdfae500721bf81f9fb82060758b'}
JOB_NAME = 'mitrac32'


class SafetyError(RuntimeError):
    """Stop this child campaign; never displace another workload."""


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    out = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            out.update(chunk)
    return out.hexdigest()


def file_identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    value = {'path': str(path), 'size_bytes': before.st_size,
             'mtime_ns': before.st_mtime_ns, 'sha256': file_sha(path)}
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'File changed during freeze: ' + str(path))
    return value


def check_identity(record, hash_content=True):
    path = Path(record['path']).resolve(strict=True)
    require(str(path) == record['path'], 'Frozen path resolution changed')
    stat = path.stat()
    require(stat.st_size == record['size_bytes'] and stat.st_mtime_ns == record['mtime_ns'],
            'Frozen file metadata changed: ' + str(path))
    if hash_content:
        require(file_sha(path) == record['sha256'], 'Frozen file content changed: ' + str(path))


def cache_path(dataset):
    require(Path(dataset).name == dataset and dataset not in ('.', '..'), 'Invalid dataset name')
    real_dir = (BENCHMARK / 'data' / dataset).resolve(strict=True)
    suffix = hashlib.md5(str(real_dir).encode('utf-8')).hexdigest()[:12]
    return BENCHMARK / 'cache' / f'{dataset}__{suffix}.npz'


def validate_manifest(man):
    require(man['manifest_id'] == digest({k: v for k, v in man.items() if k != 'manifest_id'}),
            'Frozen manifest digest mismatch')
    rows = man['rows']
    require(man['membership_count'] == len(rows) == 457 and
            [r['dataset_index'] for r in rows] == list(range(457)), 'Expected all457 ordered memberships')
    require(len({r['dataset'] for r in rows}) == 457, 'Duplicate membership name')
    actual = Counter(r['suite'] for r in rows)
    require(set(actual) <= set(COUNTS) and {suite: actual[suite] for suite in COUNTS} == COUNTS,
            'Classification suite membership changed')
    require(man['n_estimators'] == 32 and man['seed'] == 0, 'Classification recipe changed')
    for row in rows:
        require(row['input_fingerprint'] == digest(row['cache']), 'Cache fingerprint mismatch')
    return man


def prepare(manifest_path):
    benchmark_id = file_identity(BENCHMARK / 'benchmark_manifest.json')
    require(benchmark_id['sha256'] == BENCHMARK_MANIFEST_SHA, 'Audited historical benchmark manifest changed')
    source = read(benchmark_id['path'])
    require(source.get('complete') is True and source.get('protocol_validation') is True,
            'Historical benchmark not complete/validated')
    require(source['suite_counts'] == COUNTS and source['classification_memberships'] == 457,
            'Historical classification membership changed')
    weights_id = file_identity(MITRA / 'weights_manifest.json')
    require(weights_id['sha256'] == WEIGHTS_MANIFEST_SHA, 'Historical Mitra inventory changed')
    classifier = read(weights_id['path'])['classifier']
    require(classifier['repo_id'] == 'autogluon/mitra-classifier' and
            classifier['autogluon_version'] == '1.5.0', 'Wrong pretrained classifier')
    rows = []
    for index, source_row in enumerate(source['rows']):
        require(source_row['position'] == index, 'Historical membership order changed')
        path = cache_path(source_row['dataset'])
        require(path.resolve(strict=True) == Path(source_row['cache_path']).resolve(strict=True),
                'Historical cache name/path does not match real-directory MD5 contract')
        cache = file_identity(path)
        row = {'dataset_index': index, 'dataset': source_row['dataset'], 'suite': source_row['suite'],
               'cache': cache, 'input_fingerprint': digest(cache)}
        for key in ('position', 'train_rows', 'test_rows', 'features', 'classes', 'class_labels',
                    'split', 'source_name', 'source_path', 'feature_audit', 'target_audit'):
            if key in source_row:
                row[key] = source_row[key]
        require(source_row.get('finite') is True and row['classes'] >= 2 and
                row['train_rows'] > 0 and row['test_rows'] > 0, 'Invalid historical cache metadata')
        rows.append(row)
    check_identity(benchmark_id)
    man = {'schema_version': 1, 'protocol': 'historical-strict457-mitra-classifier-actual32-v1',
           'task_kind': 'classification', 'membership_count': 457, 'suite_counts': COUNTS,
           'n_estimators': 32, 'seed': 0, 'mitra_stage': str(MITRA),
           'weights_manifest': weights_id, 'class_benchmark_manifest': benchmark_id,
           'stage_dependencies': {name: file_identity(MITRA / name) for name in STAGE_SCRIPTS},
           'classification_weight': classifier,
           'data_policy': 'Read-only historical cached support/test, no resplit or cache regeneration',
           'full_test_split': True, 'fine_tune': False, 'rows': rows}
    man['manifest_id'] = digest(man)
    require({name: record['sha256'] for name, record in man['stage_dependencies'].items()} == STAGE_SHA,
            'Historical classifier construction/hierarchy helpers changed')
    validate_manifest(man)
    atomic(manifest_path, man, True)
    print(json.dumps({'manifest': str(manifest_path), 'manifest_id': man['manifest_id'],
                      'membership_count': 457, 'suite_counts': COUNTS}), flush=True)


def check_remaining_time(fields, requested_seconds, now=None):
    try:
        end = datetime.fromisoformat(fields['EndTime']).timestamp()
    except (KeyError, ValueError) as exc:
        raise RuntimeError('Cannot verify parent remaining time from EndTime') from exc
    remaining = end - (time.time() if now is None else now)
    require(remaining >= requested_seconds + 300, 'Parent lacks requested child duration plus300seconds margin')
    return remaining


def runtime_limits(hours):
    require(isinstance(hours, int) and 1 <= hours <= 72, '--hours must be an integer from1to72')
    hard_seconds = hours * 3600
    soft_seconds = hard_seconds - 300
    return {'hard_step_limit': f'{hours:02d}:00:00', 'hard_step_limit_seconds': hard_seconds,
            'soft_step_limit_seconds': soft_seconds,
            'single_dataset_limit_seconds': min(6 * 3600, soft_seconds - 60)}


def parent_fields(parent, requested_seconds=None):
    require(parent in PARENTS, 'Parent outside explicit user-owned inventory')
    raw = subprocess.check_output(['scontrol', 'show', 'job', parent, '-o'], text=True)
    fields = dict(re.findall(r'(\w+)=(\S+)', raw))
    require(fields['JobState'] == 'RUNNING' and fields['NumNodes'] == '1' and
            fields['UserId'].startswith('guangyi.chen('), 'Parent ownership/state changed')
    require(re.search(r'(?:^|,)gres/gpu=8(?:,|$)', fields['AllocTRES']), 'Parent must own eight GPUs')
    memory = re.search(r'(?:^|,)mem=([0-9.]+)([KMGTP])(?:,|$)', fields['AllocTRES'])
    require(memory is not None, 'Missing parent memory reservation')
    memory_bytes = float(memory[1]) * 1024 ** ('KMGTP'.index(memory[2]) + 1)
    require(memory_bytes >= 64 * GIB and int(fields['NumCPUs']) >= 4,
            'Parent reservation insufficient for one40GiB evaluator plus safety margin')
    if requested_seconds is not None:
        check_remaining_time(fields, requested_seconds)
    return fields


def runtime():
    python = MITRA / 'venv/bin/python'
    require(python.is_file(), 'Existing Mitra runtime absent')
    env = dict(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
               PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', HF_HOME=str(MITRA / 'hf_cache'),
               PYTHONPATH=str(REPO) + ':' + str(MITRA))
    return str(python), env


def shard_rows(rows, parents, indices=None):
    require(parents and len(set(parents)) == len(parents) and set(parents) <= PARENTS,
            'Supply unique explicit eligible parents')
    selected = list(range(457)) if indices is None else list(indices)
    require(selected and len(set(selected)) == len(selected) and set(selected) <= set(range(457)),
            'Invalid/duplicate subset indices')
    require(len(parents) <= len(selected), 'More selected parents than dataset rows')
    by_index = {r['dataset_index']: r for r in rows}
    require(len(by_index) == 457 and set(by_index) == set(range(457)), 'Incomplete frozen membership')
    ordered = sorted(selected, key=lambda i: (by_index[i]['cache']['size_bytes'], i))
    shards = [{'parent': parent, 'lane_id': f'mitra-class32-s{rank}',
               'dataset_indices': ordered[rank::len(parents)]}
              for rank, parent in enumerate(parents)]
    assigned = [i for shard in shards for i in shard['dataset_indices']]
    require(len(assigned) == len(set(assigned)) == len(selected) and set(assigned) == set(selected),
            'Dataset shard overlap or coverage failure')
    return ordered, shards


def smoke_indices(rows):
    groups = [lambda r: r['classes'] == 2, lambda r: 2 < r['classes'] <= 10,
              lambda r: r['classes'] > 10]
    require(all(any(select(r) for r in rows) for select in groups), 'Smoke class categories missing')
    return [min((r for r in rows if select(r)), key=lambda r: (r['cache']['size_bytes'], r['dataset_index']))
            ['dataset_index'] for select in groups]


def build_plan(manifest_path, plan_path, parents, indices=None, smoke=False, hours=2):
    man = validate_manifest(read(manifest_path))
    limits = runtime_limits(hours)
    require(not (indices is not None and smoke), 'Use either --indices or --smoke')
    if smoke:
        indices = smoke_indices(man['rows'])
    ordered, shards = shard_rows(man['rows'], parents, indices)
    for record in (man['weights_manifest'], man['class_benchmark_manifest'], *man['stage_dependencies'].values()):
        check_identity(record)
    require(man['weights_manifest']['sha256'] == WEIGHTS_MANIFEST_SHA, 'Wrong Mitra inventory')
    require(man['class_benchmark_manifest']['sha256'] == BENCHMARK_MANIFEST_SHA and
            {name: record['sha256'] for name, record in man['stage_dependencies'].items()} == STAGE_SHA,
            'Audited historical benchmark or helper identity changed')
    for row in man['rows']:
        check_identity(row['cache'], hash_content=False)
    capacity_paths = [ROOT / 'capacity_a.json', ROOT / 'capacity_b.json']
    observations = [read(path) for path in capacity_paths]
    first, second = observations
    require(second['epoch'] - first['epoch'] >= 10 and 0 <= time.time() - second['epoch'] < 1800,
            'Two fresh capacity observations at least10seconds apart required')
    before = {str(p['parent']): p for p in first['parents']}
    after = {str(p['parent']): p for p in second['parents']}
    nodes, used_nodes = [], set()
    python, env = runtime()
    for shard in shards:
        parent = shard['parent']
        require(parent in before and parent in after, 'Selected parent missing capacity audit')
        a, b = before[parent], after[parent]
        require(not any(p.get('error') or p.get('excluded') for p in (a, b)), 'Selected parent probe excluded')
        fields = parent_fields(parent, limits['hard_step_limit_seconds'])
        require(fields['NodeList'] == a['node'] == b['node'] and b['node'] not in used_nodes,
                'Selected parents moved or share a physical node')
        used_nodes.add(b['node'])
        for sample in (a, b):
            require(sum(p['rss'] for p in sample['owned_processes']) < 16 * GIB and
                    sample['available_ram'] > 48 * GIB, 'Insufficient reserved RAM headroom for40GiB worker')
        previous = {g['uuid']: g for g in a['gpus']}
        idle = sorted((g for g in b['gpus'] if check_idle(g) and check_idle(previous.get(g['uuid'], {}))
                       and g['pci'] == previous[g['uuid']]['pci']), key=lambda g: (g['pci'], g['uuid']))
        require(idle, 'No twice-audited idle physical GPU')
        nodes.append({**shard, 'node': b['node'], 'gpu': idle[0], 'python': python, 'env': env,
                      'cpus': 4, 'mem_gib': 48, 'target_count': len(shard['dataset_indices'])})
    plan = {'created_epoch': time.time(), 'manifest': file_identity(manifest_path),
            'manifest_id': man['manifest_id'], 'campaign': 'mitra_class32_457_20260921_v1',
            'n_estimators': 32, 'actual_gpu_workers': len(nodes), 'nodes': nodes,
            'campaign_target_results': 457, 'target_results': len(ordered), 'dataset_indices': ordered,
            'scope': 'full457' if len(ordered) == 457 else 'smoke' if smoke else 'explicit_subset',
            'assignment': 'size-sorted round-robin, one worker per selected existing parent',
            'worker_sha256': {name: file_sha(REPO / name) for name in SCRIPTS},
            'stage_dependencies': man['stage_dependencies'],
            'model_weight_sha256': man['classification_weight']['sha256'],
            'capacity_observations': [file_identity(path) for path in capacity_paths],
            'gpu_lock_dir': str(FT / 'sidecar_gpu_locks'), 'job_name': JOB_NAME,
            'process_rss_limit_gib': 40, 'node_owned_rss_limit_gib': 56, 'node_available_min_gib': 8,
            **limits, 'continue_dataset_errors': True,
            'retry_policy': 'one attempt per dataset per named launch attempt; immutable errors and logs',
            'new_allocations': False, 'parent_allocations_unchanged': True}
    plan['plan_id'] = digest(plan)
    atomic(plan_path, plan, True)
    print(json.dumps(plan), flush=True)


def load_plan(path):
    plan = read(path)
    require(plan['plan_id'] == digest({k: v for k, v in plan.items() if k != 'plan_id'}), 'Runtime plan digest mismatch')
    check_identity(plan['manifest'])
    man = validate_manifest(read(plan['manifest']['path']))
    require(plan['manifest_id'] == man['manifest_id'] and plan['n_estimators'] == 32, 'Runtime/data manifest mismatch')
    for name, sha in plan['worker_sha256'].items():
        require(name in SCRIPTS and file_sha(REPO / name) == sha, 'Pinned campaign code changed: ' + name)
    require(set(plan['worker_sha256']) == set(SCRIPTS), 'Required script pins missing')
    for record in (man['weights_manifest'], man['class_benchmark_manifest'], *plan['stage_dependencies'].values()):
        check_identity(record)
    require(man['weights_manifest']['sha256'] == WEIGHTS_MANIFEST_SHA, 'Mitra weights manifest changed')
    require(man['class_benchmark_manifest']['sha256'] == BENCHMARK_MANIFEST_SHA and
            {name: record['sha256'] for name, record in plan['stage_dependencies'].items()} == STAGE_SHA,
            'Historical benchmark/helper pins changed')
    return plan, man


def validate_result(path, row, man, plan):
    value = read(path)
    require(value['complete'] is True and value['model_name'] == 'mitra' and
            value['dataset_index'] == row['dataset_index'] and value['dataset'] == row['dataset'] and
            value['suite'] == row['suite'], 'Result membership mismatch')
    require(value['task_kind'] == 'classification' and value['manifest_id'] == man['manifest_id'] and
            value['input_fingerprint'] == row['input_fingerprint'], 'Result input/protocol mismatch')
    require(value['worker_source_sha256'] == plan['worker_sha256']['mitra_class32_one.py'] and
            value['model_sha256'] == plan['model_weight_sha256'], 'Result worker/model identity mismatch')
    require(value['actual32_verified'] is True and value['n_estimators'] == value['actual_ensemble_count'] == 32,
            'Actual32 verification missing')
    require(value['node_count'] == len(value['ensemble_audits']) > 0, 'Hierarchy member audit missing')
    for audit in value['ensemble_audits']:
        require(audit['actual32_verified'] is True and audit['actual_ensemble_count'] == 32 and
                len(audit['member_audits']) == 32 and audit['all_test_rows_covered'] is True and
                audit['minimum_contributions_per_test_row'] == audit['maximum_contributions_per_test_row'] == 32,
                'A hierarchy node did not contribute exactly32 members to every routed test row')
    require(math.isfinite(value['accuracy']) and 0 <= value['accuracy'] <= 1, 'Invalid accuracy')
    require(value['test_rows'] == row['test_rows'] and value['full_test_split'] is True,
            'Test coverage changed')
    return value


def check_start_ram(psutil):
    if owned_rss(psutil) >= 16 * GIB or psutil.virtual_memory().available <= 48 * GIB:
        raise SafetyError('Fresh node RAM no longer reserves40GiB worker plus8GiB safety margin')


def node_run(parent, attempt, plan_path):
    import psutil
    plan, man = load_plan(plan_path)
    node = next(n for n in plan['nodes'] if n['parent'] == parent)
    require(os.environ['SLURM_JOB_ID'] == parent and socket.gethostname() == node['node'], 'Wrong parent/node binding')
    require(parent_fields(parent)['NodeList'] == node['node'], 'Parent moved before launch')
    check_start_ram(psutil)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    lock_name = node['node'] + '.' + node['gpu']['uuid'] + '.lock'
    require(re.fullmatch(r'[a-zA-Z0-9_.-]+', lock_name), 'Unsafe physical lock filename')
    lock_root = Path(plan['gpu_lock_dir']); lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / lock_name).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_lane(plan, man, node, attempt, stop, psutil)


def run_lane(plan, man, node, attempt, stop, psutil):
    physical = node['gpu']
    fresh = gpu(physical['uuid'])
    require(fresh['pci'] == physical['pci'] and check_idle(fresh), 'Assigned physical GPU no longer idle')
    identity = node['parent'] + '.' + os.environ['SLURM_STEP_ID'] + '.' + node['lane_id']
    status_path = ROOT / 'workers' / (identity + '.json')
    status = {'worker': identity, 'parent': node['parent'], 'node': node['node'], 'gpu': physical,
              'plan_id': plan['plan_id'], 'manifest_id': man['manifest_id'], 'attempt': attempt,
              'started_epoch': time.time(), 'state': 'starting', 'completed': 0, 'failed': 0,
              'target': node['target_count']}
    atomic(status_path, status)
    env = os.environ.copy()
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
        env.pop(key, None)
    env.update(node['env'])
    env.update(ROCR_VISIBLE_DEVICES='GPU-' + physical['uuid'], EXPECTED_GPU_UUID=physical['uuid'],
               EXPECTED_GPU_PCI_BUS_ID=physical['pci'])
    cpus = sorted(os.sched_getaffinity(0))[:4]
    require(len(cpus) == 4, 'CPU affinity too small')
    deadline = time.monotonic() + plan['soft_step_limit_seconds']
    try:
        for index in node['dataset_indices']:
            if stop.is_set() or time.monotonic() >= deadline:
                raise SafetyError('Child stopped or bounded step deadline reached; remaining rows preserved for resume')
            row = man['rows'][index]
            output = ROOT / 'results/mitra' / f'row-{index:03d}.json'
            if output.exists():
                validate_result(output, row, man, plan); status['completed'] += 1
                continue
            error_path = ROOT / 'errors' / attempt / f'row-{index:03d}.json'
            if error_path.exists():
                status['failed'] += 1
                continue
            claim = ROOT / 'claims' / f'row-{index:03d}.json'
            token = uuid.uuid4().hex
            owner = {'worker': identity, 'token': token, 'plan_id': plan['plan_id'], 'manifest_id': man['manifest_id'],
                     'dataset_index': index, 'dataset': row['dataset'], 'started_epoch': time.time()}
            atomic(claim, owner, True)
            proc = None; reason = None; peak = 0; started = time.monotonic()
            log = ROOT / 'logs' / attempt / f'row-{index:03d}.log'
            try:
                if output.exists():
                    validate_result(output, row, man, plan); status['completed'] += 1
                    continue
                check_start_ram(psutil)
                check_identity(row['cache'], hash_content=False)
                fresh = gpu(physical['uuid'])
                for _ in range(30):
                    if fresh['vram'] < IDLE_MAX or stop.is_set():
                        break
                    stop.wait(0.5); fresh = gpu(physical['uuid'])
                if fresh['pci'] != physical['pci'] or fresh['vram'] >= IDLE_MAX or stop.is_set():
                    raise SafetyError('GPU acquired persistent memory before next evaluation')
                command = ['taskset', '-c', ','.join(map(str, cpus)), node['python'],
                           str(REPO / 'mitra_class32_one.py'), '--plan', plan['manifest']['path'],
                           '--dataset-index', str(index), '--output', str(output), '--threads', '4']
                status.update(state='running', dataset_index=index, dataset=row['dataset'], heartbeat_epoch=time.time())
                atomic(status_path, status)
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('x') as handle:
                    proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=handle,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    while proc.poll() is None:
                        try:
                            main = psutil.Process(proc.pid)
                            rss = sum(p.memory_info().rss for p in [main] + main.children(recursive=True) if p.is_running())
                        except psutil.NoSuchProcess:
                            rss = 0
                        except psutil.AccessDenied as exc:
                            raise SafetyError('Cannot inspect own evaluator RSS') from exc
                        peak = max(peak, rss)
                        if stop.is_set():
                            raise SafetyError('Dispatcher stop requested')
                        if rss > plan['process_rss_limit_gib'] * GIB:
                            raise SafetyError('Own evaluator exceeded40GiB RSS budget')
                        if owned_rss(psutil) > plan['node_owned_rss_limit_gib'] * GIB or \
                                psutil.virtual_memory().available < plan['node_available_min_gib'] * GIB:
                            raise SafetyError('Node56GiB owned RSS /8GiB available guard')
                        if time.monotonic() >= deadline:
                            raise SafetyError('Bounded child step deadline reached')
                        if time.monotonic() - started > plan['single_dataset_limit_seconds']:
                            reason = f"{plan['single_dataset_limit_seconds']}second single-dataset bound"
                            stop_own(proc); break
                        try:
                            proc.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                require(not reason and proc.returncode == 0, reason or f'Evaluator exit {proc.returncode}')
                try:
                    result = validate_result(output, row, man, plan)
                except Exception as exc:
                    raise SafetyError('Completed result failed input/model/actual32 validation: ' + str(exc)) from exc
                status['completed'] += 1
                status.update(last_accuracy=result['accuracy'], last_result=str(output), last_peak_rss_gib=peak/GIB)
                print(json.dumps({'dataset_index': index, 'complete': status['completed'],
                                  'target': node['target_count'], 'elapsed_s': time.monotonic()-started}), flush=True)
            except Exception as exc:
                if proc is not None:
                    stop_own(proc)
                error = {**owner, 'complete': False, 'reason': str(exc), 'exception': type(exc).__name__,
                         'safety_stop': isinstance(exc, SafetyError), 'log': str(log), 'peak_rss_gib': peak/GIB,
                         'returncode': proc.returncode if proc else None, 'ended_epoch': time.time()}
                atomic(error_path, error, True)
                status['failed'] += 1
                status.update(last_error=error)
                if isinstance(exc, SafetyError) or proc is None:
                    raise
            finally:
                if proc is not None:
                    stop_own(proc)
                if claim.exists() and read(claim).get('token') == token:
                    claim.unlink()
                status.update(heartbeat_epoch=time.time())
                atomic(status_path, status)
        require(not status['failed'], f"{status['failed']} bounded dataset failures; immutable evidence retained")
        status.update(state='complete', complete=True, ended_epoch=time.time())
        atomic(status_path, status)
    except Exception as exc:
        status.update(state='failed', complete=False, error=str(exc), ended_epoch=time.time())
        atomic(status_path, status)
        raise


def launch(plan_path, attempt, only_parent=None):
    plan, _ = load_plan(plan_path)
    require(0 <= time.time() - plan['created_epoch'] < 1800, 'Runtime plan expired; fresh resource audit required')
    require(only_parent is None or any(n['parent'] == only_parent for n in plan['nodes']), 'Parent absent from plan')
    with (ROOT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for node in plan['nodes']:
            parent = node['parent']
            if only_parent and parent != only_parent:
                continue
            require(parent_fields(parent, plan['hard_step_limit_seconds'])['NodeList'] == node['node'],
                    'Parent moved after resource audit')
            active = subprocess.check_output(['squeue', '--steps', '-j', parent, '-h', '-o', '%i|%j'], text=True)
            require(JOB_NAME not in active, 'Existing classification32 evaluator on selected parent')
            receipt_path = ROOT / 'launches' / (attempt + '.' + parent + '.json')
            if receipt_path.exists():
                old = read(receipt_path)
                require(old['plan_id'] == plan['plan_id'], 'Named launch attempt already belongs to another plan')
                continue
            command = ['srun', '--jobid=' + parent, '--overlap', '--exact', '-N1', '-n1', '-c4', '--mem=48G',
                       '--gpus=8', '--gpu-bind=none', '--time=' + plan['hard_step_limit'],
                       '--unbuffered', '--job-name=' + JOB_NAME,
                       node['python'], str(REPO / 'mitra_class32_dispatch.py'), 'node', '--parent', parent,
                       '--attempt', attempt, '--runtime-plan', str(plan_path)]
            log = receipt_path.with_suffix('.log'); log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('x') as handle:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            receipt = {'epoch': time.time(), 'parent': parent, 'node': node['node'], 'plan_id': plan['plan_id'],
                       'launcher_pid': process.pid, 'gpu': node['gpu'], 'physical_gpu_count': 1,
                       'command': command, 'log': str(log), 'target_results': node['target_count']}
            atomic(receipt_path, receipt, True)
            print(json.dumps(receipt), flush=True)


def self_test():
    rows = [{'dataset_index': i, 'cache': {'size_bytes': (i * 137) % 509},
             'classes': [2, 5, 12][i % 3]} for i in range(457)]
    for count in range(1, 13):
        parents = sorted(PARENTS)[:count]
        order, shards = shard_rows(rows, parents)
        flat = [i for shard in shards for i in shard['dataset_indices']]
        require(len(flat) == len(set(flat)) == 457 and set(flat) == set(range(457)), 'Self-test coverage failed')
        lengths = [len(s['dataset_indices']) for s in shards]
        require(max(lengths) - min(lengths) <= 1, 'Self-test balance failed')
        require(shard_rows(list(reversed(rows)), parents) == (order, shards), 'Self-test determinism failed')
    picks = smoke_indices(rows)
    require(len(picks) == 3 and [rows[i]['classes'] for i in picks] == [2, 5, 12], 'Smoke categories incorrect')
    order, shards = shard_rows(rows, sorted(PARENTS)[:2], picks)
    require(set(order) == set(picks), 'Smoke shard coverage incorrect')
    for indices in ([0, 0], [457], []):
        try:
            shard_rows(rows, sorted(PARENTS)[:1], indices)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Invalid subset accepted')
    require(digest({'b': 2, 'a': 1}) == digest({'a': 1, 'b': 2}), 'Digest ordering changed')
    print(json.dumps({'self_test': 'pass', 'membership_count': 457, 'layouts_tested': 12,
                      'coverage': True, 'no_overlap': True, 'deterministic': True, 'smoke_categories': 3}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'plan', 'launch', 'node', 'self-test'))
    parser.add_argument('--manifest', type=Path, default=ROOT / 'manifest.json')
    parser.add_argument('--runtime-plan', type=Path, default=ROOT / 'plan.json')
    parser.add_argument('--parents', help='Explicit comma-separated audited existing parent IDs; one worker each')
    parser.add_argument('--parent')
    parser.add_argument('--indices', help='Optional explicit smoke/recovery dataset indices; never changes manifest')
    parser.add_argument('--smoke', action='store_true', help='Three smallest binary/native-multiclass/hierarchical rows')
    parser.add_argument('--attempt', default='v1')
    parser.add_argument('--hours', type=int, default=2, help='Bounded child-step hours:2 for smoke,12 for full run')
    args = parser.parse_args()
    if args.mode == 'self-test':
        self_test(); return
    require(re.fullmatch(r'[A-Za-z0-9_-]+', args.attempt), 'Unsafe attempt name')
    require(args.parent is None or args.parent in PARENTS, 'Invalid selected parent')
    for path in (args.manifest, args.runtime_plan):
        require(path.resolve().parent == ROOT.resolve(), 'New campaign files must stay directly inside newroot')
    if args.mode == 'prepare':
        prepare(args.manifest)
    elif args.mode == 'plan':
        require(args.parents, 'plan requires explicit --parents')
        parents = [p.strip() for p in args.parents.split(',') if p.strip()]
        indices = [int(i.strip()) for i in args.indices.split(',')] if args.indices else None
        build_plan(args.manifest, args.runtime_plan, parents, indices, args.smoke, args.hours)
    elif args.mode == 'node':
        require(args.parent, 'node requires --parent')
        node_run(args.parent, args.attempt, args.runtime_plan)
    else:
        launch(args.runtime_plan, args.attempt, args.parent)


if __name__ == '__main__':
    main()
