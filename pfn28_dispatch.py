"""Bounded, resumable PFN28 foundation evaluation in audited idle allocations.

Only own child processes may be stopped. No training, downloads, parent changes,
source-checkpoint changes, old-result overwrites, or new Slurm allocations.
"""
import argparse
import concurrent.futures
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
ROOT = FT / 'pfn28_foundations_20260921_v1'
REPO = FT / 'repo'
MANIFEST = FT / 'eval224/manifest.json'
MODEL_MANIFEST = BASE / 'stage/regression_six_models_six_suites_bg7_20260822_v3/model_manifest.json'
MODEL_SHA = '9b7f86c1488dd0382e05d9ed1d98a0c044a7c544dde6b3c411360f1bb1fbcc07'
MITRA = BASE / 'stage/mitra_all_classification_regression_20260823_v3'
MITRA_SHA = '228008da42a9bef329872735200175f2b513f945e425abdb7074a94556a8330f'
PYBASE = Path('/vast/users/guangyi.chen/causal_group/zijian.li/tabicl_causal')
COMMON_PY = PYBASE / 'tabicl-main-paper2602-dataset/.conda_env/bin/python3'
TABICL_PY = PYBASE / 'new_tab/nothing3_clean_sp_lr01_1753412/.conda_env/bin/python'
ASSIGNMENTS = {'196093': ['tabiclv2', 'tabpfn2'], '200798': ['tabpfn25', 'tabpfn3'],
               '204828': ['limix2m', 'limix16m'], '194259': ['mitra']}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_fields(parent):
    assert parent in ASSIGNMENTS
    raw = subprocess.check_output(['scontrol', 'show', 'job', parent, '-o'], text=True)
    f = dict(re.findall(r'(\w+)=(\S+)', raw))
    assert f['JobState'] == 'RUNNING' and f['NumNodes'] == '1'
    assert f['UserId'].startswith('guangyi.chen(')
    assert re.search(r'(?:^|,)gres/gpu=8(?:,|$)', f['AllocTRES'])
    assert re.search(r'(?:^|,)mem=64G(?:,|$)', f['AllocTRES'])
    return f


def runtime(model):
    env = dict(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
        PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if model == 'mitra':
        py = MITRA / 'venv/bin/python'
        paths = [str(MITRA)]
        env['HF_HOME'] = str(MITRA / 'hf_cache')
    elif model == 'tabiclv2':
        py = TABICL_PY
        paths = [str(BASE / 'stage/tabicl_regression_supportonly_20260820_v5/source/src')]
    elif model.startswith('tabpfn'):
        py = COMMON_PY
        stage = BASE / 'stage/tabpfn_v2_v25_v3_exact178_20260820_v1'
        paths = [str(stage / 'deps'), str(stage / 'source/src')]
    else:
        py = COMMON_PY
        if model == 'limix2m':
            stage = BASE / 'stage/limix2m_adapted_exact178_20260810_v7'
            paths = [str(stage / 'python_deps'), str(stage / 'source')]
        else:
            stage = BASE / 'stage/five_method_exact178_rw4096_step500_20260818_v1'
            paths = [str(stage / 'limix16m_python_deps'), str(stage / 'limix16m_source')]
        env.update(RANK='0', WORLD_SIZE='1', LOCAL_RANK='0', MASTER_ADDR='127.0.0.1',
                   MASTER_PORT='33704' if model == 'limix2m' else '33705')
    assert py.is_file(), py
    env['PYTHONPATH'] = ':'.join(paths + [str(REPO)])
    return str(py), env


def build_plan():
    assert digest(MODEL_MANIFEST) == MODEL_SHA
    assert digest(MITRA / 'weights_manifest.json') == MITRA_SHA
    man = read(MANIFEST)
    assert man['source_step'] == 22175 and man['membership_count'] == 224
    rows = [r for r in man['rows'] if r['suite'] == 'PFN']
    assert len(rows) == 28 and {r['dataset_index'] for r in rows} == set(range(196, 224))
    first, second = read(ROOT / 'capacity_a.json'), read(ROOT / 'capacity_b.json')
    assert second['epoch'] - first['epoch'] >= 10 and 0 <= time.time() - second['epoch'] < 1800
    before = {p['parent']: p for p in first['parents']}
    after = {p['parent']: p for p in second['parents']}
    nodes = []
    for parent, models in ASSIGNMENTS.items():
        a, b = before[parent], after[parent]
        assert not any(x.get('error') or x.get('excluded') for x in (a, b))
        f = checked_fields(parent)
        assert f['NodeList'] == a['node'] == b['node']
        assert sum(p['rss'] for p in b['owned_processes']) < 8 * GIB
        assert b['available_ram'] > 56 * GIB
        old = {g['uuid']: g for g in a['gpus']}
        idle = [g for g in b['gpus'] if check_idle(g) and check_idle(old.get(g['uuid'], {}))
                and g['pci'] == old[g['uuid']]['pci']]
        assert len(idle) >= len(models)
        lanes = []
        for model, g in zip(models, idle):
            py, env = runtime(model)
            lanes.append({'model': model, 'gpu': g, 'python': py, 'env': env})
        nodes.append({'parent': parent, 'node': b['node'], 'lanes': lanes,
                      'cpus': 4 * len(lanes), 'mem_gib': 24 * len(lanes)})
    scripts = ['pfn28_dispatch.py', 'pfn_foundation_one.py', 'pfn_mitra_one.py', 'eval_one.py', 'eval_dispatch.py']
    plan = {'created_epoch': time.time(), 'manifest': str(MANIFEST), 'manifest_id': man['manifest_id'],
            'manifest_sha256': digest(MANIFEST), 'model_manifest_sha256': MODEL_SHA,
            'mitra_manifest_sha256': MITRA_SHA, 'target_results': 196, 'models': sum(ASSIGNMENTS.values(), []),
            'source_checkpoint_step': 22175, 'dataset_indices': [r['dataset_index'] for r in sorted(rows,
                key=lambda r: (sum(f['size_bytes'] for f in r['input_files']), r['dataset_index']))],
            'nodes': nodes, 'actual_gpu_workers': 7, 'row_rss_limit_gib': 20,
            'hard_step_limit': '02:00:00', 'parent_allocations_unchanged': True,
            'new_allocations': False, 'worker_sha256': {p: digest(REPO / p) for p in scripts}}
    plan['model_weight_sha256'] = {m['name']: m['sha256'] for m in read(MODEL_MANIFEST)['models']}
    plan['model_weight_sha256']['mitra'] = 'd8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642'
    plan['plan_id'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    atomic(ROOT / 'plan.json', plan, True)
    print(json.dumps(plan), flush=True)


def load_plan():
    p = read(ROOT / 'plan.json')
    assert p['plan_id'] == hashlib.sha256(json.dumps({k:v for k,v in p.items() if k != 'plan_id'}, sort_keys=True).encode()).hexdigest()
    assert p['manifest_sha256'] == digest(MANIFEST)
    assert digest(MODEL_MANIFEST) == MODEL_SHA and digest(MITRA / 'weights_manifest.json') == MITRA_SHA
    for name, h in p['worker_sha256'].items():
        assert digest(REPO / name) == h, 'Worker changed after campaign freeze: ' + name
    return p


def validate_result(path, model, index, man, plan):
    r = read(path)
    assert r['complete'] is True and r['model_name'] == model and r['dataset_index'] == index
    assert r['manifest_id'] == man['manifest_id'] and r['input_fingerprint'] == man['rows'][index]['input_fingerprint']
    assert r['dataset'] == man['rows'][index]['dataset'] and r['row_id'] == man['rows'][index]['row_id']
    script = 'pfn_mitra_one.py' if model == 'mitra' else 'pfn_foundation_one.py'
    assert r['worker_source_sha256'] == plan['worker_sha256'][script]
    assert r['model_sha256'] == plan['model_weight_sha256'][model]
    if model != 'mitra': assert r['model_manifest_sha256'] == plan['model_manifest_sha256']
    assert r['source_result_audit_match'] is True
    assert all(math.isfinite(r['metrics'][k]) for k in ('rmse', 'r2', 'mae'))
    return r


def lane_run(plan, node, lane, ordinal, attempt, stop):
    import psutil
    man = read(MANIFEST)
    model = lane['model']; g = lane['gpu']
    identity = node['parent'] + '.' + os.environ['SLURM_STEP_ID'] + '.' + model
    status_path = ROOT / 'workers' / (identity + '.json')
    status = {'worker': identity, 'model': model, 'node': node['node'], 'parent': node['parent'],
              'step': os.environ['SLURM_STEP_ID'], 'gpu': g, 'plan_id': plan['plan_id'],
              'started_epoch': time.time(), 'state': 'starting', 'completed': 0}
    atomic(status_path, status)
    fresh = gpu(g['uuid'])
    assert fresh['pci'] == g['pci'] and check_idle(fresh), 'Assigned GPU no longer idle'
    env = os.environ.copy()
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
        env.pop(key, None)
    env.update(lane['env'])
    env.update(ROCR_VISIBLE_DEVICES='GPU-' + g['uuid'], EXPECTED_GPU_UUID=g['uuid'], EXPECTED_GPU_PCI_BUS_ID=g['pci'])
    cpus = sorted(os.sched_getaffinity(0))[ordinal * 4: (ordinal + 1) * 4]
    assert len(cpus) == 4
    try:
        for index in plan['dataset_indices']:
            if stop.is_set():
                raise RuntimeError('Node dispatcher stop requested')
            output = ROOT / 'results' / model / f'row-{index:03d}.json'
            if output.exists():
                validate_result(output, model, index, man, plan); status['completed'] += 1; continue
            claim = ROOT / 'claims' / model / f'row-{index:03d}.json'
            claim.parent.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid4().hex
            record = {'worker': identity, 'token': token, 'started_epoch': time.time(),
                      'dataset_index': index, 'model': model, 'plan_id': plan['plan_id']}
            with claim.open('x') as h:
                json.dump(record, h); h.flush(); os.fsync(h.fileno())
            proc = None; reason = None; peak = 0; started = time.monotonic()
            log = ROOT / 'logs' / attempt / model / f'row-{index:03d}.log'
            try:
                if output.exists():
                    validate_result(output, model, index, man, plan); status['completed'] += 1; continue
                source = FT / 'eval224/results/step-22175' / f'row-{index:03d}.json'
                assert source.is_file()
                script = 'pfn_mitra_one.py' if model == 'mitra' else 'pfn_foundation_one.py'
                cmd = ['taskset', '-c', ','.join(map(str, cpus)), lane['python'], str(REPO / script),
                       '--manifest', str(MANIFEST), '--dataset-index', str(index), '--source-result', str(source),
                       '--output', str(output), '--threads', '4']
                if model == 'mitra':
                    cmd += ['--mitra-stage', str(MITRA), '--weights-manifest', str(MITRA / 'weights_manifest.json')]
                else:
                    cmd += ['--model-manifest', str(MODEL_MANIFEST), '--model-manifest-sha256', MODEL_SHA,
                            '--model-name', model, '--test-chunk', '4096']
                status.update(state='running', dataset_index=index, heartbeat_epoch=time.time())
                atomic(status_path, status)
                # Prior process has exited; allow bounded ROCm cleanup, but do
                # not take a GPU acquired by any unrelated parent workload.
                fresh = gpu(g['uuid'])
                for _ in range(30):
                    if fresh['vram'] < IDLE_MAX or stop.is_set(): break
                    stop.wait(0.5)
                    fresh = gpu(g['uuid'])
                assert fresh['pci'] == g['pci'] and fresh['vram'] < IDLE_MAX and not stop.is_set(), 'GPU acquired memory between evaluations'
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('x') as h:
                    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=h,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    while proc.poll() is None:
                        try:
                            main = psutil.Process(proc.pid)
                            rss = sum(x.memory_info().rss for x in [main] + main.children(recursive=True) if x.is_running())
                        except psutil.NoSuchProcess:
                            rss = 0
                        peak = max(peak, rss)
                        if stop.is_set(): reason = 'dispatcher stop request'
                        elif rss > 20 * GIB: reason = '20GiB evaluator RAM guard'
                        elif owned_rss(psutil) > 56 * GIB or psutil.virtual_memory().available < 8 * GIB:
                            reason = 'node RAM guard'; stop.set()
                        elif time.monotonic() - started > 1800: reason = '30 minute single dataset bound'
                        if reason:
                            stop_own(proc); break
                        try: proc.wait(timeout=1)
                        except subprocess.TimeoutExpired: pass
                assert not reason and proc.returncode == 0, reason or f'evaluator exit {proc.returncode}'
                result = validate_result(output, model, index, man, plan)
                status['completed'] += 1
                status.update(heartbeat_epoch=time.time(), last_result=str(output), last_peak_rss_gib=peak/GIB,
                              smoke_passed=True, last_metrics=result['metrics'])
                atomic(status_path, status)
                print(json.dumps({'model': model, 'complete': status['completed'], 'target': 28,
                                  'dataset_index': index, 'elapsed_s': time.monotonic()-started}), flush=True)
            except Exception as exc:
                error = {**record, 'complete': False, 'reason': str(exc), 'log': str(log),
                         'peak_rss_gib': peak/GIB, 'epoch': time.time(), 'returncode': proc.returncode if proc else None}
                atomic(ROOT / 'errors' / attempt / model / f'row-{index:03d}.json', error, True)
                raise
            finally:
                if proc is not None: stop_own(proc)
                if claim.exists() and read(claim).get('token') == token: claim.unlink()
        status.update(state='complete', complete=True, ended_epoch=time.time())
        atomic(status_path, status)
    except Exception as exc:
        status.update(state='failed', error=str(exc), ended_epoch=time.time())
        atomic(status_path, status)
        print(json.dumps(status), flush=True)
        raise


def node_run(parent, attempt):
    import psutil
    p = load_plan(); node = next(n for n in p['nodes'] if n['parent'] == parent)
    assert os.environ['SLURM_JOB_ID'] == parent and socket.gethostname() == node['node']
    assert checked_fields(parent)['NodeList'] == node['node']
    assert owned_rss(psutil) < 8*GIB and psutil.virtual_memory().available > 56*GIB
    stop = threading.Event()
    for s in (signal.SIGTERM, signal.SIGINT): signal.signal(s, lambda *_: stop.set())
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(node['lanes'])) as pool:
        futures = [pool.submit(lane_run, p, node, lane, i, attempt, stop) for i,lane in enumerate(node['lanes'])]
        errors = []
        for future in futures:
            try: future.result()
            except Exception as exc: errors.append(str(exc))
    if errors: raise RuntimeError(errors)


def launch(attempt, only_parent=None):
    p = load_plan()
    assert time.time() - p['created_epoch'] < 1800, 'Plan too old; resource audit required'
    with (ROOT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for node in p['nodes']:
            parent = node['parent']
            if only_parent and parent != only_parent: continue
            assert checked_fields(parent)['NodeList'] == node['node']
            active = subprocess.check_output(['squeue', '--steps', '-j', parent, '-h', '-o', '%i|%j'], text=True)
            assert 'pfn28base' not in active and 'regft50eval' not in active, 'Existing evaluator on selected parent'
            path = ROOT / 'launches' / (attempt + '.' + parent + '.json')
            if path.exists(): continue
            cmd = ['srun', '--jobid='+parent, '--overlap', '--exact', '-N1', '-n1', '-c'+str(node['cpus']),
                   '--mem='+str(node['mem_gib'])+'G', '--gpus=8', '--gpu-bind=none', '--time=02:00:00',
                   '--unbuffered', '--job-name=pfn28base', sys.executable, str(REPO / 'pfn28_dispatch.py'),
                   'node', '--parent', parent, '--attempt', attempt]
            # The pre-existing parent owns all8 GPUs. Expose them to this child,
            # but each evaluator is isolated to one audited physical UUID.
            log = path.with_suffix('.log'); log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('x') as h:
                process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=h, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            receipt = {'epoch':time.time(), 'parent':parent, 'node':node['node'], 'plan_id':p['plan_id'],
                       'models':[x['model'] for x in node['lanes']], 'physical_gpu_count':len(node['lanes']),
                       'launcher_pid':process.pid, 'command':cmd, 'log':str(log)}
            atomic(path, receipt, True); print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['plan', 'launch', 'node'])
    parser.add_argument('--parent'); parser.add_argument('--attempt', default='v1')
    args = parser.parse_args(); assert re.fullmatch(r'[a-zA-Z0-9_-]+', args.attempt)
    if args.mode == 'plan': build_plan()
    elif args.mode == 'node': node_run(args.parent, args.attempt)
    else: launch(args.attempt, args.parent)
