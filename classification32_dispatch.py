"""Bounded GPU workers for the isolated actual32 campaign. Never touch old jobs/results."""
from __future__ import annotations
import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

import classification32_campaign as campaign

ROOT = campaign.ROOT
GROUPS = (('tabpfn2', 'tabpfn25', 'tabpfn3'), ('limix2m', 'limix16m'),
          ('tabiclv1', 'tabiclv2', 'taffy_loop3', 'taffy_loop4'), ('mitra2',))
CHILD = None


def normalize_visibility(env):
    env = dict(env)
    campaign.require(env.get('SLURM_NTASKS') == '4'
        and env.get('SLURM_PROCID') in {'0', '1', '2', '3'}
        and env.get('SLURM_PROCID') == env.get('SLURM_LOCALID'), 'Expected four-rank single-node step')
    mask = env.get('ROCR_VISIBLE_DEVICES', '')
    campaign.require(re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+)', mask),
                     'Exactly one scheduler-assigned physical ROCr GPU mask required')
    names = ('ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')
    env['CLASS32_ORIGINAL_VISIBILITY'] = json.dumps({key: env.get(key) for key in names})
    for key in names[1:]:
        env.pop(key, None)
    return env


def stop(_sig, _frame):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
    raise SystemExit(143)


def gpu_record():
    import torch
    campaign.require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.version.hip,
                     'Expected one actually available AMD GPU; environment strings alone are insufficient')
    from pfn_mitra_one import select_loaded_hip_library
    torch.cuda.init()
    torch.cuda.set_device(0)
    library = select_loaded_hip_library(Path('/proc/self/maps').read_text())
    campaign.require(hasattr(os, 'RTLD_NOLOAD'), 'Loaded-runtime-only HIP inspection required')
    lib = ctypes.CDLL(str(library), mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
    fn = lib.hipDeviceGetPCIBusId
    fn.argtypes, fn.restype = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int], ctypes.c_int
    buf = ctypes.create_string_buffer(64)
    campaign.require(fn(buf, len(buf), 0) == 0, 'HIP PCI lookup failed')
    domain, bus, slot = buf.value.decode().lower().split(':')
    pci = f'{int(domain, 16):04x}:{bus}:{slot}'
    physical_uuid = (Path('/sys/bus/pci/devices') / pci / 'unique_id').read_text().strip().lower()
    campaign.require(physical_uuid and int(physical_uuid, 16) != 0, 'Missing real physical GPU identity')
    v = torch.ones((4, 4), device='cuda')
    campaign.require(float(v.sum().cpu()) == 16, 'GPU forward probe failed')
    return {'node': socket.gethostname(), 'uuid': physical_uuid, 'pci': pci,
            'gpu_name': torch.cuda.get_device_name(0), 'hip': torch.version.hip,
            'hip_runtime_library': str(library),
            'rank': int(os.environ['SLURM_PROCID']), 'job': os.environ['SLURM_JOB_ID'],
            'step': os.environ['SLURM_STEP_ID'], 'epoch': time.time()}


def preflight():
    man = campaign.manifest_load()
    rec = gpu_record()
    rec['manifest_id'] = man['manifest_id']
    campaign.atomic(ROOT / 'preflight' / rec['job'] / f"rank-{rec['rank']}.json", rec)
    print(json.dumps(rec), flush=True)


def check_preflight(man):
    job = os.environ['SLURM_JOB_ID']
    records = [campaign.read(ROOT / 'preflight' / job / f'rank-{i}.json') for i in range(4)]
    campaign.require({r['rank'] for r in records} == set(range(4))
        and len({r['node'] for r in records}) == 1
        and len({r['uuid'] for r in records}) == 4
        and len({r['pci'] for r in records}) == 4
        and all(r['manifest_id'] == man['manifest_id'] and r['job'] == job for r in records),
        'Four distinct physical GPUs required')
    return records


def binding(man):
    records = check_preflight(man)
    rec = gpu_record()
    expected = records[rec['rank']]
    campaign.require((rec['node'], rec['uuid'], rec['pci']) ==
        (expected['node'], expected['uuid'], expected['pci']), 'Rank binding differs from verified allocation')
    campaign.atomic(ROOT / 'bindings' / rec['job'] / rec['step'] / f"rank-{rec['rank']}.json", rec)
    return rec


def launch(man, model, row, smoke=False):
    global CHILD
    import psutil
    cfg = man['models'][model]
    env = dict(os.environ, **cfg['env'])
    env.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', TABPFN_DISABLE_TELEMETRY='1', PYTHONNOUSERSITE='1',
        PYTORCH_HIP_ALLOC_CONF='expandable_segments:True', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    # All temporary inference artifacts stay in this campaign, never old result roots.
    temp_root = ROOT / 'tmp' / os.environ['SLURM_JOB_ID'] / os.environ['SLURM_PROCID']
    temp_root.mkdir(parents=True, exist_ok=True)
    env['TMPDIR'] = str(temp_root)
    index = row['dataset_index']
    phase = 'smoke' if smoke else 'formal'
    ident = f'{model}-{index:03d}-{uuid.uuid4().hex}'
    log = ROOT / 'logs' / (ident + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [cfg['python'], str(Path(campaign.__file__).resolve()), 'one', '--model', model, '--index', str(index)]
    if smoke:
        cmd.append('--smoke')
    started = time.time()
    reason = None
    peak = 0
    heartbeat = ROOT / 'workers' / os.environ['SLURM_JOB_ID'] / f"rank-{os.environ['SLURM_PROCID']}.json"
    with log.open('x') as handle:
        CHILD = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        while CHILD.poll() is None:
            try:
                process = psutil.Process(CHILD.pid)
                rss = sum(p.memory_info().rss for p in [process] + process.children(recursive=True) if p.is_running())
                peak = max(peak, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss = 0
            elapsed = time.time() - started
            if rss > man['per_task_rss_limit_bytes'] or elapsed > man['per_task_timeout_seconds']:
                reason = 'rss_budget_exceeded' if rss > man['per_task_rss_limit_bytes'] else 'task_timeout'
                os.killpg(CHILD.pid, signal.SIGTERM)
                try:
                    CHILD.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(CHILD.pid, signal.SIGKILL)
                    CHILD.wait()
                break
            campaign.atomic(heartbeat, {'manifest_id': man['manifest_id'], 'model': model,
                'dataset_index': index, 'phase': phase, 'pid': CHILD.pid, 'rss': rss,
                'peak_rss': peak, 'started_epoch': started, 'heartbeat_epoch': time.time()}, immutable=False)
            try:
                CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        rc = CHILD.returncode
    output = ROOT / ('smoke' if smoke else 'results') / model / f'row-{index:03d}.json'
    receipt = {'phase': phase, 'model': model, 'dataset_index': index, 'dataset': row['dataset'],
        'manifest_id': man['manifest_id'], 'job': os.environ['SLURM_JOB_ID'],
        'rank': os.environ['SLURM_PROCID'], 'started_epoch': started, 'finished_epoch': time.time(),
        'exit_code': rc, 'reason': reason, 'peak_rss': peak, 'log': str(log), 'output': str(output)}
    CHILD = None
    if rc == 0 and not reason and output.exists():
        try:
            campaign.valid_result(output, man, model, row)
        except Exception as exc:
            receipt['reason'] = 'output_validation_failed: ' + repr(exc)
            reason = receipt['reason']
    if rc != 0 or reason or not output.exists():
        receipt['success'] = False
        campaign.atomic(ROOT / 'errors' / (ident + '.json'), receipt)
        print(json.dumps(receipt), flush=True)
        return False
    receipt['success'] = True
    campaign.atomic(ROOT / 'attempts' / (ident + '.json'), receipt)
    print(json.dumps(receipt), flush=True)
    return True


def smoke(shard):
    man = campaign.manifest_load()
    rec = binding(man)
    models = GROUPS[shard]
    if rec['rank'] >= len(models):
        return
    model = models[rec['rank']]
    for idx in man['smoke_indices']:
        if not launch(man, model, man['rows'][idx], smoke=True):
            campaign.atomic(ROOT / 'gates' / f'{model}.failed.json',
                {'model': model, 'manifest_id': man['manifest_id'], 'failed_index': idx, 'epoch': time.time()})
            # One adapter failure does not kill independent smoke tests/methods.
            return
    campaign.atomic(ROOT / 'gates' / f'{model}.passed.json',
        {'model': model, 'manifest_id': man['manifest_id'], 'indices': man['smoke_indices'], 'epoch': time.time()})


def claim(man, model, row, owner):
    index = row['dataset_index']
    path = ROOT / 'claims' / model / f'row-{index:03d}.json'
    try:
        campaign.atomic(path, dict(owner, model=model, dataset_index=index,
            manifest_id=man['manifest_id'], claimed_epoch=time.time()))
    except FileExistsError:
        return False
    return True


def dispatch():
    man = campaign.manifest_load()
    owner = binding(man)
    rank = owner['rank']
    # Short examples first; no omission, change of split, or old-result reuse.
    rows = sorted(man['rows'], key=lambda r: (r['train_rows'] + r['test_rows']) * r['features'])
    models = list(campaign.MODELS)
    offset = (int(os.environ['CLASS32_SHARD']) * 4 + rank) % len(models)
    models = models[offset:] + models[:offset]
    passes = set()
    # Re-evaluate gates between tasks so methods on other nodes enter when ready.
    while True:
        found = False
        for row in rows:
            for model in models:
                gate = ROOT / 'gates' / f'{model}.passed.json'
                if model not in passes and gate.exists():
                    g = campaign.read(gate)
                    campaign.require(g['manifest_id'] == man['manifest_id'] and g['indices'] == man['smoke_indices'],
                                     'Smoke gate contract changed')
                    for idx in g['indices']:
                        campaign.valid_result(ROOT / 'smoke' / model / f'row-{idx:03d}.json', man, model, man['rows'][idx])
                    passes.add(model)
                if model not in passes:
                    continue
                output = ROOT / 'results' / model / f"row-{row['dataset_index']:03d}.json"
                if output.exists():
                    campaign.valid_result(output, man, model, row)
                    continue
                if not claim(man, model, row, owner):
                    continue
                found = True
                launch(man, model, row)
        if not found:
            break
    campaign.atomic(ROOT / 'worker_done' / owner['job'] / f'rank-{rank}.json',
        dict(owner, epoch=time.time(), passed_models=sorted(passes),
             reason='no unclaimed work among currently smoke-passed models; not campaign completion'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('preflight', 'check', 'smoke', 'run'))
    p.add_argument('--shard', type=int, choices=range(4), default=0)
    args = p.parse_args()
    if args.mode != 'check':
        normalized = normalize_visibility(os.environ)
        os.environ.clear()
        os.environ.update(normalized)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.mode == 'preflight':
        preflight()
    elif args.mode == 'check':
        check_preflight(campaign.manifest_load())
    elif args.mode == 'smoke':
        smoke(args.shard)
    else:
        dispatch()


if __name__ == '__main__':
    main()
