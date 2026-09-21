"""Bounded completion of two original, un-finetuned source rows, not FT50.

Uses unchanged eval_one/manifest and canonical exclusive claims. Existing
results, errors and logs are immutable. Only absent canonical results may be
published. A separate attempt directory preserves the retry audit.
"""
import argparse
import fcntl
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

from eval_dispatch import atomic, check_idle, gpu, owned_rss, read, stop_own, GIB
from eval_one import load_manifest

PARENT = '196093'
NODE = 'auh7-1b-gpu-197'
GPU_UUID = '8db07672d655ee5b'
GPU_PCI = '0000:47:00.0'
ROW_IDS = {189: 'TabArena__diamonds', 194: 'TabArena__superconductivity'}


def parent_fields():
    raw = subprocess.check_output(['scontrol', 'show', 'job', PARENT, '-o'], text=True)
    fields = dict(re.findall(r'(\w+)=(\S+)', raw))
    assert fields['JobState'] == 'RUNNING' and fields['NumNodes'] == '1'
    assert fields['UserId'].startswith('guangyi.chen(') and fields['NodeList'] == NODE
    assert 'gres/gpu=8' in fields['AllocTRES'] and 'mem=64G' in fields['AllocTRES']
    return fields


def node_run(manifest_path, attempt):
    import psutil
    assert os.environ['SLURM_JOB_ID'] == PARENT and socket.gethostname() == NODE
    parent_fields()
    root = manifest_path.parent
    g = gpu(GPU_UUID)
    assert g['pci'] == GPU_PCI and check_idle(g), 'Audited physical GPU no longer idle'
    assert owned_rss(psutil) < 40 * GIB and psutil.virtual_memory().available > 24 * GIB
    audit = root / 'source_completion' / attempt
    audit.mkdir(parents=True, exist_ok=False)
    atomic(audit / 'start.json', {'epoch': time.time(), 'parent': PARENT,
        'step': os.environ['SLURM_STEP_ID'], 'node': NODE, 'gpu': g,
        'original_checkpoint_step': 22175, 'rows': ROW_IDS,
        'unchanged_manifest': str(manifest_path), 'proc_cgroup': Path('/proc/self/cgroup').read_text(),
        'own_rss_limit_gib': 10, 'step_reservation_gib': 16,
        'physical_gpu_count_used': 1, 'parent_unchanged': True}, True)
    env = os.environ.copy()
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
        env.pop(key, None)
    env.update(ROCR_VISIBLE_DEVICES='GPU-' + GPU_UUID, EXPECTED_GPU_UUID=GPU_UUID,
        EXPECTED_GPU_PCI_BUS_ID=GPU_PCI, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) >= 4
    current = None
    def interrupted(signum, _frame):
        if current is not None:
            stop_own(current)
        raise RuntimeError('Source completion interrupted: ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    for index, name in ROW_IDS.items():
        man, row, ckpt = load_manifest(manifest_path, 22175, index)
        assert row['dataset'] == name
        result = root / 'results' / 'step-22175' / f'row-{index:03d}.json'
        if result.exists():
            print(json.dumps({'reused': str(result)}), flush=True)
            continue
        claim = root / 'claims' / 'step-22175' / f'row-{index:03d}.json'
        claim.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        identity = {'worker': PARENT + '.' + os.environ['SLURM_STEP_ID'] + '.' + GPU_UUID,
            'pid': os.getpid(), 'node': NODE, 'started_epoch': time.time(),
            'checkpoint_step': 22175, 'dataset_index': index,
            'manifest_id': man['manifest_id'], 'token': token, 'attempt': attempt}
        with claim.open('x') as handle:
            json.dump(identity, handle); handle.flush(); os.fsync(handle.fileno())
        started = time.monotonic()
        try:
            if result.exists():
                continue
            log = audit / f'row-{index:03d}.log'
            cmd = ['taskset', '-c', ','.join(map(str, cpus[:4])), sys.executable,
                str(Path(__file__).with_name('eval_one.py')), '--manifest', str(manifest_path),
                '--checkpoint-step', '22175', '--dataset-index', str(index),
                '--output', str(result), '--threads', '4']
            print(json.dumps({'starting': index, 'dataset': name, 'output': str(result)}), flush=True)
            with log.open('x') as handle:
                current = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
                while current.poll() is None:
                    try:
                        proc = psutil.Process(current.pid)
                        rss = sum(p.memory_info().rss for p in [proc] + proc.children(recursive=True) if p.is_running())
                    except psutil.NoSuchProcess:
                        rss = 0
                    if rss > 10 * GIB or owned_rss(psutil) > 56 * GIB or psutil.virtual_memory().available < 8 * GIB:
                        stop_own(current)
                        raise RuntimeError('Conservative RAM guard stopped only source evaluator')
                    if time.monotonic() - started > 1500:
                        stop_own(current)
                        raise RuntimeError('Per-row 25 minute limit')
                    try:
                        current.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
            assert current.returncode == 0, f'Row {index} failed: {log}'
            obj = read(result)
            assert obj['complete'] and obj['checkpoint_step'] == 22175
            assert obj['checkpoint']['sha256'] == ckpt['sha256'] and obj['input_fingerprint'] == row['input_fingerprint']
            assert obj['manifest_id'] == man['manifest_id'] and obj['dataset_index'] == index
            atomic(audit / f'row-{index:03d}.receipt.json', {**identity, 'complete': True,
                'result': str(result), 'elapsed_s': time.monotonic() - started,
                'existing_error_preserved': (root / 'errors' / 'step-22175' / f'row-{index:03d}.json').exists()}, True)
            print(json.dumps({'completed': index, 'metrics': obj['metrics']}), flush=True)
        finally:
            if current is not None:
                stop_own(current)
            if claim.exists() and read(claim).get('token') == token:
                claim.unlink()
    atomic(audit / 'finished.json', {'epoch': time.time(), 'complete': True, 'checkpoint_step': 22175}, True)


def launch(manifest_path, attempt):
    parent_fields()
    steps = subprocess.check_output(['squeue', '--steps', '-j', PARENT, '-h', '-o', '%i|%j'], text=True)
    assert not any(x in steps for x in ('regsrc22175', 'regft50eval')), 'Evaluation already active on chosen parent'
    for index in ROW_IDS:
        load_manifest(manifest_path, 22175, index)
    root = manifest_path.parent
    folder = root / 'source_completion'
    folder.mkdir(exist_ok=True)
    with (folder / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log = folder / (attempt + '.launch.log')
        command = ['srun', '--jobid=' + PARENT, '--overlap', '--exact', '-N1', '-n1', '-c4',
            '--mem=16G', '--gpus=8', '--gpu-bind=none', '--time=01:00:00',
            '--unbuffered', '--job-name=regsrc22175', sys.executable, str(Path(__file__).resolve()),
            'node', '--manifest', str(manifest_path), '--attempt', attempt]
        # Eight devices are exposed by the existing whole-node allocation; the
        # evaluator masks to exactly ONE audited UUID. No allocation is added.
        with log.open('x') as handle:
            proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        receipt = {'epoch': time.time(), 'parent': PARENT, 'node': NODE, 'launcher_pid': proc.pid,
            'command': command, 'physical_evaluation_gpus': [GPU_UUID], 'log': str(log),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
        atomic(folder / (attempt + '.launch.json'), receipt, True)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['launch', 'node'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.attempt)
    {'launch': launch, 'node': node_run}[args.mode](args.manifest.resolve(), args.attempt)
