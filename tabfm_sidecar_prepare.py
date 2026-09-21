"""Read-only resource snapshots and a pinned plan for one idle TabFM GPU."""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from tabfm_default_dispatch import atomic, digest, read, require
from tabfm_prepare import OUT, identity

PARENT = '196092'
NODE = 'auh7-1b-gpu-196'
UUID = '8b8b827ace9944c1'
PCI = '0000:88:00.0'
SIDECAR = 'existing196092-single-20260922-v1'
SIDECAR_SCRIPT = 'tabfm_existing_sidecar.py'


def cpu_snapshot():
    import psutil
    rows = []
    for p in psutil.process_iter(['pid', 'uids', 'create_time', 'cpu_times']):
        try:
            if p.info['uids'].real == os.getuid():
                t = p.info['cpu_times']
                rows.append({'pid': p.pid, 'created': p.info['create_time'],
                             'cpu_seconds': t.user + t.system})
        except psutil.NoSuchProcess:
            continue
    return {'epoch': time.time(), 'node': socket.gethostname(),
            'parent': os.environ['SLURM_JOB_ID'], 'processes': rows}


def check_parent(record):
    f = record['job_fields']
    require(record['parent'] == PARENT and record['node'] == NODE, 'Wrong resource target')
    require(f['JobId'] == PARENT and f['JobState'] == 'RUNNING' and
            f['UserId'].startswith('guangyi.chen(') and f['NumNodes'] == '1', 'Parent not eligible')
    require(f['NodeList'] == NODE and f['NumCPUs'] == '64' and
            set(('mem=64G', 'gres/gpu=8')).issubset(f['AllocTRES'].split(',')),
            'Parent GPU/CPU/memory ownership differs')
    gpu = [g for g in record['gpus'] if g.get('uuid') == UUID]
    require(len(gpu) == 1 and gpu[0]['pci'] == PCI and gpu[0]['busy'] == 0
            and gpu[0]['vram'] < 128 * 1024**2, 'Selected physical GPU is not empty')
    rss = sum(p['rss'] for p in record['owned_processes'])
    require(rss <= 20 * 1024**3, 'Existing allocations leave insufficient memory headroom')
    return gpu[0], rss


def probe(man, root, label):
    from probe_eval_capacity import inspect_parent
    record = inspect_parent(PARENT)
    check_parent(record)
    cmd = ['srun', '--jobid=' + PARENT, '--overlap', '--exact', '-N1', '-n1', '-c1',
           '--mem=1G', '--gres=none', '--gpus=0', '--time=00:02:00',
           sys.executable, str(Path(__file__).resolve()), 'cpu-snapshot']
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=90, check=True)
    record['cpu_sample'] = json.loads(p.stdout.strip().splitlines()[-1])
    require(record['cpu_sample']['node'] == NODE and record['cpu_sample']['parent'] == PARENT,
            'CPU sample belongs to wrong node/allocation')
    atomic(man, root / ('probe-' + label + '.json'), record)
    print(json.dumps({'probe': label, 'epoch': record['epoch'],
                      'gpu': check_parent(record)[0], 'other_rss': check_parent(record)[1]}))


def prepare(man, root):
    a, b = [read(root / ('probe-' + n + '.json')) for n in ('1', '2')]
    for r in (a, b):
        check_parent(r)
    require(15 <= b['epoch'] - a['epoch'] <= 900 and time.time() - b['epoch'] <= 180,
            'Need two recent separated observations')
    before, after = a['cpu_sample'], b['cpu_sample']
    elapsed = after['epoch'] - before['epoch']
    require(elapsed >= 15, 'CPU sampling interval too short')
    old = {(r['pid'], r['created']): r['cpu_seconds'] for r in before['processes']}
    cpu = 0.0
    for r in after['processes']:
        key = r['pid'], r['created']
        require(key in old or r['created'] >= before['epoch'] - 1,
                'Cannot account for existing CPU process')
        cpu += max(0.0, r['cpu_seconds'] - old.get(key, 0.0))
    free = 64 - cpu / elapsed - 4  # reserve four additional cores as guard
    require(free >= 4, 'No safe four-CPU headroom')
    f = b['job_fields']
    end = datetime.datetime.fromisoformat(f['EndTime']).replace(tzinfo=datetime.timezone.utc).timestamp()
    deadline = min(end - 600, time.time() + 18 * 3600)
    require(deadline - time.time() > 3 * 3600, 'Parent has insufficient lifetime')
    plan = {'sidecar_id': SIDECAR, 'campaign_path': str(OUT / 'manifest.json'),
            'parent_job_id': PARENT, 'node': NODE, 'gpus': [{'uuid': UUID, 'pci': PCI}],
            'cpu_count': 4, 'mem_gib': 40, 'max_lanes': 1, 'parent_gpu_count': 8,
            'parent_mem_gib': 64, 'deadline_epoch': deadline, 'probe_epoch': b['epoch'],
            'sidecar_source': identity(Path(__file__).with_name(SIDECAR_SCRIPT)),
            'external_idle_probes': [{'epoch': r['epoch'], 'uuid': UUID, 'pci': PCI,
              'used_vram_bytes': check_parent(r)[0]['vram'], 'gpu_busy_percent': 0} for r in (a, b)],
            'startup_other_rss_bytes': check_parent(b)[1], 'free_cpu_cores': free,
            'all8_gpu_reservation_verified': True, 'resources_available_verified': True,
            'inherited_cpu_count': 64 if SIDECAR_SCRIPT.endswith(('_v2.py', '_v3.py')) else 4,
            'resource_budget_note': 'One existing GPU; extra guard32GiB own /60GiB sameuid total; model defaults unchanged',
            'cpu_observation': {'seconds': elapsed, 'existing_cpu_cores': cpu / elapsed}}
    if SIDECAR_SCRIPT.endswith('_v3.py'):
        plan['source_records'] = [identity(Path(__file__).with_name('tabfm_local_tmp.py'))]
        plan['runtime_TMPDIR_override'] = 'private tempfile.mkdtemp(prefix=tfm-, dir=/tmp); only TMPDIR changes'
    plan['plan_id'] = digest(plan)
    atomic(man, root / 'plan.json', plan)
    print(json.dumps(plan, sort_keys=True))


def main():
    global SIDECAR, SIDECAR_SCRIPT
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['cpu-snapshot', 'probe1', 'probe2', 'prepare', 'launch'])
    versions = parser.add_mutually_exclusive_group()
    versions.add_argument('--cpu-binding-v2', action='store_true')
    versions.add_argument('--local-tmp-v3', action='store_true')
    args = parser.parse_args()
    if args.cpu_binding_v2:
        SIDECAR = 'existing196092-single-20260922-v2'
        SIDECAR_SCRIPT = 'tabfm_existing_sidecar_v2.py'
    elif args.local_tmp_v3:
        SIDECAR = 'existing196092-single-20260922-v3'
        SIDECAR_SCRIPT = 'tabfm_existing_sidecar_v3.py'
    if args.mode == 'cpu-snapshot':
        print(json.dumps(cpu_snapshot()))
        return
    man = read(OUT / 'manifest.json')
    root = OUT / 'sidecars' / SIDECAR
    if args.mode.startswith('probe'):
        probe(man, root, args.mode[-1])
    elif args.mode == 'prepare':
        prepare(man, root)
    else:
        plan = read(root / 'plan.json')
        require(time.time() - plan['probe_epoch'] < 300, 'Launch probe stale')
        cmd = [man['worker_python'], str(plan['sidecar_source']['path']),
               '--plan', str(root / 'plan.json')]
        atomic(man, root / 'launcher-attempt.json', {'epoch': time.time(), 'command': cmd,
                                                  'plan_id': plan['plan_id']})
        with (root / 'launcher.log').open('x') as handle:
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        atomic(man, root / 'launcher-started.json', {'epoch': time.time(), 'pid': child.pid,
                                                  'plan_id': plan['plan_id'], 'command': cmd})
        print(json.dumps({'pid': child.pid, 'log': str(root / 'launcher.log')}))


if __name__ == '__main__':
    main()
