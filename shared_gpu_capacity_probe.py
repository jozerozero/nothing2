"""Read-only /proc + DRM ownership evidence for explicit existing allocations.

Invoke --sample twice, at least 15 seconds apart, using NEW output paths. No
automatic inter-sample wait or model launch exists. CPU utilization alone uses
two bounded one-second measurement windows. Hardware-idle is deliberately NOT
declared safe to borrow: process masks/open FDs do not prove future intentions.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import uuid

PARENTS = ('196093', '200797', '204828', '204827', '204826', '194259')
PYTHON = '/vast/users/guangyi.chen/anaconda3/envs/tabicl/bin/python'
GIB = 1024 ** 3
ENV_KEYS = frozenset(('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
                      'GPU_DEVICE_ORDINAL', 'SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_PROCID',
                      'SLURM_LOCALID', 'SLURM_NODEID', 'SLURM_NTASKS', 'SLURM_CPUS_PER_TASK'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def publish(path, value):
    path = Path(path)
    require(not path.is_symlink(), 'refuse symlink output')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def selected_environment(raw):
    selected = {}
    for item in raw.split(b'\0'):
        if b'=' not in item:
            continue
        name, value = item.split(b'=', 1)
        name = name.decode('ascii', errors='ignore')
        if name in ENV_KEYS:
            require(name not in selected, f'duplicate selected environment key: {name}')
            selected[name] = value.decode('utf-8', errors='replace')
    return selected


def parsed_fields(raw):
    fields = {}
    for key, value in re.findall(r'(?:^|\s)([^\s=]+)=([^\s]+)', raw):
        require(key not in fields, f'duplicate allocation field: {key}')
        fields[key] = value
    return fields


def bytes_from_slurm(value):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)', value)
    require(match is not None, f'unrecognized Slurm memory: {value}')
    return int(float(match.group(1)) * 1024 ** {'': 2, 'K': 1, 'M': 2, 'G': 3, 'T': 4}[match.group(2)])


def parent_record(raw, parent, uid=None):
    require(parent in PARENTS, 'parent outside explicit allowlist')
    fields = parsed_fields(raw)
    require(fields.get('JobId') == parent and fields.get('JobState') == 'RUNNING'
            and fields.get('NumNodes') == '1', 'parent must remain exact one-node running allocation')
    uid = os.getuid() if uid is None else uid
    require(re.fullmatch(r'[^()]+\(' + str(uid) + r'\)', fields.get('UserId', '')) is not None,
            'parent allocation not owned by this uid')
    tres = dict(part.split('=', 1) for part in fields.get('AllocTRES', '').split(',') if '=' in part)
    require(int(tres.get('gres/gpu', '0')) == 8 and bytes_from_slurm(tres.get('mem', '0')) == 64 * GIB,
            'parent does not match eight-GPU / 64GiB proof scope')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', fields.get('NodeList', '')) is not None, 'invalid single node')
    return fields


def run(args, *, timeout=20, env=None):
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, env=env)
    require(result.returncode == 0, f'{args[0]} rc={result.returncode}: {result.stderr[-2000:]}')
    return result.stdout


def probe_environment():
    prefixes = ('SLURM_', 'SBATCH_', 'SRUN_', 'PMI_', 'PMIX_', 'OMPI_')
    env = {key: value for key, value in os.environ.items() if not key.startswith(prefixes)}
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def probe_command(parent, node):
    require(parent in PARENTS, 'parent outside explicit allowlist')
    # --overlap is exclusively for this tiny read-only observation step, not
    # authorization to overlap any subsequent model computation.
    return ['srun', '--jobid=' + parent, '--nodelist=' + node, '--overlap', '--exact',
            '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=1G', '--gpus=0',
            '--gpus-per-task=0', '--gres=none', '--time=00:02:00', '--immediate=10',
            '--input=none', '--export=ALL', 'env', 'CUDA_VISIBLE_DEVICES=',
            'HIP_VISIBLE_DEVICES=-1', 'ROCR_VISIBLE_DEVICES=-1', 'GPU_DEVICE_ORDINAL=-1',
            'nice', '-n', '19', PYTHON, '-B', str(Path(__file__).resolve()),
            '--node', '--parent', parent, '--expected-node', node]


def collect_gpus(drm_root=Path('/sys/class/drm')):
    devices = {}
    fd_devices = {}
    for card in sorted(drm_root.iterdir()):
        if re.fullmatch(r'card\d+', card.name) is None:
            continue
        device = (card / 'device').resolve()
        if not (device / 'unique_id').exists():
            continue
        pci = device.name.lower()
        require(re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', pci), 'unrecognized GPU PCI identity')
        uuid_value = (device / 'unique_id').read_text().strip().lower()
        require(uuid_value and uuid_value not in {d['uuid'] for d in devices.values()}, 'missing/duplicate GPU UUID')
        busy = int((device / 'gpu_busy_percent').read_text())
        used = int((device / 'mem_info_vram_used').read_text())
        total = int((device / 'mem_info_vram_total').read_text())
        require(0 <= busy <= 100 and 0 <= used <= total and total > 0, 'invalid GPU utilization')
        aliases = sorted(p.name for p in (device / 'drm').iterdir()
                         if re.fullmatch(r'(?:card\d+|renderD\d+)', p.name))
        require(card.name in aliases, 'DRM card aliases incomplete')
        value = {'uuid': uuid_value, 'pci': pci, 'card': card.name, 'drm_nodes': aliases,
                 'busy_percent': busy, 'vram_used_bytes': used, 'vram_total_bytes': total,
                 'hardware_idle': busy == 0 and used < 128 * 1024 ** 2}
        require(pci not in devices, 'duplicate PCI card mapping')
        devices[pci] = value
        for alias in aliases:
            require('/dev/dri/' + alias not in fd_devices, 'duplicate DRM fd alias')
            fd_devices['/dev/dri/' + alias] = {'pci': pci, 'uuid': uuid_value}
    require(len(devices) == 8, f'expected exactly eight identifiable DRM GPUs, got {len(devices)}')
    return list(devices.values()), fd_devices


def read_stat(path):
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'state': fields[0], 'parent_pid': int(fields[1]), 'start_ticks': int(fields[19]),
            'rss_bytes': int(fields[21]) * os.sysconf('SC_PAGE_SIZE')}


def drm_fds(path, fd_devices):
    records = []
    for fd in (path / 'fd').iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target in fd_devices:
            records.append({'fd': int(fd.name), 'path': target, **fd_devices[target]})
        elif target == '/dev/kfd':
            records.append({'fd': int(fd.name), 'path': target, 'pci': None, 'uuid': None,
                            'note': 'KFD handle alone does not identify a particular GPU'})
        elif target.startswith('/dev/dri/'):
            raise RuntimeError('open DRM FD missing from observed device map')
    return sorted(records, key=lambda item: item['fd'])


def collect_processes(fd_devices, proc_root=Path('/proc')):
    own, other_gpu_owners, errors, unknown_other = [], [], [], []
    uid = os.getuid()
    for path in sorted(proc_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not path.name.isdigit():
            continue
        pid = int(path.name)
        try:
            process_uid = path.stat().st_uid
            initial = read_stat(path)
            if initial['state'] == 'Z':
                continue
            if process_uid != uid:
                # No foreign command line/environment is collected. Inability
                # to inspect its FDs is explicit uncertainty, never "no owner".
                try:
                    descriptors = drm_fds(path, fd_devices)
                    if descriptors:
                        other_gpu_owners.append({'pid': pid, 'uid': process_uid, 'gpu_fds': descriptors})
                except PermissionError:
                    unknown_other.append({'pid': pid, 'uid': process_uid, 'reason': 'foreign_fd_permission_denied'})
                continue
            record = {'pid': pid, 'uid': process_uid, **initial,
                      'cmdline': [v.decode('utf-8', errors='replace') for v in (path / 'cmdline').read_bytes().split(b'\0') if v],
                      'selected_environment': selected_environment((path / 'environ').read_bytes()),
                      'cgroup': (path / 'cgroup').read_text().splitlines(),
                      'cpu_affinity': sorted(os.sched_getaffinity(pid)),
                      'gpu_fds': drm_fds(path, fd_devices), 'is_probe': pid == os.getpid()}
            after = read_stat(path)
            require(after['start_ticks'] == initial['start_ticks'], 'PID identity changed during observation')
            own.append(record)
        except (FileNotFoundError, ProcessLookupError):
            continue  # Process exited; no claim about its continued ownership.
        except Exception as exc:
            errors.append({'pid': pid, 'error': type(exc).__name__ + ': ' + str(exc)})
    return {'own_processes': own, 'foreign_gpu_fd_owners': other_gpu_owners,
            'foreign_fd_inspection_unknown': unknown_other, 'process_errors': errors,
            'same_uid_rss_bytes': sum(p['rss_bytes'] for p in own),
            'other_same_uid_rss_bytes': sum(p['rss_bytes'] for p in own if not p['is_probe']),
            'own_process_inspection_complete': not errors,
            'all_uid_gpu_fd_inspection_complete': not errors and not unknown_other}


def mask_evidence(process, gpus):
    """Only explicit UUIDs identify hardware; ordinal namespaces are ambiguous."""
    environment = process['selected_environment']
    mask = environment.get('ROCR_VISIBLE_DEVICES')
    known = {g['uuid'].lower() for g in gpus}
    if mask is not None and mask.strip() in ('', '-1'):
        return {'kind': 'explicit_gpu_disabled', 'uuids': []}
    if mask:
        tokens = mask.split(',')
        uuids = [token[4:].lower() for token in tokens if token.startswith('GPU-')]
        if len(uuids) == len(tokens) and len(set(uuids)) == len(uuids) and set(uuids) <= known:
            aliases = [environment.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')]
            if any(v not in (None, '', '0', '-1', mask) for v in aliases):
                return {'kind': 'ambiguous_alias_filters', 'uuids': uuids}
            return {'kind': 'explicit_rocr_uuid_mask', 'uuids': uuids}
        return {'kind': 'ambiguous_ordinal_or_unknown_mask', 'uuids': []}
    return {'kind': 'unrestricted_or_unknown_future_gpu_access', 'uuids': []}


def ownership_evidence(gpu, processes, gpus):
    direct, enumeration, ambiguous = [], [], []
    for process in processes:
        if process.get('is_probe'):
            continue
        evidence = mask_evidence(process, gpus)
        descriptors = any(fd['uuid'] == gpu['uuid'] for fd in process['gpu_fds'])
        if evidence['kind'] == 'explicit_rocr_uuid_mask':
            if gpu['uuid'] in evidence['uuids']:
                direct.append(process['pid'])
            elif descriptors:
                enumeration.append(process['pid'])
        elif evidence['kind'] == 'explicit_gpu_disabled':
            if descriptors:
                ambiguous.append(process['pid'])  # Environment and current handle conflict.
        else:
            ambiguous.append(process['pid'])
    return {'explicit_target_mask_pids': direct, 'other_uuid_driver_enumeration_fd_pids': enumeration,
            'ambiguous_or_future_owner_pids': ambiguous,
            'ownership_proven': False, 'requires_future_work_review': True}


def node_sample(parent, expected_node):
    require(parent in PARENTS and os.environ.get('SLURM_JOB_ID') == parent, 'wrong node probe allocation')
    require(socket.gethostname() == expected_node and os.environ.get('SLURM_STEP_ID', '').isdigit(), 'wrong node/step identity')
    require(os.environ.get('SLURM_NTASKS') == '1' and os.environ.get('SLURM_CPUS_PER_TASK') == '1', 'probe must use exactly one CPU task')
    started = time.time()
    raw = run(['scontrol', 'show', 'job', '-o', parent])
    allocation = parent_record(raw, parent)
    require(allocation['NodeList'] == expected_node, 'parent node changed')
    gpus, fd_devices = collect_gpus()
    processes = collect_processes(fd_devices)
    import psutil
    # Two bounded measurements, not an automatic second capacity sample.
    cpu = [psutil.cpu_percent(interval=1.0, percpu=True) for _ in range(2)]
    require(len(cpu[0]) == len(cpu[1]) > 0 and all(0 <= x <= 100 for row in cpu for x in row), 'invalid CPU counters')
    memory = psutil.virtual_memory()
    # A one-CPU probe's own mask is not the full allocation mask. Use only
    # cgroup-verified parent batch processes as evidence for its accessible CPUs.
    batch = [p for p in processes['own_processes'] if any(
        f'/job_{parent}/' in cgroup and '/step_batch/' in cgroup for cgroup in p['cgroup'])]
    parent_cpus = sorted({cpu_id for p in batch for cpu_id in p['cpu_affinity']})
    idle_cpus = [cpu_id for cpu_id in parent_cpus if all(cpu_id < len(row) and row[cpu_id] < 25 for row in cpu)]
    for gpu in gpus:
        gpu['same_uid_fd_owner_pids'] = [p['pid'] for p in processes['own_processes']
                                       if any(fd['uuid'] == gpu['uuid'] for fd in p['gpu_fds'])]
        gpu['foreign_fd_owner_pids'] = [p['pid'] for p in processes['foreign_gpu_fd_owners']
                                      if any(fd['uuid'] == gpu['uuid'] for fd in p['gpu_fds'])]
        gpu['ownership_evidence'] = ownership_evidence(gpu, processes['own_processes'], gpus)
    return {'parent': parent, 'node': expected_node, 'step_id': os.environ['SLURM_STEP_ID'],
            'started_epoch': started, 'epoch': time.time(), 'allocation': allocation, 'job_fields': allocation,
            'probe_pid': os.getpid(), 'probe_cpu_affinity': sorted(os.sched_getaffinity(0)),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'gpus': gpus, 'cpu_percent_samples': cpu, 'cpu_sample_window_seconds': 1.0,
            'cpu_busy_equivalents': [sum(row) / 100 for row in cpu],
            'parent_batch_cpu_ids': parent_cpus, 'idle_parent_cpu_ids_twice': idle_cpus,
            'parent_cpu_proof_complete': bool(parent_cpus), 'idle_cpu_threshold_percent_exclusive': 25,
            'node_memory_available_bytes': memory.available, 'node_memory_total_bytes': memory.total,
            **processes, 'owned_processes': processes['own_processes'],
            'complete': processes['own_process_inspection_complete'],
            'ownership_proven': False, 'evaluation_launch_authorized': False,
            'ownership_caveat': 'Hardware idle, ROCR masks, and open DRM FDs are observations only; existing batch/process future GPU work requires separate review.'}


def sample_parent(parent):
    try:
        env = probe_environment()
        allocation = parent_record(run(['scontrol', 'show', 'job', '-o', parent], env=env), parent)
        node = allocation['NodeList']
        output = run(probe_command(parent, node), timeout=150, env=env)
        payloads = [line.removeprefix('CAPACITY_JSON=') for line in output.splitlines() if line.startswith('CAPACITY_JSON=')]
        require(len(payloads) == 1, 'probe returned no unique JSON receipt')
        record = json.loads(payloads[0])
        require(record['parent'] == parent and record['node'] == node, 'probe receipt target mismatch')
        require(record['source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'node source differs from controller')
        return record
    except Exception as exc:
        return {'parent': parent, 'complete': False, 'ownership_proven': False,
                'evaluation_launch_authorized': False, 'error': type(exc).__name__ + ': ' + str(exc)}


def compare_samples(first, second):
    require(first['parents_requested'] == second['parents_requested'] == list(PARENTS), 'different target allowlists')
    require(second['started_epoch'] - first['epoch'] >= 15, 'separate samples must be at least 15 seconds apart')
    require(first['source_sha256'] == second['source_sha256'], 'probe source changed between samples')
    a, b = {p['parent']: p for p in first['parents']}, {p['parent']: p for p in second['parents']}
    rows = []
    for parent in PARENTS:
        left, right = a[parent], b[parent]
        row = {'parent': parent, 'ownership_proven': False, 'evaluation_launch_authorized': False,
               'eligible_gpus': []}
        if not left.get('complete') or not right.get('complete'):
            rows.append({**row, 'complete': False, 'reason': 'incomplete node inspection'})
            continue
        require(left['node'] == right['node'], 'parent moved nodes between observations')
        old = {g['uuid']: g for g in left['gpus']}
        stable = [g for g in right['gpus'] if g['uuid'] in old and old[g['uuid']]['pci'] == g['pci']
                  and old[g['uuid']]['hardware_idle'] and g['hardware_idle']]
        rows.append({**row, 'complete': True, 'node': right['node'],
                     'hardware_idle_twice': [{'uuid': g['uuid'], 'pci': g['pci'],
                                             'same_uid_fd_owner_pids': g['same_uid_fd_owner_pids'],
                                             'foreign_fd_owner_pids': g['foreign_fd_owner_pids'],
                                             'ownership_evidence': g.get('ownership_evidence', {})} for g in stable],
                     'idle_parent_cpu_ids': sorted(set(left.get('idle_parent_cpu_ids_twice', [])) &
                                                   set(right.get('idle_parent_cpu_ids_twice', []))),
                     'same_uid_rss_bytes': right['same_uid_rss_bytes'],
                     'all_uid_gpu_fd_inspection_complete': left['all_uid_gpu_fd_inspection_complete'] and right['all_uid_gpu_fd_inspection_complete'],
                     'reason': 'Review full process command/mask/cgroup/FD evidence before any borrowing decision'})
    return {'schema': 'shared_gpu_capacity_comparison_v1', 'parents': rows,
            'first_epoch': first['epoch'], 'second_epoch': second['epoch'],
            'timestamp_clock': 'login controller; never compare node epochs to login freshness',
            'evaluation_launch_authorized': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sample', type=Path)
    group.add_argument('--compare', type=Path, nargs=2)
    group.add_argument('--node', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--parent', choices=PARENTS)
    parser.add_argument('--expected-node')
    args = parser.parse_args(argv)
    if args.node:
        print('CAPACITY_JSON=' + json.dumps(node_sample(args.parent, args.expected_node), allow_nan=False))
    elif args.compare:
        require(args.output is not None, '--compare requires NEW --output')
        publish(args.output, compare_samples(*(json.loads(p.read_text()) for p in args.compare)))
    else:
        require(not args.sample.exists(), 'sample output already exists; no repeated probe')
        # Reserve a distinct intent BEFORE any srun. Unknown/failed invocation is
        # retained and never automatically retried under the same output name.
        intent = args.sample.with_name(args.sample.name + '.intent.json')
        started = time.time()
        publish(intent, {'started_epoch': started, 'parents': list(PARENTS), 'sample': str(args.sample.resolve())})
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            parents = list(pool.map(sample_parent, PARENTS))
        publish(args.sample, {'schema': 'shared_gpu_capacity_sample_v1', 'started_epoch': started,
                'epoch': time.time(), 'parents_requested': list(PARENTS), 'parents': parents,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'complete': all(p['complete'] for p in parents), 'evaluation_launch_authorized': False})
    return 0


if __name__ == '__main__':
    sys.exit(main())
