"""Full-eight-GPU Slurm allocation -> audited physical UUID rank binding.

No partial allocation is accepted. Bootstrap runs as one real Slurm task with
access to the complete eight-GPU allocation, after verifying scontrol and the
node's eight physical DRM devices. Each subsequent real Slurm rank receives
one observed UUID; no scheduler ordinal is interpreted as a physical ordinal.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

from table6_restart_deadline import parse_fields, derive_deadline
from table6_restart_ag import publish

MASKS = ('ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def memory_bytes(value):
    match = re.fullmatch(r'([0-9]+)([KMGT])(?:B)?', value or '')
    require(match is not None, 'Unparseable Slurm allocated memory')
    return int(match.group(1))*1024**{'K': 1, 'M': 2, 'G': 3, 'T': 4}[match.group(2)]


def allocation(raw, *, job, node, cpus, mem_gib, uid=None):
    fields = parse_fields(raw)
    uid = os.getuid() if uid is None else uid
    tres = dict(item.split('=', 1) for item in fields.get('AllocTRES', '').split(',') if '=' in item)
    require(fields.get('JobId') == str(job) and fields.get('JobState') == 'RUNNING' and
            fields.get('NumNodes') == '1' and fields.get('NodeList') == node and
            fields.get('UserId', '').endswith('('+str(uid)+')'), 'Wrong owned running one-node allocation')
    require(tres.get('gres/gpu') == '8' and tres.get('cpu') == str(cpus) and
            memory_bytes(tres.get('mem')) == mem_gib*1024**3, 'Requires exact planned CPU/RAM and all eight allocated GPUs')
    return fields


def query_allocation():
    start = time.monotonic()
    raw = subprocess.run(['scontrol', 'show', 'job', '-o', os.environ['SLURM_JOB_ID']],
                         capture_output=True, text=True, check=True, timeout=30,
                         env={**os.environ, 'TZ': 'UTC'}).stdout
    return raw, start, time.monotonic()


def validate_devices(gpus):
    require(len(gpus) == 8 and len({g['uuid'] for g in gpus}) == 8 and
            len({g['pci'] for g in gpus}) == 8, 'Exactly eight unique physical UUIDs/PCIs required')
    require(all(re.fullmatch(r'[0-9a-f]{16}', g['uuid']) and int(g['uuid'], 16) > 0 and
                re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', g['pci']) for g in gpus),
            'Invalid physical GPU identity')
    return sorted(gpus, key=lambda g: g['pci'])


def inventory():
    devices = {}
    for card in Path('/sys/class/drm').glob('card[0-9]*'):
        if not re.fullmatch(r'card[0-9]+', card.name):
            continue
        device = (card/'device').resolve()
        if (device/'unique_id').is_file():
            devices[device.name] = {'pci': device.name.lower(),
                                    'uuid': (device/'unique_id').read_text().strip().lower()}
    return validate_devices(list(devices.values()))


def physical_runtime_devices():
    import torch
    from pfn_mitra_one import select_loaded_hip_library
    require(torch.version.hip and torch.cuda.is_available() and torch.cuda.device_count() == 8,
            'Bootstrap must actually see all eight allocated AMD GPUs')
    torch.cuda.init()
    library = select_loaded_hip_library(Path('/proc/self/maps').read_text())
    require(hasattr(os, 'RTLD_NOLOAD'), 'HIP inspection requires already-loaded runtime only')
    lib = ctypes.CDLL(str(library), mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
    fn = lib.hipDeviceGetPCIBusId
    fn.argtypes, fn.restype = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int], ctypes.c_int
    result = []
    for index in range(8):
        buffer = ctypes.create_string_buffer(64)
        require(fn(buffer, len(buffer), index) == 0, 'HIP physical PCI lookup failed')
        domain, bus, slot = buffer.value.decode().lower().split(':')
        pci = f'{int(domain, 16):04x}:{bus}:{slot}'
        uuid = (Path('/sys/bus/pci/devices')/pci/'unique_id').read_text().strip().lower()
        with torch.cuda.device(index):
            tensor = torch.ones((4, 4), device='cuda:'+str(index))
            require(float(tensor.sum().cpu()) == 16, 'Bootstrap GPU compute probe failed')
            del tensor
        result.append({'pci': pci, 'uuid': uuid, 'bootstrap_runtime_index': index,
                       'name': torch.cuda.get_device_name(index), 'hip_version': str(torch.version.hip)})
    return validate_devices(result)


def bootstrap(path, *, expected_cpus=64, expected_mem_gib=256, expected_seconds=259200):
    require(os.environ.get('SLURM_NTASKS') == '1' and os.environ.get('SLURM_PROCID') == '0' and
            os.environ.get('SLURM_LOCALID') == '0' and os.environ.get('SLURM_STEP_ID', '').isdigit(),
            'Bootstrap requires one genuine Slurm task, not a batch/login process')
    node, job = socket.gethostname(), os.environ.get('SLURM_JOB_ID', '')
    raw, started, finished = query_allocation()
    observed_epoch = time.time()
    fields = allocation(raw, job=job, node=node, cpus=expected_cpus, mem_gib=expected_mem_gib)
    sysfs = inventory()  # Verify full physical-eight node before clearing aliases.
    original = {key: os.environ.get(key) for key in MASKS}
    for key in MASKS:
        os.environ.pop(key, None)
    private = tempfile.mkdtemp(prefix='alloc8-', dir='/tmp')
    os.environ['TMPDIR'] = private
    tempfile.tempdir = None
    actual = physical_runtime_devices()
    require({(g['pci'], g['uuid']) for g in actual} == {(g['pci'], g['uuid']) for g in sysfs},
            'Runtime GPUs differ from the complete allocated physical node')
    # One shared same-node monotonic budget, never renewed by later ranks.
    deadline = derive_deadline(raw, job_id=job, query_started_monotonic=started,
                               query_finished_monotonic=finished, observed_local_epoch=observed_epoch,
                               expected_limit_seconds=expected_seconds)
    record = {'schema': 'allocated_gpu_uuid_v1', 'job': job, 'node': node, 'gpus': actual,
              'expected_cpus': expected_cpus, 'expected_mem_gib': expected_mem_gib,
              'expected_seconds': expected_seconds, 'allocation': fields,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'original_visibility': original, 'bootstrap_private_TMPDIR': private,
              'all_eight_physical_gpus_verified': True, 'deadline': deadline.record(), 'epoch': time.time()}
    record['mapping_id'] = digest(record)
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink() and not path.parent.is_symlink(), 'Unsafe mapping output')
    publish(path, record)
    return record


def load_mapping(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'Missing absolute immutable mapping')
    value = json.loads(path.read_text())
    require(value['mapping_id'] == digest({k:v for k,v in value.items() if k != 'mapping_id'}), 'Mapping digest changed')
    require(value['source_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'Mapping helper source changed')
    require(value['schema'] == 'allocated_gpu_uuid_v1' and value['all_eight_physical_gpus_verified'] is True,
            'Mapping lacks full-allocation runtime proof')
    require(value['gpus'] == validate_devices(value['gpus']), 'Physical mapping order changed')
    return value


def rank_environment(mapping, env, *, hostname=None):
    node = socket.gethostname() if hostname is None else hostname
    require(env.get('SLURM_JOB_ID') == mapping['job'] and node == mapping['node'], 'Mapping belongs to another job/node')
    require(env.get('SLURM_NTASKS') == '8' and env.get('SLURM_PROCID') == env.get('SLURM_LOCALID') and
            env.get('SLURM_PROCID') in {str(i) for i in range(8)} and env.get('SLURM_STEP_ID', '').isdigit(),
            'Exactly eight genuine one-node Slurm ranks required')
    rank = int(env['SLURM_PROCID'])
    device = mapping['gpus'][rank]
    result = dict(env)
    for key in MASKS:
        result.pop(key, None)
    result.update(ROCR_VISIBLE_DEVICES='GPU-'+device['uuid'], EXPECTED_GPU_UUID=device['uuid'],
                  EXPECTED_GPU_PCI_BUS_ID=device['pci'], ALLOCATED_GPU_MAPPING_ID=mapping['mapping_id'],
                  **mapping['deadline']['environment'])
    return result


def exec_bound(mapping_path, command):
    mapping = load_mapping(mapping_path)
    raw, _, _ = query_allocation()
    allocation(raw, job=mapping['job'], node=mapping['node'], cpus=mapping['expected_cpus'], mem_gib=mapping['expected_mem_gib'])
    env = rank_environment(mapping, os.environ)
    require(command and Path(command[0]).is_absolute() and Path(command[0]).is_file(), 'Absolute executable required')
    os.execvpe(command[0], command, env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    boot = sub.add_parser('bootstrap'); boot.add_argument('--mapping', type=Path, required=True)
    boot.add_argument('--expected-cpus', type=int, default=64)
    boot.add_argument('--expected-mem-gib', type=int, default=256)
    boot.add_argument('--expected-seconds', type=int, default=259200)
    bound = sub.add_parser('exec'); bound.add_argument('--mapping', type=Path, required=True)
    bound.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.mode == 'bootstrap':
        print(json.dumps(bootstrap(args.mapping, expected_cpus=args.expected_cpus,
                                   expected_mem_gib=args.expected_mem_gib, expected_seconds=args.expected_seconds)), flush=True)
    else:
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        exec_bound(args.mapping, command)


if __name__ == '__main__':
    main()
