import json
import os
import socket
import torch
from common import OUTPUT, atomic_json

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert int(os.environ['SLURM_NTASKS']) == 8
assert int(os.environ['SLURM_CPUS_PER_TASK']) == 4
torch.cuda.set_device(0)
assert torch.isfinite(torch.ones((16, 16), device='cuda') @ torch.ones((16, 16), device='cuda')).all()
receipt = {'complete': True, 'rank': int(os.environ['SLURM_PROCID']), 'host': socket.gethostname(),
           'device_count': torch.cuda.device_count(), 'device_name': torch.cuda.get_device_name(0),
           'visible': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')}}
atomic_json(OUTPUT / 'preflight' / f'gpu-rank-{receipt["rank"]}.json', receipt)
print(json.dumps(receipt), flush=True)
