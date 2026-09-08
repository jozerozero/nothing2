"""Real-checkpoint smoke test; the two-pass runs are checks, not full evaluations."""
import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', choices=('old', 'loop'), required=True)
parser.add_argument('--passes', type=int, choices=(2, 3, 4), required=True)
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / f'source_{args.source}/src'))
os.environ['CROSS_TABLE_ARM'] = 'E4'
import numpy as np
import torch
from tabicl._model.tabicl import TabICL
from common import CHECKPOINT, OUTPUT, atomic_json
from loop_runtime import override

torch.set_num_threads(4)
torch.manual_seed(0)
if args.device == 'cuda':
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
assert checkpoint['config']['shared_depth_icl_enabled'] is True
assert checkpoint['config']['shared_depth_icl_dataset_conditioned'] is True
assert int(checkpoint['config'].get('shared_depth_icl_num_passes', 2)) == 2
model = TabICL(**checkpoint['config'])
model.load_state_dict(checkpoint['state_dict'], strict=True)
if args.source == 'loop':
    override(model, args.passes)
else:
    assert args.passes == 2
assert all(torch.equal(v, checkpoint['state_dict'][k]) for k, v in model.state_dict().items())
model.eval().to(args.device)
# Exercise the real trained ICL stack and its support-conditioned alpha directly.
# This bypasses unrelated column/row inference batching while retaining all ICL weights.
encoder = model.icl_predictor.tf_icl
width = encoder.blocks[0].norm1.normalized_shape[0]
generator = torch.Generator().manual_seed(19750)
x = torch.randn(1, 12, width, generator=generator).to(args.device)
context = torch.randn(1, 51, generator=generator).to(args.device)
counts = [0] * len(encoder.blocks)
def hook(index):
    def record(_module, _args, _output):
        counts[index] += 1
    return record
for index, block in enumerate(encoder.blocks):
    block.register_forward_hook(hook(index))
with torch.inference_mode():
    output = encoder(x, train_size=8, dataset_context=context)
assert counts == [args.passes] * len(counts), counts
assert output.dtype == torch.float32 and torch.isfinite(output).all()
subdirectory = 'preflight' if args.device == 'cuda' else 'cpu_preflight'
path = OUTPUT / subdirectory / f'{args.source}-loop{args.passes}.npz'
path.parent.mkdir(parents=True, exist_ok=True)
np.savez(path, output=output.cpu().numpy())
atomic_json(path.with_suffix('.json'), {'complete': True, 'source': args.source,
            'passes': args.passes, 'block_calls': counts, 'parameter_tensors_unchanged': True,
            'finite': True, 'dtype': str(output.dtype)})
print(json.dumps({'event': 'model_probe_pass', 'source': args.source, 'passes': args.passes,
                  'block_calls': counts}), flush=True)
