import json
import numpy as np
from common import OUTPUT, atomic_json

before = np.load(OUTPUT / 'preflight/old-loop2.npz')['output']
after = np.load(OUTPUT / 'preflight/loop-loop2.npz')['output']
assert np.array_equal(before, after), f'Loop2 changed: max_abs={np.max(np.abs(before-after))}'
for source, passes in [('old', 2), ('loop', 2), ('loop', 3), ('loop', 4)]:
    receipt = json.loads((OUTPUT / f'preflight/{source}-loop{passes}.json').read_text())
    assert receipt['complete'] and receipt['passes'] == passes and receipt['parameter_tensors_unchanged']
for rank in range(8):
    receipt = json.loads((OUTPUT / f'preflight/gpu-rank-{rank}.json').read_text())
    assert receipt['rank'] == rank and receipt['device_count'] == 1 and receipt['complete']
atomic_json(OUTPUT / 'preflight/contract_pass.json', {'complete': True, 'gpu_ranks': 8,
            'old_vs_new_loop2_bitwise_equal': True, 'parameter_changes': 0, 'realized_passes': [2, 3, 4]})
print('__LOOP34_PREFLIGHT_PASS__', flush=True)
