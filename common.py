"""Immutable contract for the authorized step19750 inference-only experiment."""
import json
import math
import os
from collections import Counter
from pathlib import Path

ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
NAME = 'g5sc19750_inference_loop34_gt8_20260908_v1'
STAGE = ROOT / 'stage' / NAME
OUTPUT = ROOT / 'evaluation' / NAME
BENCH = ROOT / 'evaluation/benchmark8_seven_suites_20260820_v1'
CHECKPOINT = ROOT / 'checkpoints/e4_g5_support_condition_alpha_20260905_v1/g36-g5scalpha-histe4-25k-v1/e4g5sc25r1-174368/step-19750.ckpt'
OLD_SOURCE = ROOT / 'stage/e4_g5_support_condition_alpha_base_20260905_v1/source'
LOOP_SOURCE = ROOT / 'stage/e4_g5_support_condition_alpha_loops_base_20260907_v1/source'
EVALUATOR = ROOT / 'stage/five_method_exact178_rw4096_step500_20260818_v1/talent_eval_online.py'
COUNTS = {'talent': 200, 'BCCO': 106, 'OpenML-CC18': 62, 'PFN': 29, 'TabArena': 33, 'TabZilla': 27}
METHODS = ('e4', 'tabiclv1', 'tabiclv2', 'limix2m', 'limix16m', 'tabpfn2', 'tabpfn25', 'tabpfn3')

def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)

def manifest_rows():
    data = json.loads((STAGE / 'benchmark_manifest.json').read_text())
    rows = data['rows']
    assert data['complete'] is True
    assert len(rows) == 457 and len({r['dataset'] for r in rows}) == 457
    assert dict(Counter(r['suite'] for r in rows)) == COUNTS
    return rows

def validate_panel(rows, expected):
    names = [r['dataset'].strip() for r in rows]
    assert len(names) == len(expected) and len(set(names)) == len(names)
    assert set(names) == set(expected)
    for row in rows:
        assert math.isfinite(float(row['accuracy'])) and 0 <= float(row['accuracy']) <= 1
        assert str(row.get('status', 'complete')).lower() not in {'error', 'skip', 'unsupported', 'failed'}

def checkpoint_identity():
    st = CHECKPOINT.stat()
    return {'path': str(CHECKPOINT), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}
