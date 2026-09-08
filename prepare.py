"""Prepare new, isolated snapshots; never modify training sources or checkpoints."""
import csv
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from common import *

assert Path(__file__).resolve().parent == STAGE.resolve()
assert not (OUTPUT / 'input_contract.json').exists(), 'already prepared; do not overwrite'
assert CHECKPOINT.is_file() and CHECKPOINT.stat().st_size > 100_000_000
assert time.time() - CHECKPOINT.stat().st_mtime > 60
OUTPUT.mkdir(parents=True, exist_ok=False)
for name, original in [('source_old', OLD_SOURCE), ('source_loop', LOOP_SOURCE)]:
    shutil.copytree(original / 'src', STAGE / name / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copy2(BENCH / 'benchmark_manifest.json', STAGE / 'benchmark_manifest.json')
rows = manifest_rows()
for row in rows:
    assert (BENCH / 'data' / row['dataset']).is_dir(), row['dataset']
shutil.copy2(BENCH / 'benchmark8_dataset_detail.tsv', STAGE / 'baseline_detail.tsv')
with (STAGE / 'baseline_detail.tsv').open() as f:
    baseline = list(csv.DictReader(f, delimiter='\t'))
assert {r['dataset'] for r in baseline} == {r['dataset'] for r in rows}
def finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False
baseline_missing = {key: sum(not finite(r.get(f'{key}_accuracy')) for r in baseline) for key in METHODS}
(STAGE / 'evaluator').mkdir()
# Keep the complete small evaluator directory to retain imports from sibling helpers.
for p in EVALUATOR.parent.iterdir():
    if p.is_file() and p.suffix == '.py':
        shutil.copy2(p, STAGE / 'evaluator' / p.name)
assert (STAGE / 'evaluator/talent_eval_online.py').is_file()
source_hashes = {}
for prefix in ('source_old/src', 'source_loop/src', 'evaluator'):
    for p in sorted((STAGE / prefix).rglob('*.py')):
        source_hashes[str(p.relative_to(STAGE))] = hashlib.sha256(p.read_bytes()).hexdigest()
contract = {'step': 19750, 'training_job': 174381, 'training_loops': 2, 'inference_loops': [3, 4],
            'checkpoint_identity': checkpoint_identity(), 'suite_counts': COUNTS,
            'dataset_count_per_loop': 457, 'evaluation_units': 914, 'world': 8,
            'qos': 'gtqos', 'nodes': 1, 'gpus': 8, 'cpus_per_task': 4,
            'n_estimators': 32, 'norm_methods': ['none', 'power'], 'kv_cache': False,
            'precision': 'FP32', 'amp': False, 'fa3': False,
            'baseline_missing': baseline_missing, 'source_hashes': source_hashes}
atomic_json(OUTPUT / 'input_contract.json', contract)
print(json.dumps({k: v for k, v in contract.items() if k != 'source_hashes'}), flush=True)
