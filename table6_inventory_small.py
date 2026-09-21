"""Read-only, bounded audit of R19 CPU tails and recorded failures."""
import collections
import json
from pathlib import Path
import sys

ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
BASE = ROOT/'stage/table6_remaining19_standard_hpo_bg8_20260912_v1'
OUT = ROOT/'evaluation/table6_remaining19_standard_hpo_bg8_20260912_v1'

def main():
    sys.path.insert(0, str(BASE))
    from common import load_plan, CPU_METHODS
    from worker import validate_complete
    plan = load_plan()
    assert plan['plan_id'] == '47c3d448249235ad4f7ae6248424a1f8f38094ae204d2d4ece472b4effb61aff'
    counts, missing = {}, []
    for method in sorted(CPU_METHODS):
        count = collections.Counter()
        for pair in plan['pairs']:
            if pair['method'] != method:
                continue
            count['target'] += 1
            path = OUT/'results'/method/(pair['key']+'.json')
            if path.exists():
                value = json.loads(path.read_text())
                validate_complete(value, pair, plan)
                count['complete'] += 1
            else:
                count['missing'] += 1
                missing.append(pair)
        counts[method] = dict(count)
    errors = []
    for path in sorted((OUT/'errors').glob('*.json')):
        value = json.loads(path.read_text())
        errors.append({'path':str(path), 'record':value})
    return {'cpu_counts':counts, 'cpu_missing_pairs':missing, 'errors':errors,
            'policy':'read-only; native aggregate/selection validation; no fit or seed-tree traversal'}

if __name__ == '__main__':
    print(json.dumps(main(), allow_nan=False))
