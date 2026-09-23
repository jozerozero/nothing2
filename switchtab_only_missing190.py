#!/usr/bin/env python3
"""Use an unchanged missing190 Runtime with SwitchTab-only queue admission.

Run only inside a verified GPU allocation after native preflight and gate for
that allocation. This adapter never edits the frozen plan or legacy sources.
"""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--plan', type=Path, required=True)
args = parser.parse_args()
sys.dont_write_bytecode = True
sys.path.insert(0, str(args.repo.resolve(strict=True)))
from table6_missing190_worker import Runtime

runtime = Runtime(args.plan)
expected_plan = 'f35686cd72710801e14ccf00e1d44f108d87627ee404927a178efc9c32c86066'
if runtime.plan['plan_id'] != expected_plan:
    raise ValueError('unexpected frozen missing190 plan')
original_sources = dict(runtime.sources)
original_admission = runtime.short.GPU_METHODS
pairs = [p for p in runtime.plan['pairs'] if p['method'] == 'SwitchTab']
if len(pairs) != 21 or any(p['task_kind'] != 'classification' for p in pairs):
    raise ValueError('SwitchTab scientific membership differs from 21 classification gaps')
print(json.dumps({'adapter': 'switchtab_only_missing190_v1',
                  'plan_id': expected_plan,
                  'admitted_method': 'SwitchTab',
                  'pair_keys': [p['key'] for p in pairs],
                  'runtime_sources': original_sources,
                  'resume_policy': 'native locks; completed/error/deferred skipped; native HPO100/seed15 retained'},
                 sort_keys=True), flush=True)
runtime.activate_budget()
runtime.short.GPU_METHODS = ('SwitchTab',)
try:
    summary = runtime.worker('gpu')
    if runtime.sources != original_sources:
        raise RuntimeError('frozen runtime source identity changed')
    print(json.dumps({'adapter': 'switchtab_only_missing190_v1', 'summary': summary},
                     sort_keys=True), flush=True)
except runtime.short.BudgetStop as exc:
    runtime.short.lifecycle('paused', phase='SwitchTab_only_worker', reason=str(exc), actual_execution_mode='gpu')
    print(json.dumps({'state': 'paused', 'reason': str(exc)}), flush=True)
finally:
    runtime.short.GPU_METHODS = original_admission
