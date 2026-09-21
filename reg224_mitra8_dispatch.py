#!/usr/bin/env python3
"""Launch only actual-eight Mitra on frozen all224, using eight audited idle GPUs.

Eight deterministic, disjoint size-sorted round-robin shards contain 28 rows
each. Default layout is four existing parents times two GPUs. No allocations,
parent workloads, historical results, or non-Mitra models are changed.
"""
import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import pfn28_dispatch as common


ELIGIBLE_PARENTS = {'196092', '196093', '200798', '200797', '204828', '204827',
                    '204826', '206117', '206116', '194259', '194181', '194180'}
DEFAULT_PARENTS = ('196093', '200798', '204828', '194259')
ROOT = common.FT / 'reg224_mitra8_20260921_v1'
SCRIPTS = ('reg224_mitra8_dispatch.py', 'pfn28_dispatch.py', 'eval_dispatch.py',
           'pfn_mitra8_one.py', 'pfn_mitra_one.py', 'eval_one.py')
MITRA_WEIGHT_SHA = 'd8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def parent_fields(parent):
    require(parent in ELIGIBLE_PARENTS, 'Parent not in explicit user-owned eligible inventory')
    raw = subprocess.check_output(['scontrol', 'show', 'job', parent, '-o'], text=True)
    fields = dict(re.findall(r'(\w+)=(\S+)', raw))
    require(fields['JobState'] == 'RUNNING' and fields['NumNodes'] == '1'
            and fields['UserId'].startswith('guangyi.chen('), 'Parent ownership/state changed')
    require(re.search(r'(?:^|,)gres/gpu=8(?:,|$)', fields['AllocTRES']), 'Parent must own eight GPUs')
    memory = re.search(r'(?:^|,)mem=([0-9.]+)([KMGTP])(?:,|$)', fields['AllocTRES'])
    require(memory is not None, 'Parent memory reservation absent')
    memory_bytes = float(memory[1]) * 1024 ** ('KMGTP'.index(memory[2]) + 1)
    require(memory_bytes >= 64 * common.GIB, 'Parent reserves less than64GiB')
    return fields


def configure_common():
    common.ROOT = ROOT
    common.ASSIGNMENTS = {parent: ['mitra'] for parent in ELIGIBLE_PARENTS}
    common.checked_fields = parent_fields


def shard_indices(rows, parents, workers_per_parent):
    require(len(parents) == len(set(parents)) and set(parents) <= ELIGIBLE_PARENTS, 'Invalid/duplicate parent selection')
    require(workers_per_parent in (1, 2) and len(parents) * workers_per_parent == 8,
            'Exactly eight workers are required, at most two per parent')
    require(len(rows) == 224 and {r['dataset_index'] for r in rows} == set(range(224)), 'Expected exact all224 indices')
    ordered = [r['dataset_index'] for r in sorted(rows,
        key=lambda r: (sum(f['size_bytes'] for f in r['input_files']), r['dataset_index']))]
    shards = []
    for rank in range(8):
        shards.append({'parent': parents[rank // workers_per_parent], 'shard_rank': rank,
            'lane_id': f'mitra-s{rank}', 'dataset_indices': ordered[rank::8], 'target_count': 28})
    flat = [i for shard in shards for i in shard['dataset_indices']]
    require(len(flat) == len(set(flat)) == 224 and set(flat) == set(range(224)), 'Shard coverage/overlap failure')
    return ordered, shards


def build_plan(plan_path, parents, workers_per_parent, only_parent=None):
    require(common.digest(common.MITRA / 'weights_manifest.json') == common.MITRA_SHA, 'Mitra weight inventory changed')
    manifest = common.read(common.MANIFEST)
    require(manifest['source_step'] == 22175 and manifest['membership_count'] == 224, 'Wrong frozen source membership')
    require(manifest['manifest_id'] == hashlib.sha256(json.dumps({k:v for k,v in manifest.items()
        if k != 'manifest_id'}, sort_keys=True, allow_nan=False).encode()).hexdigest(), 'Manifest hash mismatch')
    counts = {suite: sum(r['suite'] == suite for r in manifest['rows'])
              for suite in ('talent', 'BCCO', 'CTR23', 'TabArena', 'PFN')}
    require(counts == {'talent': 100, 'BCCO': 50, 'CTR23': 33, 'TabArena': 13, 'PFN': 28}, 'all224 suite inventory changed')
    ordered, shards = shard_indices(manifest['rows'], parents, workers_per_parent)
    require(only_parent is None or only_parent in parents, '--parent must belong to --parents')
    first, second = common.read(ROOT / 'capacity_a.json'), common.read(ROOT / 'capacity_b.json')
    require(second['epoch'] - first['epoch'] >= 10 and 0 <= time.time() - second['epoch'] < 1800,
            'Fresh, separate capacity observations are required')
    before = {p['parent']: p for p in first['parents']}
    after = {p['parent']: p for p in second['parents']}
    nodes, used_physical, used_nodes = [], set(), set()
    for parent in parents:
        if only_parent and parent != only_parent:
            continue
        a, b = before[parent], after[parent]
        require(not any(p.get('error') or p.get('excluded') for p in (a, b)), 'Capacity probe excluded selected parent')
        fields = parent_fields(parent)
        require(fields['NodeList'] == a['node'] == b['node'] and b['node'] not in used_nodes,
                'Parent node changed or multiple selected allocations share one node')
        used_nodes.add(b['node'])
        require(sum(p['rss'] for p in b['owned_processes']) < 8 * common.GIB
                and b['available_ram'] > 56 * common.GIB, 'Selected node is not safely idle in host RAM')
        require(int(fields['NumCPUs']) >= 4 * workers_per_parent, 'Parent CPU reservation too small')
        previous = {g['uuid']: g for g in a['gpus']}
        idle = sorted((g for g in b['gpus'] if common.check_idle(g)
            and common.check_idle(previous.get(g['uuid'], {})) and g['pci'] == previous[g['uuid']]['pci']),
            key=lambda g: (g['pci'], g['uuid']))
        require(len(idle) >= workers_per_parent, 'Insufficient twice-audited idle physical GPUs')
        selected_shards = [s for s in shards if s['parent'] == parent]
        lanes = []
        for shard, physical in zip(selected_shards, idle):
            key = (b['node'], physical['uuid'])
            require(key not in used_physical, 'Duplicate physical GPU assignment')
            used_physical.add(key)
            python, env = common.runtime('mitra')
            lanes.append({**shard, 'model': 'mitra', 'gpu': physical, 'python': python, 'env': env})
        nodes.append({'parent': parent, 'node': b['node'], 'lanes': lanes,
                      'cpus': 4 * len(lanes), 'mem_gib': 24 * len(lanes)})
    assigned = [i for node in nodes for lane in node['lanes'] for i in lane['dataset_indices']]
    require(len(assigned) == len(set(assigned)), 'Selected plan shards overlap')
    if not only_parent:
        require(set(assigned) == set(range(224)), 'Full plan does not cover all224')
    plan = {'created_epoch': time.time(), 'manifest': str(common.MANIFEST),
        'manifest_id': manifest['manifest_id'], 'manifest_sha256': common.digest(common.MANIFEST),
        'mitra_manifest_sha256': common.MITRA_SHA, 'model_weight_sha256': {'mitra': MITRA_WEIGHT_SHA},
        'models': ['mitra'], 'strict_actual8': True, 'worker_scripts': {'mitra': 'pfn_mitra8_one.py'},
        'dispatcher_script': 'reg224_mitra8_dispatch.py', 'job_name': 'mitra8reg',
        'blocked_step_names': ['mitra8reg', 'regft50eval'],
        'coexistence_policy': 'Only freshly revalidated idle physical GPUs and<8GiB owned RSS; never displace unrelated workloads',
        'gpu_lock_dir': str(common.FT / 'sidecar_gpu_locks'),
        'source_checkpoint_step': 22175, 'dataset_indices': ordered, 'nodes': nodes,
        'assignment': 'size-sorted all224; disjoint round-robin across eight named lanes',
        'campaign_shards': shards, 'campaign_target_results': 224, 'target_results': len(assigned),
        'campaign_gpu_workers': 8, 'actual_gpu_workers': sum(len(n['lanes']) for n in nodes),
        'workers_per_parent': workers_per_parent, 'requested_parents': list(parents),
        'row_rss_limit_gib': 20, 'single_dataset_limit_seconds': 1800, 'hard_step_limit': '02:00:00',
        'continue_errors': True, 'failed_dataset_policy': 'one bounded attempt; preserve error/log; continue assigned shard unless safety guard stops',
        'parent_allocations_unchanged': True, 'new_allocations': False,
        'worker_sha256': {name: common.digest(common.REPO / name) for name in SCRIPTS}}
    plan['plan_id'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    common.atomic(plan_path, plan, True)
    print(json.dumps(plan), flush=True)


def self_test():
    rows = [{'dataset_index': i, 'input_files': [{'size_bytes': (i * 137) % 401}]} for i in range(224)]
    order, shards = shard_indices(rows, DEFAULT_PARENTS, 2)
    require(all(len(s['dataset_indices']) == 28 for s in shards), 'Incorrect shard sizes')
    require([i for rank in range(28) for i in [s['dataset_indices'][rank] for s in shards]] == order,
            'Round-robin sequence mismatch')
    _, repeat = shard_indices(list(reversed(rows)), DEFAULT_PARENTS, 2)
    require(shards == repeat, 'Assignment depends on incoming row order')
    _, singles = shard_indices(rows, tuple(sorted(ELIGIBLE_PARENTS))[:8], 1)
    require(len({s['parent'] for s in singles}) == 8, 'Single-worker parent layout failed')
    print(json.dumps({'self_test': 'pass', 'memberships': 224, 'workers': 8, 'rows_per_worker': 28,
                      'overlap': 0, 'deterministic': True, 'layouts_tested': 2}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'launch', 'node', 'self-test'])
    parser.add_argument('--parents', default=','.join(DEFAULT_PARENTS))
    parser.add_argument('--workers-per-parent', type=int, default=2)
    parser.add_argument('--parent')
    parser.add_argument('--attempt', default='v1')
    parser.add_argument('--plan', type=Path, default=ROOT / 'plan.json')
    args = parser.parse_args()
    if args.mode == 'self-test':
        self_test(); return
    configure_common()
    parents = tuple(p.strip() for p in args.parents.split(',') if p.strip())
    require(re.fullmatch(r'[a-zA-Z0-9_-]+', args.attempt), 'Invalid attempt name')
    require(args.plan.resolve().parent == ROOT.resolve(), 'Plan must remain inside the new campaign root')
    require(args.parent is None or args.parent in ELIGIBLE_PARENTS, 'Invalid --parent')
    if args.mode == 'plan':
        build_plan(args.plan, parents, args.workers_per_parent, args.parent)
    elif args.mode == 'node':
        require(args.parent is not None, 'node requires --parent')
        common.node_run(args.parent, args.attempt, args.plan)
    else:
        common.launch(args.attempt, args.plan, args.parent)


if __name__ == '__main__':
    main()
