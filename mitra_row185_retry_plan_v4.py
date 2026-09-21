"""Create a one-row retry plan without modifying the frozen Mitra8 campaign."""
import hashlib
import json
import time

import reg224_mitra8_dispatch as d


def main():
    d.configure_common()
    c = d.common
    parent, attempt = '204827', 'row185_retry4'
    a = c.read(d.ROOT / 'capacity_row185_retry4_a.json')
    b = c.read(d.ROOT / 'capacity_row185_retry4_b.json')
    assert b['epoch'] - a['epoch'] >= 10 and 0 <= time.time() - b['epoch'] < 600
    first, last = a['parents'][0], b['parents'][0]
    assert first['parent'] == last['parent'] == parent
    assert not any(x.get('error') or x.get('excluded') for x in (first, last))
    assert first['node'] == last['node'] == d.parent_fields(parent)['NodeList']
    assert sum(p['rss'] for p in last['owned_processes']) < 8 * c.GIB
    assert last['available_ram'] > 56 * c.GIB
    previous = {g['uuid']: g for g in first['gpus']}
    idle = sorted((g for g in last['gpus'] if c.check_idle(g)
        and c.check_idle(previous.get(g['uuid'], {}))
        and previous[g['uuid']]['pci'] == g['pci']), key=lambda g: g['pci'])
    assert idle, 'No twice-audited idle GPU'
    plan = c.load_plan(d.ROOT / 'plan_v2.json')
    manifest = c.read(c.MANIFEST)
    row = manifest['rows'][185]
    assert row['suite'] == 'TabArena' and row['dataset'] == 'TabArena__QSAR-TID-11', row
    assert not (d.ROOT / 'results/mitra/row-185.json').exists()
    assert not (d.ROOT / 'claims/mitra/row-185.json').exists()
    preserved = {}
    for index in range(224):
        if index == 185:
            continue
        path = d.ROOT / 'results/mitra' / f'row-{index:03d}.json'
        c.validate_result(path, 'mitra', index, manifest, plan)
        preserved[str(path)] = c.digest(path)
    py, env = c.runtime('mitra')
    env['MITRA_RETRY_TAB2D_SHA256'] = 'a3f4b9ef6fa72ab870c66105683493bd32c4ecaced3af42601146b73b0887826'
    lane = {'parent': parent, 'lane_id': 'mitra-row185', 'model': 'mitra',
        'dataset_indices': [185], 'target_count': 1, 'gpu': idle[0], 'python': py, 'env': env}
    script = 'pfn_mitra8_row185_retry_v4.py'
    plan.pop('plan_id')
    plan.update(created_epoch=time.time(), dataset_indices=[185], target_results=1,
        nodes=[{'parent': parent, 'node': last['node'], 'lanes': [lane], 'cpus': 4, 'mem_gib': 24}],
        actual_gpu_workers=1, workers_per_parent=1, requested_parents=[parent],
        worker_scripts={'mitra': script}, job_name='mitra8retry',
        assignment='Only failed TabArena QSAR-TID-11 row185; preserve the223 completed results',
        retry_attempt=attempt, retry_of='v2', prior_result_count=223,
        prior_result_sha256=preserved,
        capacity_evidence={'first': str(d.ROOT / 'capacity_row185_retry4_a.json'),
                           'second': str(d.ROOT / 'capacity_row185_retry4_b.json')},
        implementation_exception='Native layer memory-lifetime optimization and tokenwise feedforward streaming; full attention contexts, native1024query chunks and RNG unchanged')
    plan['worker_sha256'][script] = c.digest(c.REPO / script)
    plan['worker_sha256']['mitra_row185_retry_plan_v4.py'] = c.digest(c.REPO / 'mitra_row185_retry_plan_v4.py')
    plan['plan_id'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    path = d.ROOT / 'plan_row185_retry4.json'
    c.atomic(path, plan, True)
    print(json.dumps({'plan': str(path), 'plan_id': plan['plan_id'], 'parent': parent,
        'node': last['node'], 'gpu': idle[0], 'preserved_results': len(preserved), 'target': 1}), flush=True)


if __name__ == '__main__':
    main()
