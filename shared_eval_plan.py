"""Build, never launch, two source-pinned existing-allocation runtime plans.

Only the explicitly reviewed hold/supervisor/video-worker command families below
are accepted. UUID masks, not GPU ordinals or enumerated DRM FDs, establish each
existing worker's device. Scientific manifests, queues, rows and results are not
modified. Each target fails independently; an existing plan is never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import shared_eval_launch as launch
import tabfm_default_dispatch as q
from table6_restart_ag import publish

HERE = Path(__file__).resolve().parent
GIB = 1024**3
VIDEO = '/vast/users/guangyi.chen/causal_group/jinyuan.hu/eec-bench/EECBench/experiments/wan_generation_400_20260921'
SUPERVISOR_PY = '/vast/users/guangyi.chen/anaconda3/bin/python3'
VIDEO_PY = '/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python'
GAP_PY = '/vast/users/guangyi.chen/anaconda3/envs/tabicl/bin/python'
TARGETS = {
    'gap190': {'parent_job_id': '196093', 'node': 'auh7-1b-gpu-197',
               'gpu': {'uuid': 'bb018c40b923a974', 'pci': '0000:05:00.0'},
               'cpus': 16, 'startup_gib': 22, 'worker_cards': [1, 2, 3, 4, 5, 6, 7],
               'entry': 'table6_existing_gap190.py'},
    'swiftdual': {'parent_job_id': '204826', 'node': 'auh7-1b-gpu-284',
                  'gpu': {'uuid': '894e4c5615b2294e', 'pci': '0000:47:00.0'},
                  'cpus': 4, 'startup_gib': 20, 'worker_cards': [1, 3, 4, 5, 6, 7],
                  'entry': 'shared_foundation_sidecar.py'},
}
COMMON_SOURCES = ('shared_eval_plan.py', 'shared_eval_launch.py', 'shared_gpu_capacity_probe.py',
                  'table6_restart_deadline.py', 'table6_restart_ag.py', 'tabfm_default_dispatch.py',
                  'classification32_dispatch.py', 'classification32_campaign.py',
                  'classification32_submit.py', 'pfn_mitra_one.py', 'eval_one.py')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def identity(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), 'Missing or symlink input: '+str(path))
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'Input changed while hashing')
    return {'path': str(path.resolve()), 'size_bytes': after.st_size,
            'mtime_ns': after.st_mtime_ns, 'sha256': h.hexdigest()}


def read_pinned(path):
    record = identity(path)
    value = q.read(path)
    q.verify_file(record)
    return record, value


def merge_records(records):
    unique = {}
    for record in records:
        path = str(q.verify_file(record))
        require(not Path(record['path']).is_symlink(), 'Symlink source forbidden')
        if path in unique:
            require(unique[path]['sha256'] == record['sha256'], 'Conflicting source identities')
        else:
            unique[path] = identity(path)
    return [unique[path] for path in sorted(unique)]


def reviewed_foreign_service(process):
    """Public process/cgroup identities, not blanket foreign-UID approval.

    All eight GPUs are assigned to this parent. Known OS services and an
    administrator's login infrastructure are not GPU workers. Any foreign
    Python/model/unknown executable still blocks this lane.
    """
    if process['reason'] != 'foreign_fd_permission_denied':
        return False
    if process['uid'] == 0:
        return True  # Privileged FD uncertainty remains explicit in the audit.
    services = {
        ('systemd-network','systemd-network'): 'systemd-networkd.service',
        ('systemd-resolve','systemd-resolve'): 'systemd-resolved.service',
        ('systemd-timesync','systemd-timesyn'): 'systemd-timesyncd.service',
        ('messagebus','dbus-daemon'): 'dbus.service',
        ('syslog','rsyslogd'): 'rsyslog.service',
        ('_rpc','rpcbind'): 'rpcbind.service',
        ('statd','rpc.statd'): 'rpc-statd.service',
        ('munge','munged'): 'munge.service',
    }
    key = (process.get('username'),process.get('comm'))
    groups = process.get('cgroup',[])
    if key in services:
        return groups == ['0::/system.slice/'+services[key]]
    if key == ('nobody','node_exporter'):
        return len(groups)==1 and re.fullmatch(r'0::/system.slice/docker-[0-9a-f]{64}\.scope',groups[0]) is not None
    if process['uid']==1001 and process.get('username')=='ubuntu' and len(groups)==1:
        if process.get('comm') in ('systemd','(sd-pam)'):
            return groups == ['0::/user.slice/user-1001.slice/user@1001.service/init.scope']
        if process.get('comm') in ('sshd','bash'):
            return re.fullmatch(r'0::/user.slice/user-1001.slice/session-[0-9]+\.scope',groups[0]) is not None
    return False


def reviewed_processes(observation, target):
    """Reject unknown/future-owning commands; return stable identities for A/B."""
    parent, node = target['parent_job_id'], target['node']
    gpu_uuids = {g['uuid'] for g in observation['gpus']}
    require(len(gpu_uuids) == len(observation['gpus']) == 8, 'Expected eight distinct physical GPUs')
    require(observation['own_process_inspection_complete'] is True and not observation['process_errors'],
            'Incomplete same-UID process inspection')
    # Root-owned daemons are outside unprivileged /proc inspection. This is NOT
    # an all-UID ownership proof; retain the explicit limitation in the plan.
    unknown = observation['foreign_fd_inspection_unknown']
    require(all(reviewed_foreign_service(p) for p in unknown),
            'Unreviewed foreign non-root process could own the target GPU')
    processes = observation['owned_processes']
    require(len({p['pid'] for p in processes}) == len(processes), 'Duplicate process PID')
    require(sum(p['rss_bytes'] for p in processes) == observation['same_uid_rss_bytes'], 'Inconsistent RSS sum')
    workers, holds, supervisors, signatures = {}, {}, [], []
    probe_count = 0
    for p in processes:
        cmd, env = p['cmdline'], p['selected_environment']
        require(env.get('SLURM_JOB_ID') == parent and any('/job_'+parent+'/' in c for c in p['cgroup']),
                'Same-user process outside the reviewed parent: '+str(p['pid']))
        step = env.get('SLURM_STEP_ID', '')
        if not step and any('/job_'+parent+'/step_batch/' in c for c in p['cgroup']):
            step = 'batch'  # Actual batch shells do not export SLURM_STEP_ID.
        require(any('/step_'+step+'/' in c for c in p['cgroup']) and
                (step == 'batch' or step.isdigit()), 'Process step/cgroup binding differs')
        if p['is_probe']:
            require(p['pid'] == observation['probe_pid'] and len(cmd) >= 5 and
                    str(HERE/'shared_gpu_capacity_probe.py') in cmd and '--node' in cmd
                    and '--parent' in cmd and parent in cmd, 'Unrecognized probe process')
            require(not p['gpu_fds'], 'Read-only probe unexpectedly opened a GPU')
            probe_count += 1
            continue
        if cmd == ['/bin/sh', '/var/spool/slurmd/job'+parent+'/slurm_script']:
            kind = 'batch'
        elif cmd == ['bash', '-lc', 'echo ['+node+'] holding allocation across nodes: '+node+
                     ' ; trap : TERM INT; sleep infinity']:
            kind = 'hold'
        elif cmd == ['sleep', 'infinity']:
            kind = 'sleep'
        elif cmd == [SUPERVISOR_PY, VIDEO+'/node_supervisor.py', '--cards', '1,2,3,4,5,6,7']:
            require(parent == '196093', 'Unexpected supervisor on Swift target')
            supervisors.append(p)
            kind = 'supervisor'
        else:
            prefix = [VIDEO_PY, '-u', VIDEO+'/campaign.py', 'worker', '--worker']
            require(len(cmd) == 8 and cmd[:5] == prefix and cmd[6:] == ['--memory-mode', 'stream'],
                    'Unreviewed existing command: '+repr(cmd))
            match = re.fullmatch('node'+node.rsplit('-', 1)[-1]+r'_gpu([1-7])', cmd[5])
            require(match is not None, 'Unexpected video worker identity')
            card = int(match.group(1))
            require(card in target['worker_cards'] and card not in workers, 'Unexpected/duplicate video worker')
            mask = env.get('ROCR_VISIBLE_DEVICES', '')
            require(re.fullmatch(r'GPU-[0-9a-f]{16}', mask) and mask[4:] in gpu_uuids
                    and mask[4:] != target['gpu']['uuid'], 'Worker GPU mask includes/does not exclude target')
            # Legacy GPU_DEVICE_ORDINAL may still list all8. It is not used to
            # infer usage: the physical ROCr UUID and actual idle card are both checked.
            require(env.get('HIP_VISIBLE_DEVICES') in (None, '', '0') and
                    env.get('CUDA_VISIBLE_DEVICES') in (None, '', '0'), 'Unreviewed worker alias mask')
            workers[card] = p
            kind = 'worker'
        if kind in ('batch', 'hold', 'sleep'):
            require(step == 'batch' and not p['gpu_fds'] and kind not in holds,
                    'Hold process has unexpected GPU access/step/duplicate')
            holds[kind] = p
        elif kind == 'supervisor':
            require(not p['gpu_fds'], 'Supervisor unexpectedly opened a GPU')
        signatures.append({'pid': p['pid'], 'parent_pid': p['parent_pid'], 'start_ticks': p['start_ticks'],
                           'cmdline': cmd, 'environment': env, 'cgroup': p['cgroup'], 'kind': kind})
    require(probe_count == 1 and set(holds) == {'batch', 'hold', 'sleep'}, 'Missing unique probe/hold chain')
    require(holds['hold']['parent_pid'] == holds['batch']['pid'] and
            holds['sleep']['parent_pid'] == holds['hold']['pid'], 'Hold chain differs')
    require(set(workers) == set(target['worker_cards']), 'Reviewed worker roster changed')
    masks = [p['selected_environment']['ROCR_VISIBLE_DEVICES'] for p in workers.values()]
    require(len(set(masks)) == len(masks), 'Existing workers are not independently UUID-pinned')
    if parent == '196093':
        require(len(supervisors) == 1 and all(p['parent_pid'] == supervisors[0]['pid'] for p in workers.values()),
                'Supervisor/card exclusion cannot be bound to its seven actual workers')
    else:
        require(not supervisors, 'Unreviewed future-work supervisor')
    return {'stable_processes': sorted(signatures, key=lambda p: p['pid']),
            'worker_uuid_masks': sorted(masks), 'privileged_fd_inspection_unknown': unknown,
            'all_uid_ownership_proven': not unknown}


def proof(sample_paths, target, *, now=None):
    pinned = [read_pinned(path) for path in sample_paths]
    records, samples = zip(*pinned)
    require(len(samples) == 2 and 15 <= samples[1]['started_epoch']-samples[0]['epoch'] <= 900,
            'Need two separately invoked samples at least15s apart')
    require(0 <= (time.time() if now is None else now)-samples[1]['epoch'] <= 300,
            'Latest login-clock resource sample is stale')
    probe_sha = identity(HERE/'shared_gpu_capacity_probe.py')['sha256']
    reviews, observations, idle_cpus = [], [], []
    for sample in samples:
        require(sample['schema'] == 'shared_gpu_capacity_sample_v1' and sample['source_sha256'] == probe_sha,
                'Unknown/changed probe source')
        matches = [p for p in sample['parents'] if p['parent'] == target['parent_job_id']]
        require(len(matches) == 1, 'Missing/duplicate target parent')
        p = matches[0]
        require(p['complete'] is True and p['node'] == target['node'] and p['source_sha256'] == probe_sha,
                'Wrong/incomplete node resource proof')
        launch.check_parent(' '.join(k+'='+str(v) for k, v in p['job_fields'].items()), target)
        require(0 <= p['same_uid_rss_bytes'] <= target['startup_gib']*GIB, 'Insufficient parent RSS headroom')
        require(p['node_memory_available_bytes'] >= 40*GIB, 'Node available memory below40GiB')
        cards = [g for g in p['gpus'] if g['uuid'] == target['gpu']['uuid'] and g['pci'] == target['gpu']['pci']]
        require(len(cards) == 1 and cards[0]['hardware_idle'] is True and cards[0]['busy_percent'] == 0
                and 0 <= cards[0]['vram_used_bytes'] < 128*1024**2 and not cards[0]['foreign_fd_owner_pids'],
                'Selected physical GPU is occupied or foreign-owned')
        require(p['parent_cpu_proof_complete'] is True and p['idle_cpu_threshold_percent_exclusive'] == 25,
                'No verified parent CPU scope')
        cpu_samples, parent_cpus = p['cpu_percent_samples'], set(p['parent_batch_cpu_ids'])
        require(len(cpu_samples) == 2 and len(parent_cpus) >= target['cpus'], 'CPU proof insufficient')
        idle = {c for c in parent_cpus if all(0 <= c < len(row) and 0 <= row[c] < 25 for row in cpu_samples)}
        require(idle == set(p['idle_parent_cpu_ids_twice']), 'CPU idle evidence inconsistent')
        idle_cpus.append(idle)
        reviews.append(reviewed_processes(p, target))
        observations.append({'node_epoch': p['epoch'], 'rss_bytes': p['same_uid_rss_bytes'],
                             'node_available_bytes': p['node_memory_available_bytes']})
    require(reviews[0]['stable_processes'] == reviews[1]['stable_processes'], 'Existing process ownership changed between samples')
    intersection = sorted(idle_cpus[0] & idle_cpus[1])
    require(len(intersection) >= target['cpus'], 'Insufficient CPUs idle in both observations')
    rationale = ('Reviewed exact unchanged batch hold chain and video commands; '
                 + ('supervisor --cards1..7 is bound to seven actual UUID-pinned children, ' if target['parent_job_id']=='196093'
                    else 'six existing video workers each have an explicit non-target ROCr UUID, ')
                 + 'all worker UUIDs exclude the selected physical PCI/UUID in both complete same-UID observations. '
                 'Driver enumeration FDs and legacy ordinal aliases are not treated as device usage. '
                 'Unreadable privileged and specifically reviewed OS/login-service FDs remain explicitly uncertain; '
                 'no foreign GPU/model application was observed. The eight-GPU owned Slurm allocation, '
                 'no observed foreign target FD, physical idleness and node recheck are the authorization basis.')
    return {'gpu_idle_verified': True, 'resources_available_verified': True, 'sample_records': list(records),
            'controller_finished_epoch': samples[1]['epoch'], 'idle_parent_cpu_ids_both_samples': intersection,
            'observations': observations}, {'approved': True, 'rationale': rationale, 'reviews': reviews,
                                            'all_uid_ownership_proven': all(r['all_uid_ownership_proven'] for r in reviews)}


def build(kind, sample_paths, attempt, scientific_paths):
    require(kind in TARGETS and re.fullmatch(r'[A-Za-z0-9_-]+', attempt), 'Unsafe attempt/target')
    target = TARGETS[kind]
    evidence, review = proof(sample_paths, target)
    records = [identity(HERE/name) for name in COMMON_SOURCES]
    plan = {k: target[k] for k in ('parent_job_id', 'node', 'gpu', 'cpus')}
    plan.update(schema='shared_eval_existing_plan_v1', sidecar_id=attempt+'-'+kind, short_name='shared-'+kind,
                max_step_seconds=7200, mem_gib=40, parent_mem_gib=64,
                entry_script=str(HERE/target['entry']), proof=evidence, reviewed_ownership=review,
                parent_allocation_unchanged=True, inherited_gpu_access=8, actual_model_gpus=1,
                scientific_configuration_unchanged=True)
    if kind == 'gap190':
        import table6_existing_gap190 as entry
        import table6_missing190_worker as worker
        import table6_missing190_fit as fit
        rec, scientific = read_pinned(scientific_paths[0])
        fit.validate_plan(scientific)
        require(scientific['plan_id'] == entry.SCIENTIFIC_PLAN_ID, 'Wrong canonical missing190 plan')
        plan.update(scientific_plan=rec, python=GAP_PY)
        records += [rec] + [identity(HERE/name) for name in entry.REQUIRED_SOURCES]
        expected = {worker.SHORT: worker.PINNED['short_worker.py'], **{worker.R19/name: sha for name, sha in fit.PINNED_R19.items()},
                    worker.R19/'worker.py': worker.PINNED['worker.py']}
        for path, sha in expected.items():
            rec = identity(path)
            require(rec['sha256'] == sha, 'Frozen original HPO dependency changed: '+str(path))
            records.append(rec)
    else:
        import shared_foundation_sidecar as entry
        require(len(scientific_paths) == 2, 'Both original Swift manifests required')
        campaigns, overlays = [], {}
        import tabfm_regression_shapes as shapes
        for path in scientific_paths:
            rec, raw = read_pinned(path)
            man, tasks = q.load_campaign(path)
            require(raw == man, 'Scientific manifest changed while validating')
            campaigns.append(man)
            overlays[man['manifest_id']] = shapes.build(man, tasks)
            shapes.validate(overlays[man['manifest_id']], man, tasks)
            records += [rec, man['worker_script'], *man['worker_sources'],
                        man['classification_manifest'], man['regression_manifest']]
        require([m['protocol']['variant'] for m in campaigns] == ['official16', 'budget32x8'], 'Wrong Swift variant order')
        require(len({m['worker_python'] for m in campaigns}) == 1 and
                len({m['output_root'] for m in campaigns}) == 2, 'Swift runtime/output collision')
        records += [identity(HERE/name) for name in ('shared_foundation_sidecar.py', 'tabfm_local_tmp.py',
                                                    'tabswift_dispatch.py', 'tabfm_regression_shapes.py')]
        plan.update(python=campaigns[0]['worker_python'], campaign_paths=[str(Path(p).resolve()) for p in scientific_paths],
                    resource=entry.RESOURCE, eligibility=entry.ELIGIBILITY,
                    regression_shape_overlays=overlays,
                    coverage_note='Original22175 receipt dimensions are used only for small-task eligibility; '
                                  'missing/invalid dimensions are audited and skipped. Task rows, full splits and scientific manifests unchanged.')
    require(Path(plan['python']).is_absolute() and Path(plan['python']).is_file(), 'Missing target Python')
    plan['source_records'] = merge_records(records)
    plan['plan_id'] = q.digest(plan)
    return plan


def build_both(sample_paths, attempt, gap_scientific_plan, swift_official, swift_budget):
    outcomes = {}
    for kind, paths in [('gap190', [gap_scientific_plan]), ('swiftdual', [swift_official, swift_budget])]:
        try:
            plan = build(kind, sample_paths, attempt, paths)
            path = launch.STAGE/plan['sidecar_id']/'plan.json'
            require(not path.parent.is_symlink() and not path.is_symlink(), 'Unsafe plan output')
            publish(path, plan)
            outcomes[kind] = {'state': 'plan_created_not_launched', 'path': str(path), 'plan_id': plan['plan_id']}
        except Exception as exc:
            outcomes[kind] = {'state': 'not_created', 'error': type(exc).__name__+': '+str(exc)}
    return outcomes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-a', required=True); p.add_argument('--sample-b', required=True)
    p.add_argument('--attempt', required=True)
    p.add_argument('--gap-scientific-plan', default=str(launch.ROOT/'evaluation/table6_missing190_standard_hpo_20260922_v1/plan.json'))
    p.add_argument('--swift-official', default=str(launch.ROOT/'evaluation/tabswift_official16_standard681_20260922_v1/manifest.json'))
    p.add_argument('--swift-budget', default=str(launch.ROOT/'evaluation/tabswift_budget32x8_standard681_20260922_v1/manifest.json'))
    a = p.parse_args()
    result = build_both([a.sample_a, a.sample_b], a.attempt, a.gap_scientific_plan, a.swift_official, a.swift_budget)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return int(any(row['state'] != 'plan_created_not_launched' for row in result.values()))


if __name__ == '__main__':
    raise SystemExit(main())
