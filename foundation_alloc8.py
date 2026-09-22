"""New eight-real-rank gates for unchanged TabFM/dual-TabSwift scientific workers.

Full-eight bootstrap and physical UUID binding replace the old four-rank
visibility path. Original scientific manifests/results/CAS claims are reused;
only independent runtime audits and isolated attempts are new. No submission,
automatic retry, claim release or completed-result overwrite is implemented.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import shlex
import socket
import subprocess
import time
import uuid

import allocated_gpu_uuid as binding
import tabfm_default_dispatch as q
import shared_foundation_full_v2 as full
from table6_restart_deadline import EnvironmentBudget

GIB = 1024**3
ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
STAGE = ROOT/'stage/foundation_alloc8_20260922_v1'
RESOURCE = {'gpus': 8, 'allocated_cpus': 64, 'rank_cpus': 4, 'allocated_mem_gib': 256,
            'rank_rss_gib': 28, 'allocation_rss_gib': 224, 'min_available_gib': 8,
            'allocation_cgroup_gib': 240, 'seconds': 259200}
REQUIRED = ('foundation_alloc8.py', 'allocated_gpu_uuid.py', 'shared_foundation_full_v2.py',
            'shared_foundation_sidecar.py', 'tabfm_default_dispatch.py', 'tabfm_local_tmp.py',
            'table6_restart_deadline.py', 'table6_restart_ag.py', 'pfn_mitra_one.py', 'eval_one.py',
            'classification32_dispatch.py', 'classification32_campaign.py', 'classification32_submit.py')


def load_plan(path):
    path = Path(path)
    q.require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'Absolute immutable runtime plan required')
    plan = q.read(path)
    q.require(plan['plan_id'] == q.digest({k:v for k,v in plan.items() if k != 'plan_id'}), 'Runtime plan digest changed')
    q.require(plan['resource'] == RESOURCE, 'Eight-rank resource contract changed')
    records = plan['source_records']
    sources = [q.verify_file(rec) for rec in records]
    q.require(len(sources) == len(set(sources)) and
              {Path(__file__).with_name(name).resolve() for name in REQUIRED} <= set(sources), 'Missing/duplicate runtime source pins')
    q.require(q.verify_file(plan['runtime_script']) == Path(__file__).resolve(), 'Wrong runtime entry identity')
    campaigns = []
    for record in plan['campaign_manifests']:
        campaign_path = q.verify_file(record)
        q.require(campaign_path in sources, 'Scientific manifest must be runtime pinned')
        man, tasks = q.load_campaign(campaign_path)
        campaigns.append((campaign_path, man, tasks))
    if plan['family'] == 'tabfm':
        q.require(len(campaigns) == 1 and campaigns[0][1]['name'] == 'tabfm_defaults_standard681_20260922_v1', 'Wrong TabFM campaign')
    else:
        q.require(plan['family'] == 'tabswift' and
                  [m.get('protocol', {}).get('variant') for _,m,_ in campaigns] == ['official16', 'budget32x8'], 'Wrong Swift protocols')
        q.require(Path(__file__).with_name('tabswift_dispatch.py').resolve() in sources, 'Strict ensemble validator unpinned')
        import tabswift_dispatch
        q.require(q.valid_result is tabswift_dispatch.validated_result, 'Original strict budget validator not installed')
    script = Path(__file__).with_name('tabfm_alloc8_slurm.sh' if plan['family'] == 'tabfm' else 'tabswift_alloc8_slurm.sh').resolve()
    q.require(q.verify_file(plan['slurm_script']) == script and script in sources, 'Slurm wrapper identity differs')
    q.require(q.verify_file(plan['generated_slurm_script']) == path.resolve().parent/'run.sh',
              'Generated Slurm script changed or points outside this runtime plan')
    q.require(len({m['output_root'] for _,m,_ in campaigns}) == len(campaigns) and
              {m['worker_python'] for _,m,_ in campaigns} == {plan['worker_python']}, 'Campaign runtime/output mismatch')
    root = q.checked_path(campaigns[0][1], Path(plan['runtime_root']))
    q.require(root.is_absolute() and root.name == plan['run_id'] and root.parent.name == 'allocated8', 'Unsafe runtime audit root')
    return plan, campaigns


def prepare(family, run_id, campaign_paths):
    """Create a new immutable runtime plan only; no scheduler calls or launch."""
    import re
    from table6_restart_ag import publish
    def identity(path):
        path = Path(path).absolute()
        q.require(path.is_file() and not path.is_symlink(), 'Missing/unsafe source file')
        before = path.stat(); raw = path.read_bytes(); after = path.stat()
        q.require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                  (after.st_ino, after.st_size, after.st_mtime_ns), 'Source changed while hashing')
        return {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                'size_bytes': after.st_size, 'mtime_ns': after.st_mtime_ns}

    def merge_records(records):
        unique = {}
        for record in records:
            actual = q.verify_file(record)
            if str(actual) in unique:
                q.require(unique[str(actual)]['sha256'] == record['sha256'], 'Conflicting source identities')
            else:
                unique[str(actual)] = identity(actual)
        return [unique[key] for key in sorted(unique)]
    q.require(family in ('tabfm', 'tabswift') and re.fullmatch(r'[A-Za-z0-9_-]+', run_id), 'Unsafe family/run ID')
    if not campaign_paths:
        names = ['tabfm_defaults_standard681_20260922_v1'] if family == 'tabfm' else [
            'tabswift_official16_standard681_20260922_v1', 'tabswift_budget32x8_standard681_20260922_v1']
        campaign_paths = [ROOT/'evaluation'/name/'manifest.json' for name in names]
    campaigns = [(Path(path).resolve(strict=True), *q.load_campaign(path)) for path in campaign_paths]
    script = Path(__file__).with_name('tabfm_alloc8_slurm.sh' if family == 'tabfm' else 'tabswift_alloc8_slurm.sh')
    sources = [identity(Path(__file__).with_name(name)) for name in REQUIRED]
    sources += [identity(script)]
    manifest_records = []
    for path, man, _tasks in campaigns:
        rec = identity(path); manifest_records.append(rec)
        sources += [rec, man['worker_script'], *man['worker_sources'],
                    man['classification_manifest'], man['regression_manifest']]
    if family == 'tabswift':
        sources.append(identity(Path(__file__).with_name('tabswift_dispatch.py')))
    destination = STAGE/run_id/'plan.json'
    generated = render_run_script(script.read_text(), destination)
    plan = {'schema': 'foundation_alloc8_runtime_v1', 'family': family, 'run_id': run_id,
            'resource': RESOURCE, 'worker_python': campaigns[0][1]['worker_python'],
            'runtime_script': identity(__file__), 'slurm_script': identity(script),
            'generated_slurm_script': {'path': str(destination.parent/'run.sh'),
                                       'sha256': hashlib.sha256(generated.encode()).hexdigest()},
            'job_name': 'fm681g8' if family == 'tabfm' else 'swift2g8',
            'runtime_root': str(Path(campaigns[0][1]['output_root'])/'allocated8'/run_id),
            'campaign_manifests': manifest_records, 'source_records': merge_records(sources),
            'science_unchanged': True, 'original_full_memberships': True,
            'resource_deferrals_retain_claims': True, 'automatic_retry': False}
    plan['plan_id'] = q.digest(plan)
    # Validate first, without publishing a potentially unusable plan. The body
    # contract is shared with load_plan, which also validates the final file.
    q.require((family == 'tabfm' and len(campaigns) == 1 and campaigns[0][1]['name'] == 'tabfm_defaults_standard681_20260922_v1')
              or (family == 'tabswift' and [m.get('protocol', {}).get('variant') for _,m,_ in campaigns] == ['official16', 'budget32x8']),
              'Wrong scientific campaign roster')
    q.require(len({m['worker_python'] for _,m,_ in campaigns}) == 1 and
              len({m['output_root'] for _,m,_ in campaigns}) == len(campaigns), 'Campaign runtime/output differs')
    publish_text(destination.parent/'run.sh', generated)
    publish(destination, plan)
    load_plan(destination)
    return {'state': 'prepared_not_submitted', 'plan_path': str(destination), 'plan_id': plan['plan_id'],
            'job_name': plan['job_name'], 'slurm_script': str(destination.parent/'run.sh')}


def render_run_script(source, plan_path):
    plan_path = Path(plan_path)
    directory = plan_path.parent
    lines = source.splitlines()
    last = max(i for i, line in enumerate(lines) if line.startswith('#SBATCH '))
    lines[last+1:last+1] = ['#SBATCH --chdir='+str(directory),
                           '#SBATCH --output='+str(directory/'slurm-%j.out'),
                           '#SBATCH --error='+str(directory/'slurm-%j.err')]
    text = '\n'.join(lines)+'\n'
    original = '[[ $# == 1 && "$1" = /* && -f "$1" ]] || { echo \'Expected absolute immutable runtime plan\' >&2; exit 2; }\nALLOC8_PLAN="$1"'
    q.require(text.count(original) == 1, 'Original Slurm parameter binding changed')
    fixed = '[[ $# == 0 ]] || { echo \'No command-line arguments accepted by the frozen job\' >&2; exit 2; }\nALLOC8_PLAN='+shlex.quote(str(plan_path))
    return text.replace(original, fixed)


def publish_text(path, text):
    path = Path(path)
    q.require(not path.is_symlink() and not path.parent.is_symlink(), 'Unsafe script publication')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic(plan, campaigns, relative, value):
    q.atomic(campaigns[0][1], Path(plan['runtime_root'])/relative, value)


def mapping_for(plan):
    mapping = binding.load_mapping(Path(plan['runtime_root'])/'physical-mapping.json')
    q.require(mapping['job'] == os.environ.get('SLURM_JOB_ID') and mapping['node'] == socket.gethostname(), 'Mapping job/node differs')
    q.require(mapping['expected_cpus'] == RESOURCE['allocated_cpus'] and
              mapping['expected_mem_gib'] == RESOURCE['allocated_mem_gib'] and
              mapping['expected_seconds'] == RESOURCE['seconds'], 'Mapping resource contract differs')
    return mapping


def eight_records(records, mapping, plan, expected_phase=None):
    q.require(len(records) == 8 and {r['rank'] for r in records} == set(range(8)), 'Need all eight unique rank receipts')
    for record in records:
        device = mapping['gpus'][record['rank']]
        q.require(record['plan_id'] == plan['plan_id'] and record['mapping_id'] == mapping['mapping_id'] and
                  record['job'] == mapping['job'] and record['node'] == mapping['node'] and
                  record['uuid'] == device['uuid'] and record['pci'] == device['pci'] and
                  record['runtime_visible_count'] == 1 and record['actual_cpu_count'] == 4 and
                  len(record['cpu_affinity']) == len(set(record['cpu_affinity'])) == 4 and
                  all(type(c) is int and c >= 0 for c in record['cpu_affinity']),
                  'Rank receipt does not prove its single planned physical GPU/four CPUs')
    q.require(len({r['uuid'] for r in records}) == len({r['pci'] for r in records}) == 8,
              'Ranks collide on a physical GPU')
    q.require(len({r['step'] for r in records}) == 1 and str(records[0]['step']).isdigit() and
              len({r['phase'] for r in records}) == 1 and
              (expected_phase is None or records[0]['phase'] == expected_phase), 'Receipts cross Slurm step/phase')
    q.require(len({c for r in records for c in r['cpu_affinity']}) == 32, 'Rank CPU affinities overlap')
    return records


def verify_gate(plan, campaigns, mapping, name):
    gate = q.read(Path(plan['runtime_root'])/(name+'.json'))
    q.require(gate['plan_id'] == plan['plan_id'] and gate['mapping_id'] == mapping['mapping_id'] and
              gate['passed'] is True, 'Current runtime gate not passed: '+name)
    return gate


def prepare_rank(plan, campaigns, phase):
    mapping = mapping_for(plan)
    expected = binding.rank_environment(mapping, os.environ)
    q.require(all(os.environ.get(key) == expected.get(key) for key in binding.MASKS+('EXPECTED_GPU_UUID', 'EXPECTED_GPU_PCI_BUS_ID')),
              'Must enter through allocated_gpu_uuid exec after Slurm visibility setup')
    rank = int(os.environ['SLURM_PROCID'])
    q.require(len(os.sched_getaffinity(0)) == 4, 'Each actual rank must be bound to exactly four CPUs')
    for name in ('PYTHONPATH', 'PYTHONHOME'):
        os.environ.pop(name, None)
    os.environ.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                      OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    from tabfm_local_tmp import activate
    activate(campaigns[0][1], plan, Path(plan['runtime_root'])/'tmp'/phase/('rank-'+str(rank)))
    import torch
    from pfn_mitra_one import gpu_identity
    gpu = gpu_identity(torch)
    owner = {'job': mapping['job'], 'node': mapping['node'], 'step': os.environ['SLURM_STEP_ID'], 'rank': rank,
             'uuid': gpu['uuid'], 'pci': gpu['pci_bus_id'], 'runtime_visible_count': gpu['runtime_visible_count'],
             'actual_cpu_count': len(os.sched_getaffinity(0)), 'plan_id': plan['plan_id'], 'mapping_id': mapping['mapping_id']}
    owner.update(cpu_affinity=sorted(os.sched_getaffinity(0)), phase=phase)
    q.require(owner['uuid'] == mapping['gpus'][rank]['uuid'] and owner['pci'] == mapping['gpus'][rank]['pci'], 'Actual rank GPU differs')
    full.base.subreaper()
    full.base.STOP = None
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(number, full.base.signal_stop)
    return mapping, owner


def job_cgroup_memory(job):
    lines = Path('/proc/self/cgroup').read_text().splitlines()
    matches = [line.split('::', 1)[1] for line in lines if line.startswith('0::') and '/job_'+job+'/' in line]
    q.require(len(matches) == 1, 'Cannot identify unique v2 Slurm allocation cgroup')
    components = Path(matches[0]).parts
    q.require(components.count('job_'+job) == 1, 'Ambiguous Slurm job cgroup path')
    relative = Path(*components[1:components.index('job_'+job)+1])
    path = Path('/sys/fs/cgroup')/relative/'memory.current'
    value = int(path.read_text().strip())
    q.require(value >= 0, 'Invalid allocation cgroup memory.current')
    return value


def allocation_snapshot():
    """Count this exact Slurm job's same-user process tree, not peer jobs."""
    import psutil
    mine = psutil.Process(os.getpid())
    owned = {mine.pid, *(p.pid for p in mine.children(recursive=True))}
    job = os.environ['SLURM_JOB_ID']
    total = own = 0
    for process in psutil.process_iter():
        try:
            if process.uids().real != os.getuid():
                continue
            cgroup = (Path('/proc')/str(process.pid)/'cgroup').read_text()
            if '/job_'+job+'/' not in cgroup:
                q.require(process.pid not in owned, 'Owned descendant escaped allocation cgroup')
                continue
            rss = process.memory_info().rss
            total += rss
            if process.pid in owned:
                own += rss
        except (psutil.NoSuchProcess, FileNotFoundError, ProcessLookupError):
            pass
        # AccessDenied/other errors propagate; never silently undercount.
    return {'own_tree_rss_bytes': own, 'allocation_rss_bytes': total,
            'allocation_cgroup_bytes': job_cgroup_memory(job),
            'node_available_bytes': psutil.virtual_memory().available, 'epoch': time.time()}


def resource_guard(memory, budget, startup=False):
    if full.base.STOP is not None or budget.remaining() <= 60:
        raise full.base.OperationalDeferral('signal_or_allocation_deadline')
    for key in ('own_tree_rss_bytes', 'allocation_rss_bytes', 'allocation_cgroup_bytes', 'node_available_bytes'):
        q.require(type(memory[key]) is int and memory[key] >= 0, 'Invalid allocation memory observation')
    if memory['own_tree_rss_bytes'] > RESOURCE['rank_rss_gib']*GIB:
        raise full.base.OperationalDeferral('rank_RSS_exceeds28GiB')
    if memory['allocation_rss_bytes'] > RESOURCE['allocation_rss_gib']*GIB:
        raise full.base.OperationalDeferral('aggregate_allocation_RSS_exceeds224GiB')
    if memory['allocation_cgroup_bytes'] >= RESOURCE['allocation_cgroup_gib']*GIB:
        raise full.base.OperationalDeferral('allocation_cgroup_memory_including_cache_reaches240GiB')
    if memory['node_available_bytes'] < RESOURCE['min_available_gib']*GIB:
        raise full.base.OperationalDeferral('node_available_RAM_below8GiB')


@contextmanager
def allocation_guards():
    old = full.base.snapshot, full.base.guard
    full.base.snapshot, full.base.guard = allocation_snapshot, resource_guard
    try:
        yield
    finally:
        full.base.snapshot, full.base.guard = old


def preflight(plan, campaigns):
    mapping, owner = prepare_rank(plan, campaigns, 'preflight')
    resource_guard(allocation_snapshot(), EnvironmentBudget.from_environment())
    atomic(plan, campaigns, 'preflight/rank-'+str(owner['rank'])+'.json', owner)
    return owner


def check_preflight(plan, campaigns):
    mapping = mapping_for(plan)
    records = [q.read(Path(plan['runtime_root'])/'preflight'/('rank-'+str(rank)+'.json')) for rank in range(8)]
    eight_records(records, mapping, plan, expected_phase='preflight')
    atomic(plan, campaigns, 'preflight-gate.json', {'plan_id': plan['plan_id'], 'mapping_id': mapping['mapping_id'],
                                                'passed': True, 'ranks': records})


def smoke(plan, campaigns):
    mapping, owner = prepare_rank(plan, campaigns, 'smoke')
    verify_gate(plan, campaigns, mapping, 'preflight-gate')
    budget = EnvironmentBudget.from_environment()
    results = []
    with allocation_guards():
        # Four original smoke memberships per campaign, assigned once. Ranks4–7
        # still verify actual single-GPU binding and participate in the 8-rank gate.
        if owner['rank'] < 4:
            for path, man, tasks in campaigns:
                task = q.smoke_tasks(man, tasks)[owner['rank']]
                root = Path(man['output_root'])/'allocated8'/plan['run_id']/('rank-'+str(owner['rank']))
                value = full.attempt(plan, path, man, task, owner, budget, root, smoke=True)
                results.append(value)
                q.require(value['state'] == 'complete', 'Original full-data smoke failed; no formal dispatch')
    atomic(plan, campaigns, 'smoke/rank-'+str(owner['rank'])+'.json', {**owner, 'results': results, 'passed': True})


def check_smoke(plan, campaigns):
    mapping = mapping_for(plan)
    verify_gate(plan, campaigns, mapping, 'preflight-gate')
    records = [q.read(Path(plan['runtime_root'])/'smoke'/('rank-'+str(rank)+'.json')) for rank in range(8)]
    eight_records(records, mapping, plan, expected_phase='smoke')
    for rank, receipt in enumerate(records):
        q.require(receipt['passed'] is True and len(receipt['results']) == (len(campaigns) if rank < 4 else 0), 'Incomplete smoke matrix')
        for (path, man, tasks), result in zip(campaigns, receipt['results']):
            task = q.smoke_tasks(man, tasks)[rank]
            q.require(result['state'] == 'complete' and result['owner']['rank'] == rank, 'Smoke failed/rank changed')
            actual = q.valid_result(result['attempt_output'], man, task)
            q.require(actual['physical_gpu']['uuid'] == receipt['uuid'] and actual['physical_gpu']['pci_bus_id'] == receipt['pci'],
                      'Smoke completed on a different physical GPU')
    atomic(plan, campaigns, 'smoke-gate.json', {'plan_id': plan['plan_id'], 'mapping_id': mapping['mapping_id'],
                                            'passed': True, 'ranks': records})


def pending_order(campaigns):
    queues = [sorted(tasks, key=lambda t:(q.work_size(t), t['task_kind'], t['dataset_index'])) for _,_,tasks in campaigns]
    q.require(all(len(queue) == 681 for queue in queues), 'Original full681 membership required')
    for index in range(681):
        for campaign, queue in zip(campaigns, queues):
            yield campaign, queue[index]


def dispatch(plan, campaigns):
    mapping, owner = prepare_rank(plan, campaigns, 'dispatch')
    verify_gate(plan, campaigns, mapping, 'smoke-gate')
    budget = EnvironmentBudget.from_environment()
    summary = {**owner, 'attempts': [], 'state': 'starting'}
    try:
        with allocation_guards():
            for (path, man, _), task in pending_order(campaigns):
                if budget.remaining() <= 120 or full.base.STOP is not None:
                    raise full.base.OperationalDeferral('no_new_task_budget_or_signal')
                output = q.task_path(man, 'results', task)
                if output.exists():
                    if q.read(output).get('complete') is True:
                        q.valid_result(output, man, task)
                    q.require(q.task_path(man, 'claims', task).exists(), 'Existing result lacks canonical claim')
                    continue
                root = Path(man['output_root'])/'allocated8'/plan['run_id']/('rank-'+str(owner['rank']))
                value = full.attempt(plan, path, man, task, owner, budget, root)
                if value['state'] != 'already_claimed':
                    summary['attempts'].append(value)
                if value['state'] in ('retained_resource_deferral', 'operationally_deferred'):
                    raise full.base.OperationalDeferral(value['state'])
            summary['state'] = 'no_unclaimed_work_not_campaign_completion'
    except full.base.OperationalDeferral as exc:
        summary.update(state='operationally_deferred', reason=str(exc))
    finally:
        summary['cleanup'] = full.base.cleanup_children()
        summary['finished_epoch'] = time.time()
        atomic(plan, campaigns, 'finished/rank-'+str(owner['rank'])+'.json', summary)
    return summary


def step_command(plan, mode):
    here = Path(__file__).resolve()
    common = ['srun', '--exact', '--nodes=1', '--gpus=8', '--gpu-bind=none', '--cpu-bind=threads',
              '--kill-on-bad-exit=1', '--unbuffered', '--export=ALL']
    if mode == 'bootstrap':
        return common+['--ntasks=1', '--ntasks-per-node=1', '--cpus-per-task=64', plan['worker_python'],
                       str(here), 'bootstrap', '--plan', plan['_path']]
    q.require(mode in ('preflight', 'smoke', 'run'), 'Unexpected worker stage')
    return common+['--ntasks=8', '--ntasks-per-node=8', '--cpus-per-task=4', plan['worker_python'],
                   str(here.with_name('allocated_gpu_uuid.py')), 'exec', '--mapping',
                   str(Path(plan['runtime_root'])/'physical-mapping.json'), '--',
                   plan['worker_python'], str(here), mode, '--plan', plan['_path']]


def launch_job(plan, campaigns, plan_path, family):
    """Run only inside the already allocated batch job; this never sbatches."""
    q.require(plan['family'] == family and os.environ.get('SLURM_JOB_ID', '').isdigit() and
              os.environ.get('SLURM_JOB_NUM_NODES') == '1', 'Wrong family or not an allocated single-node job')
    atomic(plan, campaigns, 'job-start.json', {'plan_id': plan['plan_id'], 'job': os.environ['SLURM_JOB_ID'],
                                            'node': socket.gethostname(), 'epoch': time.time()})
    runtime_plan = {**plan, '_path': str(Path(plan_path).resolve())}
    # The new allocation is owned in full; srun sets its task visibility again.
    env = dict(os.environ)
    for key in (*binding.MASKS, 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    state, error = 'failed', None
    try:
        subprocess.run(step_command(runtime_plan, 'bootstrap'), env=env, check=True)
        subprocess.run(step_command(runtime_plan, 'preflight'), env=env, check=True)
        check_preflight(plan, campaigns)
        subprocess.run(step_command(runtime_plan, 'smoke'), env=env, check=True)
        check_smoke(plan, campaigns)
        subprocess.run(step_command(runtime_plan, 'run'), env=env, check=True)
        state = 'workers_finished_not_campaign_completion'
    except BaseException as exc:
        error = type(exc).__name__+': '+str(exc)
        raise
    finally:
        atomic(plan, campaigns, 'job-finished.json', {'plan_id': plan['plan_id'], 'job': os.environ['SLURM_JOB_ID'],
               'state': state, 'error': error, 'epoch': time.time()})
    return {'state': state}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify', 'launch-job', 'bootstrap', 'preflight', 'check-preflight', 'smoke', 'check-smoke', 'run'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--family', choices=['tabfm', 'tabswift'])
    parser.add_argument('--run-id')
    parser.add_argument('--campaign', action='append', type=Path)
    args = parser.parse_args(argv)
    q.require(__debug__, 'Optimized Python unsupported')
    if args.mode == 'prepare':
        q.require(args.family and args.run_id, 'prepare requires --family and --run-id')
        print(json.dumps(prepare(args.family, args.run_id, args.campaign)), flush=True)
        return
    q.require(args.plan is not None, '--plan required')
    plan, campaigns = load_plan(args.plan)
    if args.mode == 'verify':
        value = {'valid': True, 'plan_id': plan['plan_id'], 'family': plan['family'], 'job_name': plan['job_name']}
    elif args.mode == 'launch-job':
        value = launch_job(plan, campaigns, args.plan, args.family)
    elif args.mode == 'bootstrap':
        value = binding.bootstrap(Path(plan['runtime_root'])/'physical-mapping.json',
                                  expected_cpus=RESOURCE['allocated_cpus'], expected_mem_gib=RESOURCE['allocated_mem_gib'],
                                  expected_seconds=RESOURCE['seconds'])
    else:
        function = {'preflight': preflight, 'check-preflight': check_preflight, 'smoke': smoke,
                    'check-smoke': check_smoke, 'run': dispatch}[args.mode]
        value = function(plan, campaigns)
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
