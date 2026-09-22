"""Explicitly authorized CPU-contention sidecar, preserving scientific results.

One worker per approved allocation, native16/32/64 CPUs, at most128GiB and2h.
No idle-CPU test; memory/current, exact CPU binding, parent lifetime, sources,
claim flock/CAS and cleanup remain enforced. Existing206117.377 is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import types

import table6_ag_existing_v3 as v3
from table6_restart_deadline import derive_deadline, EnvironmentBudget, parse_duration, parse_fields

ag, old = v3.ag, v3.old
require, read, publish = ag.require, ag.read, ag.publish
ORIGINAL_LOAD = ag.load_original
STAGE = ag.ROOT / 'stage/table6_autogluon_fleet_20260922_v1'
CAMPAIGN = STAGE.name
GIB = 1024 ** 3
ALLOWED = dict(zip(('196092','196093','200798','200797','204828','204827','204826',
                   '208823','206117','194259','194181','194180'),
                  ('auh7-1b-gpu-'+n for n in ('196','197','209','254','295','257','284',
                                             '200','306','307','264','255'))))
PROTECTED_JOB = '206117'
PROTECTED_STEP = '377'
PROTECTED_CPUS = list(range(16)) + list(range(32, 48)) + list(range(64, 80)) + list(range(96, 112))
PROTECTED_MEMORY = 256 * GIB
PINS = {
    'table6_ag_existing_v3.py': '109055f3213fc121c40a9ccb33da2ab3216f61bbfd9e8e7dbec484aab9f3a106',
    'table6_restart_ag.py': '3ca3bf4d620b026f29409a9361630f0110227c116a68317a975948ccb79ecfaf',
    'table6_restart_ag_sidecar.py': '4b7a609073c3a362795a820ce1f8a336f304e48d5fcb6841387ba06780026a21',
    'table6_restart_deadline.py': '047352b1b52c947ea3b2017692c422d65f451b56dca18283564383b51a8b9950',
}


def sources():
    values = {name: hashlib.sha256((v3.REPO / name).read_bytes()).hexdigest() for name in PINS}
    require(values == PINS, 'frozen source changed')
    values[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return values


def load_frozen():
    original, plan, pairs = ORIGINAL_LOAD()
    require(all(pair['config'].get('time_limit', 'missing') is None for pair in pairs),
            'native CPU parallelism extension requires no per-fit time budget')
    return original, plan, pairs


def parent_fields(raw, parent, node, ranks=1, uid=None):
    require(ALLOWED.get(parent) == node and ranks == 1, 'unapproved parent/node or worker count')
    fields = parse_fields(raw)
    require(fields.get('JobId') == parent and fields.get('JobState') == 'RUNNING', 'parent not exact RUNNING')
    require(fields.get('NumNodes') == '1' and fields.get('NodeList') == node, 'wrong node or multinode parent')
    uid = os.getuid() if uid is None else uid
    require(re.fullmatch(r'[^()]+\(' + str(uid) + r'\)', fields.get('UserId', '')), 'wrong parent owner')
    require(int(fields['NumCPUs']) >= 16, 'parent lacks16 CPUs')
    allocated = dict(item.split('=', 1) for item in fields.get('AllocTRES', '').split(',') if '=' in item)
    memory = old.memory_bytes(allocated.get('mem', fields.get('MinMemoryNode', '0')))
    require(memory > 0, 'unknown allocated memory')
    require(parse_duration(fields['TimeLimit']) - parse_duration(fields['RunTime']) > 900,
            'parent has insufficient bounded fit/cleanup time')
    return fields


def validate_capacity(proof, parent, node, ranks=1, now=None):
    require(ALLOWED.get(parent) == node and ranks == 1, 'unapproved capacity target')
    require(proof.get('allow_cpu_sidecar') is True and proof.get('allow_cpu_contention') is True and
            proof.get('parent_job_id') == parent and proof.get('node') == node, 'contention authorization/target absent')
    observations = proof.get('observations', [])
    require(len(observations) == 2, 'exactly two capacity samples required')
    now = time.time() if now is None else now
    times = []
    for row in observations:
        require(row.get('parent_job_id') == parent and row.get('node') == node, 'sample parent/node changed')
        stamp = float(row['observed_epoch'])
        require(math.isfinite(stamp) and 0 <= now - stamp <= 300, 'stale/future capacity sample')
        times.append(stamp)
        for key in ('parent_memory_current_bytes', 'parent_memory_limit_bytes', 'available_memory_bytes'):
            require(type(row.get(key)) is int and row[key] >= 0, 'invalid memory counter: ' + key)
        require(row['parent_memory_limit_bytes'] > 0 and row.get('parent_memory_source') in ('cgroup_v1','cgroup_v2'),
                'whole-parent cgroup memory required')
        cpus = row.get('parent_cpu_ids')
        require(isinstance(cpus, list) and len(cpus) == len(set(cpus)) and len(cpus) >= 16 and
                all(type(cpu) is int and cpu >= 0 for cpu in cpus), 'invalid actual parent CPU set')
    require(times[1] - times[0] >= 15, 'capacity samples must be15s apart')
    return proof


def resource_plan(raw, proof, parent, node, now=None):
    fields = parent_fields(raw, parent, node)
    validate_capacity(proof, parent, node, now=now)
    rows = proof['observations']
    allocated = dict(item.split('=', 1) for item in fields.get('AllocTRES', '').split(',') if '=' in item)
    slurm_memory = old.memory_bytes(allocated.get('mem', fields.get('MinMemoryNode', '0')))
    limit = min(slurm_memory, *(row['parent_memory_limit_bytes'] for row in rows))
    margin = min(64 * GIB, max(8 * GIB, limit // 8))
    protected_memory = PROTECTED_MEMORY if parent == PROTECTED_JOB else 0
    protected_cpus = PROTECTED_CPUS if parent == PROTECTED_JOB else []
    free = min(limit - max(row['parent_memory_current_bytes'] for row in rows) - margin - protected_memory,
               min(row['available_memory_bytes'] for row in rows) - 32 * GIB - protected_memory)
    memory = min(128 * GIB, (free // GIB) * GIB)
    require(memory >= 32 * GIB, 'insufficient actual parent memory; no file-cache credit allowed')
    cpus = sorted(set(rows[0]['parent_cpu_ids']) & set(rows[1]['parent_cpu_ids']) - set(protected_cpus))
    logical_budget = int(fields['NumCPUs']) - len(protected_cpus)
    choices = [n for n in (16, 32, 64) if n <= logical_budget and n <= len(cpus)]
    require(choices, 'insufficient CPUs after preserving existing AG reservation')
    count = max(choices)
    selected = cpus[:count]
    remaining = parse_duration(fields['TimeLimit']) - parse_duration(fields['RunTime'])
    hard_seconds = min(7200, remaining - 300)
    require(hard_seconds >= 600, 'no useful bounded step window')
    return {'ranks': 1, 'cpus_per_rank': count, 'selected_cpu_ids': selected, 'memory_bytes': memory,
            'rss_limit_bytes': min(120 * GIB, memory - 8 * GIB), 'allocated_memory_bytes': slurm_memory,
            'parent_margin_bytes': margin, 'node_margin_bytes': 32 * GIB,
            'protected_memory_bytes': protected_memory, 'protected_cpu_ids': protected_cpus,
            'protected_step': '206117.377' if parent == PROTECTED_JOB else None,
            'step_time_limit_seconds': hard_seconds, 'gpus': 0, 'cpu_contention_authorized': True,
            'idle_cpu_check': False, 'file_cache_credit_bytes': 0}


def clean_environment(cpus=16):
    env = old.clean_environment()
    env.pop('PYTHONHOME', None)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key] = str(cpus)
    return env


def srun_command(parent, node, resources, directory):
    require(ALLOWED.get(parent) == node, 'unapproved launch target')
    mask = hex(sum(1 << cpu for cpu in resources['selected_cpu_ids']))
    seconds = resources['step_time_limit_seconds']
    duration = f'{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}'
    return ['srun', '--jobid='+parent, '--nodelist='+node, '--nodes=1', '--ntasks=1', '--ntasks-per-node=1',
            '--cpus-per-task='+str(resources['cpus_per_rank']), '--mem='+str(resources['memory_bytes']//GIB)+'G',
            '--gpus=0', '--gpus-per-task=0', '--gres=none', '--overlap', '--exact', '--immediate=10',
            '--time='+duration, '--cpu-bind=mask_cpu:'+mask, '--kill-on-bad-exit=1', '--input=none', '--export=ALL',
            'env', 'CPU_ONLY=1', 'CUDA_VISIBLE_DEVICES=', 'HIP_VISIBLE_DEVICES=-1', 'ROCR_VISIBLE_DEVICES=-1',
            'GPU_DEVICE_ORDINAL=-1', old.PYTHON, '-B', str(Path(__file__).resolve()), 'rank', '--launch-dir', str(directory)]


def verify_plan(directory):
    directory = Path(directory).resolve()
    require(directory.parent == (STAGE/'launches').resolve(), 'foreign fleet namespace')
    plan = read(directory/'plan.json')
    require(ALLOWED.get(plan['parent']) == plan['node'] and plan['ranks'] == 1 and plan['plan_id'] == ag.PLAN,
            'fleet/scientific identity changed')
    require(plan['source_hashes'] == sources() and plan['operational_digest'] ==
            ag.digest({k:v for k,v in plan.items() if k != 'operational_digest'}), 'fleet source/plan changed')
    require(plan['command'] == srun_command(plan['parent'], plan['node'], plan['resources'], directory), 'fleet command changed')
    return plan


def check_memory(snapshot, resources, startup=False):
    additional = resources['memory_bytes'] if startup else 0
    require(snapshot['parent_memory_current_bytes'] + additional + resources['parent_margin_bytes'] +
            resources['protected_memory_bytes'] <= snapshot['parent_memory_limit_bytes'], 'whole_parent_memory_guard')
    require(snapshot['available_memory_bytes'] >= additional + resources['node_margin_bytes'] +
            resources['protected_memory_bytes'], 'node_memory_guard')


def launch(parent, node, proof_path, launch_id):
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,80}', launch_id or ''), 'invalid immutable launch identity')
    require(not Path(proof_path).is_symlink(), 'symlink proof refused')
    data = Path(proof_path).read_bytes(); proof = json.loads(data)
    raw = old.command(['scontrol','show','job','-o',parent], env=dict(clean_environment(), TZ='UTC'))
    resources = resource_plan(raw, proof, parent, node)
    load_frozen()  # Validate all457 memberships/configs before any launch write.
    directory = STAGE/'launches'/launch_id
    plan = {'plan_id': ag.PLAN, 'parent': parent, 'node': node, 'ranks': 1, 'launch_id': launch_id,
            'created_epoch': time.time(), 'source_hashes': sources(), 'resources': resources,
            'allocated_memory_bytes': resources['allocated_memory_bytes'], 'parent_snapshot': raw,
            'capacity_proof': proof, 'capacity_proof_sha256': hashlib.sha256(data).hexdigest(),
            'command': srun_command(parent, node, resources, directory),
            'scope': 'CPU-only native parallelism; exact frozen15seeds; authorized contention; no parent mutation'}
    plan['operational_digest'] = ag.digest(plan)
    publish(STAGE/'parents'/(parent+'.json'), {'launch_id':launch_id,'plan':plan})
    directory.mkdir(parents=True, exist_ok=False); publish(directory/'plan.json', plan)
    with (directory/'supervisor.log').open('x') as log:
        child = subprocess.Popen([old.PYTHON,'-B',str(Path(__file__).resolve()),'supervise','--launch-dir',str(directory)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=clean_environment(resources['cpus_per_rank']), start_new_session=True, close_fds=True)
    receipt = {'state':'supervisor_started_not_yet_step_verified','pid':child.pid,'parent':parent,'epoch':time.time()}
    publish(directory/'launch.json',receipt)
    return receipt


def rank_entry(directory):
    directory = Path(directory); plan = verify_plan(directory); res = plan['resources']
    parent, node = plan['parent'], plan['node']; step = os.environ.get('SLURM_STEP_ID','')
    require(os.environ.get('SLURM_JOB_ID') == parent and socket.gethostname() == node and step.isdigit(), 'wrong step identity')
    require(os.environ.get('SLURM_PROCID') == '0' and os.environ.get('SLURM_NTASKS') == '1' and
            os.environ.get('SLURM_CPUS_PER_TASK') == str(res['cpus_per_rank']) and os.environ.get('SLURM_NNODES') == '1',
            'actual native CPU step resources differ')
    require(sorted(os.sched_getaffinity(0)) == res['selected_cpu_ids'], 'Slurm binding differs; never widen actual CPU mask')
    require(all(os.environ.get(k) == val for k,val in ag.CPU_ENV.items()) and os.environ.get('GPU_DEVICE_ORDINAL') == '-1', 'GPU mask lost')
    validate_capacity(plan['capacity_proof'], parent, node)
    snapshot = v3.memory_snapshot(parent,res['allocated_memory_bytes']); check_memory(snapshot,res,startup=True)
    started=time.monotonic(); raw=old.command(['scontrol','show','job','-o',parent],env=dict(os.environ,TZ='UTC'))
    finished,wall=time.monotonic(),time.time(); fields=parent_fields(raw,parent,node)
    derived=derive_deadline(raw,job_id=parent,query_started_monotonic=started,query_finished_monotonic=finished,
                            observed_local_epoch=wall,expected_limit_seconds=parse_duration(fields['TimeLimit']),
                            safety_margin_seconds=300,minimum_remaining_seconds=180,hostname=node)
    # Conservatively count all elapsed time since immutable launch creation as
    # possible step time; startup/queue delay can only shorten this budget.
    age=wall-plan['created_epoch']; require(0 <= age <= 300, 'launch too old/future for start budget')
    remaining=min(derived.safe_remaining_seconds,res['step_time_limit_seconds']-age-300)
    require(remaining > 180,'no safe work budget remains')
    budget={'step_id':step,'memory':snapshot,'remaining_seconds':remaining,'parent_derivation':derived.record(),
            'environment':{'JOB_BUDGET_END_EPOCH':str(wall+remaining),'JOB_BUDGET_END_MONOTONIC':str(finished+remaining),
                           'JOB_BUDGET_MONOTONIC_HOST':node,'JOB_BUDGET_JOB_ID':parent}}
    publish(directory/'rank-0.json',{'job_id':parent,'step_id':step,'host':node,'rank':0,'pid':os.getpid(),
                                    'cpu_affinity':sorted(os.sched_getaffinity(0)),'native_fit_cpus':res['cpus_per_rank']})
    publish(directory/'budget.json',budget); os.environ.update(budget['environment'])
    os.nice(19-os.getpriority(os.PRIO_PROCESS,0))
    os.execvpe('ionice',['ionice','-c','3',old.PYTHON,'-B',str(Path(__file__).resolve()),'work','--launch-dir',str(directory)],os.environ)


def preflight(original,budget,audit,stop,ranks,plan):
    res=plan['resources']; require(ranks == 1 and os.environ.get('SLURM_NTASKS') == '1','one rank required')
    require(os.environ.get('SLURM_CPUS_PER_TASK') == str(res['cpus_per_rank']) and
            sorted(os.sched_getaffinity(0)) == res['selected_cpu_ids'],'native CPU contract changed')
    require(all(os.environ.get(k) == val for k,val in ag.CPU_ENV.items()) and os.environ.get('GPU_DEVICE_ORDINAL') == '-1','CPU-only mask required')
    require(os.getpriority(os.PRIO_PROCESS,0) == 19 and budget.monotonic_end is not None and 0<budget.remaining()<=6901,'nice/budget invalid')
    record={'pass':True,'plan_id':ag.PLAN,'rank':0,'job_id':plan['parent'],'step_id':os.environ['SLURM_STEP_ID'],
            'host':plan['node'],'pid':os.getpid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'nice':19,'environment':dict(ag.CPU_ENV),'worker_sha256':old.WORKER_SHA,
            'native_fit_cpus':res['cpus_per_rank'],'adapter_sources':sources(),**original.cpu_check()}
    require(record['devices'] == 0 and record.get('fastai'),'zero GPU / fastai preflight failed')
    publish(audit/'preflight-0.json',record)
    return record


def normalized_query(command, **kwargs):
    if command[:2] == ['squeue','--steps']:
        command=['%i' if item=='%i|%T' else item for item in command]
    result=subprocess.run(command,**kwargs)
    if command[0]=='squeue' and result.returncode!=0 and not result.stdout.strip() and \
            result.stderr.strip()=='slurm_load_jobs error: Invalid job id specified':
        return types.SimpleNamespace(returncode=result.returncode,stdout=result.stdout,stderr='squeue: error: Invalid job id specified')
    return result


def exact_step_evidence(claim, *, run=None, now=None):
    return v3.exact_step_evidence(claim,run=normalized_query if run is None else run,now=now)


def guarded_run_seed(command,log,lock_fd,budget,stop,plan):
    require(command[:4]==[sys.executable,'-B',str(v3.REPO/'table6_restart_ag.py'),'seed'],'unknown frozen seed command')
    if stop.reason or budget.remaining()<=ag.NEW_FIT_GUARD:return None,stop.reason or 'allocation_budget'
    res=plan['resources']
    try:check_memory(v3.memory_snapshot(plan['parent'],res['allocated_memory_bytes']),res)
    except (RuntimeError,OSError,ValueError) as exc:return None,'memory_guard: '+str(exc)
    directory=STAGE/'launches'/plan['launch_id']
    rewritten=[*command[:2],str(Path(__file__).resolve()),*command[3:],'--launch-dir',str(directory)]
    env=dict(os.environ,T6_AG_PARENT_PID=str(os.getpid()),T6_AG_LOCK_FD=str(lock_fd))
    with Path(log).open('x') as handle:
        child=subprocess.Popen(rewritten,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True,pass_fds=(lock_fd,))
        reason=None
        try:
            while child.poll() is None:
                processes=ag.process_snapshot()
                require(os.getpid() in processes,'own process RSS unavailable')
                rss=processes[os.getpid()]['rss']+sum(row['rss'] for row in ag.descendants(processes,os.getpid()).values())
                try:check_memory(v3.memory_snapshot(plan['parent'],res['allocated_memory_bytes']),res)
                except (RuntimeError,OSError,ValueError) as exc:reason='memory_guard: '+str(exc);break
                if stop.reason or budget.remaining()<=ag.STOP_GUARD or rss>res['rss_limit_bytes']:
                    reason=stop.reason or ('allocation_budget' if budget.remaining()<=ag.STOP_GUARD else 'rss_guard');break
                time.sleep(.5)
            code=child.poll()
        finally:
            try:ag.clean_children(child)
            except Exception as exc:raise ag.UnsafeCleanup('fleet descendants unproven; retain pair lock') from exc
        return code,reason or stop.reason or ('allocation_budget' if budget.remaining()<=ag.STOP_GUARD else None)


def seed_entry(directory,key,seed):
    plan=verify_plan(directory); res=plan['resources']
    require(os.environ.get('SLURM_JOB_ID')==plan['parent'] and socket.gethostname()==plan['node'] and
            os.environ.get('SLURM_CPUS_PER_TASK')==str(res['cpus_per_rank']) and
            sorted(os.sched_getaffinity(0))==res['selected_cpu_ids'],'seed native resources changed')
    require(re.fullmatch(r'[a-f0-9]{24}',key or '') and seed in ag.SEEDS,'invalid seed identity')
    ag.load_original=load_frozen
    audit=ag.OUT/'auxiliary'/CAMPAIGN/f"j{plan['parent']}-s{os.environ['SLURM_STEP_ID']}"
    publish(audit/f'native-cpu-{key}-{seed:02d}.json',{'native_cpus':res['cpus_per_rank'],'actual_affinity':res['selected_cpu_ids'],
            'model_configuration_unchanged':True,'fit_time_limit_is_none':True,'operational_digest':plan['operational_digest']})
    ag.seed_entry(key,seed)


PRIVATE=dict(vars(v3))
PRIVATE.update(STAGE=STAGE,CAMPAIGN=CAMPAIGN,verify_plan=verify_plan,validate_capacity=validate_capacity,
               validate_parent=parent_fields,clean_environment=clean_environment,sources=sources,
               exact_step_evidence=exact_step_evidence,guarded_run_seed=guarded_run_seed)
for _name in ('supervise','work_entry'):
    _fn=getattr(v3,_name)
    PRIVATE[_name]=types.FunctionType(_fn.__code__,PRIVATE,_fn.__name__,_fn.__defaults__,_fn.__closure__)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('launch','supervise','rank','work','seed'))
    p.add_argument('--parent');p.add_argument('--node');p.add_argument('--proof',type=Path);p.add_argument('--launch-id')
    p.add_argument('--launch-dir',type=Path);p.add_argument('--key');p.add_argument('--seed',type=int);a=p.parse_args(argv)
    if a.action=='launch':print(json.dumps(launch(a.parent,a.node,a.proof,a.launch_id)),flush=True);return 0
    plan=verify_plan(a.launch_dir)
    if a.action=='supervise':
        PRIVATE['clean_environment']=lambda:clean_environment(plan['resources']['cpus_per_rank'])
        return PRIVATE['supervise'](a.launch_dir)
    if a.action=='rank':rank_entry(a.launch_dir);return 0
    if a.action=='seed':seed_entry(a.launch_dir,a.key,a.seed);return 0
    ag.load_original=load_frozen
    PRIVATE['dynamic_preflight']=lambda original,budget,audit,stop,ranks:preflight(original,budget,audit,stop,ranks,plan)
    return PRIVATE['work_entry'](a.launch_dir)


if __name__=='__main__':sys.exit(main())
