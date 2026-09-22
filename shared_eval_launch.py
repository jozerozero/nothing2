"""Audited single-physical-GPU child launch in an unchanged owned allocation.

The Slurm child inherits access to its parent's eight GPUs so the proven idle
physical UUID can be selected without guessing Slurm/ROCm ordinal equivalence.
Only that UUID is exposed to evaluation. No allocation is submitted or changed.
"""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from table6_restart_deadline import parse_duration, parse_fields, derive_deadline
from tabfm_default_dispatch import digest, read, require, verify_file
from table6_restart_ag import publish

ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
STAGE = ROOT/'stage/shared_eval_existing_20260922_v1'
ALLOWED = {'196093', '200797', '204828', '204827', '204826', '194259'}
GIB = 1024**3


def check_parent(raw, plan):
    f = parse_fields(raw)
    require(f.get('JobId') == plan['parent_job_id'] and f.get('JobState') == 'RUNNING'
            and f.get('NumNodes') == '1' and f.get('NodeList') == plan['node'], 'parent identity/state changed')
    require(f.get('UserId', '').startswith('guangyi.chen('), 'wrong owner')
    require({'cpu=64','mem=64G','gres/gpu=8'} <= set(f.get('AllocTRES','').split(',')),
            'requires unchanged64CPU64GiB8GPU allocation')
    require(parse_duration(f['TimeLimit'])-parse_duration(f['RunTime']) > 7500,
            'insufficient remaining parent lifetime')
    return f


def load_plan(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(STAGE) and path.name == 'plan.json', 'unexpected operational plan path')
    p = read(path)
    require(p['plan_id'] == digest({k:v for k,v in p.items() if k!='plan_id'}), 'plan digest mismatch')
    require(p['parent_job_id'] in ALLOWED and re.fullmatch(r'[A-Za-z0-9_-]+',p['sidecar_id']), 'bad target')
    require(p['max_step_seconds']==7200 and p['mem_gib']==40 and p['parent_mem_gib']==64,
            'unexpected memory/time budget')
    require(p['cpus'] in (4,16), 'unsupported actual CPU budget')
    startup_limit=22 if p['entry_script'].endswith('table6_existing_gap190.py') else 20
    require(p['reviewed_ownership']['approved'] is True and p['reviewed_ownership']['rationale'],
            'ownership evidence not reviewed')
    require(p['proof']['gpu_idle_verified'] is True and p['proof']['resources_available_verified'] is True,
            'resources not verified')
    require(len(p['proof']['sample_records'])==2, 'requires two capacity samples')
    samples=[read(verify_file(rec)) for rec in p['proof']['sample_records']]
    require(samples[1]['started_epoch']-samples[0]['epoch']>=15,'capacity observations too close')
    require(p['proof']['controller_finished_epoch']==samples[1]['epoch'],'proof freshness identity differs')
    for sample in samples:
        observations=[x for x in sample['parents'] if x['parent']==p['parent_job_id']]
        require(len(observations)==1,'ambiguous parent observation')
        observation=observations[0]
        require(observation.get('complete') is True and observation['node']==p['node'], 'incomplete resource inspection')
        require(observation['same_uid_rss_bytes']<=startup_limit*GIB,'sample memory headroom insufficient')
        cards=[g for g in observation['gpus'] if g['uuid']==p['gpu']['uuid'] and g['pci']==p['gpu']['pci']]
        require(len(cards)==1 and cards[0]['hardware_idle'] is True and not cards[0]['foreign_fd_owner_pids'],
                'GPU evidence not idle/exclusively owned')
    for rec in p['source_records']:
        verify_file(rec)
    require(any(Path(r['path']).resolve()==Path(__file__).resolve() for r in p['source_records']),
            'launcher must be source pinned')
    entry=Path(p['entry_script']).resolve(strict=True)
    require(entry.parent==Path(__file__).resolve().parent and
            entry.name in ('table6_existing_gap190.py','shared_foundation_sidecar.py'), 'unexpected entry')
    require(any(Path(r['path']).resolve()==entry for r in p['source_records']), 'entry not source pinned')
    require(Path(p['python']).is_absolute() and Path(p['python']).is_file(), 'missing interpreter')
    return path,p


def clean_environment():
    prefixes=('SLURM_','SBATCH_','SRUN_','PMI_','PMIX_','OMPI_')
    env={k:v for k,v in os.environ.items() if not k.startswith(prefixes)}
    for k in ('PYTHONPATH','PYTHONHOME','CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES',
              'ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'):
        env.pop(k,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    return env


def launch(path):
    path,p=load_plan(path)
    require(0 <= time.time()-p['proof']['controller_finished_epoch'] <= 300,'resource proof stale')
    control=subprocess.run(['scontrol','show','job','-o',p['parent_job_id']],capture_output=True,text=True,check=True).stdout
    check_parent(control,p)
    directory=path.parent
    command=['srun','--jobid='+p['parent_job_id'],'--nodelist='+p['node'],'--overlap','--exact',
             '--immediate=10','--nodes=1','--ntasks=1','--cpus-per-task=64','--mem=40G',
             '--gpus=8','--gpu-bind=none','--cpu-bind=cores','--time=02:00:00','--input=none',
             '--job-name='+p['short_name'],'--unbuffered','--export=ALL',
             p['python'],'-B',str(Path(__file__).resolve()),'node','--plan',str(path)]
    intent={'state':'launching','plan_id':p['plan_id'],'epoch':time.time(),'command':command,
            'parent_control':control,'actual_model_gpus':1,'actual_model_cpus':p['cpus'],
            'parent_unchanged':True,'new_allocation':False,
            'note':'child inherits all8GPU access; exactly one physicalUUID exposed after fresh idle check'}
    publish(directory/'launch-intent.json',intent)
    with (directory/'launch.log').open('x') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               env=clean_environment(),start_new_session=True)
    receipt={**intent,'state':'srun_dispatched_not_yet_preflighted','controller_pid':child.pid}
    publish(directory/'launch-receipt.json',receipt)
    return receipt


def gpu_idle(g):
    require(re.fullmatch(r'[0-9a-f]{16}',g['uuid']) and
            re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',g['pci']), 'bad GPU identity')
    d=Path('/sys/bus/pci/devices')/g['pci']
    require((d/'unique_id').read_text().strip().lower()==g['uuid'],'GPU UUID mismatch')
    rec={'busy':int((d/'gpu_busy_percent').read_text()),
         'vram':int((d/'mem_info_vram_used').read_text()),'uuid':g['uuid'],'pci':g['pci']}
    require(rec['busy']==0 and rec['vram']<128*1024**2,'target GPU is no longer genuinely idle')
    return rec


def enable_subreaper():
    libc=ctypes.CDLL(None,use_errno=True)
    require(libc.prctl(36,1,0,0,0)==0,'cannot enable owned descendant subreaper')


def cleanup_owned_descendants():
    """Only descendants of this supervisor; never enumerate/kill peer workers."""
    import psutil
    me=psutil.Process(os.getpid())
    for sig,seconds in ((signal.SIGTERM,10),(signal.SIGKILL,10)):
        children=me.children(recursive=True)
        for proc in reversed(children):
            try: proc.send_signal(sig)
            except psutil.NoSuchProcess: pass
        psutil.wait_procs(children,timeout=seconds)
        while True:
            try:
                pid,_=os.waitpid(-1,os.WNOHANG)
                if not pid: break
            except ChildProcessError: break
        if not me.children(recursive=True): return
    require(not me.children(recursive=True),'owned descendants survived shutdown; no clean lease-release receipt')


def cleanup_or_hold(stage, identity):
    try:
        cleanup_owned_descendants()
    except BaseException as exc:
        # Keep the physical lease until Slurm tears down this whole child step.
        # Returning/raising here would close the flock while a fit may survive.
        try:
            publish(stage/'cleanup-fatal.json',{**identity,'error':repr(exc),'epoch':time.time(),
                    'state':'lease_retained_waiting_for_step_cgroup_teardown'})
        finally:
            while True:
                signal.pause()


def node(path):
    path,p=load_plan(path)
    require(socket.gethostname()==p['node'] and os.environ.get('SLURM_JOB_ID')==p['parent_job_id'],
            'node/parent mismatch')
    require(os.environ.get('SLURM_NTASKS')=='1' and os.environ.get('SLURM_PROCID')=='0'
            and os.environ.get('SLURM_STEP_ID','').isdigit(), 'genuine single-rank child required')
    stage=path.parent
    lockroot=ROOT/'evaluation/shared_gpu_leases_20260922_v1'
    lockroot.mkdir(parents=True,exist_ok=True)
    lock=lockroot/(p['node']+'.'+p['gpu']['uuid']+'.lock')
    parent_lock=lockroot/('parent-'+p['parent_job_id']+'.40GiB.lock')
    require(not lock.is_symlink() and not parent_lock.is_symlink(),'unsafe resource lock')
    with parent_lock.open('a') as resource, lock.open('a') as held:
        fcntl.flock(resource,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic()
        raw=subprocess.run(['scontrol','show','job','-o',p['parent_job_id']],capture_output=True,text=True,check=True).stdout
        finish=time.monotonic()
        f=check_parent(raw,p)
        budget=derive_deadline(raw,job_id=p['parent_job_id'],query_started_monotonic=start,
                              query_finished_monotonic=finish,observed_local_epoch=time.time(),
                              expected_limit_seconds=parse_duration(f['TimeLimit']))
        remaining=min(6900,budget.safe_remaining_seconds)
        env_budget=budget.environment()
        env_budget.update(JOB_BUDGET_END_EPOCH=str(time.time()+remaining),
                          JOB_BUDGET_END_MONOTONIC=str(time.monotonic()+remaining))
        import psutil
        existing=[]
        for proc in psutil.process_iter(['pid','uids','memory_info']):
            try:
                if proc.info['uids'].real==os.getuid(): existing.append(proc.info['memory_info'].rss)
            except psutil.NoSuchProcess: pass
        startup_limit=22 if p['entry_script'].endswith('table6_existing_gap190.py') else 20
        require(sum(existing)<=startup_limit*GIB,'parent lacks40GiB plus at least2GiB safety headroom')
        allowed=sorted(os.sched_getaffinity(0))
        samples=[psutil.cpu_percent(interval=1,percpu=True) for _ in range(2)]
        idle=[c for c in allowed if all(c<len(s) and s[c]<25 for s in samples)]
        require(len(idle)>=p['cpus'],'insufficient currently idle allowed CPUs')
        chosen=sorted(idle,key=lambda c:(max(s[c] for s in samples),c))[:p['cpus']]
        os.sched_setaffinity(0,set(chosen))
        gpu=gpu_idle(p['gpu'])
        original_visibility={k:os.environ.get(k) for k in
                             ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')}
        env=dict(os.environ)
        for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL','PYTHONPATH','PYTHONHOME'):
            env.pop(k,None)
        env.update(env_budget,ROCR_VISIBLE_DEVICES='GPU-'+p['gpu']['uuid'],
                   EXPECTED_GPU_UUID=p['gpu']['uuid'],EXPECTED_GPU_PCI_BUS_ID=p['gpu']['pci'],
                   OMP_NUM_THREADS=str(p['cpus']),MKL_NUM_THREADS=str(p['cpus']),
                   OPENBLAS_NUM_THREADS=str(p['cpus']),NUMEXPR_NUM_THREADS=str(p['cpus']))
        if p['entry_script'].endswith('table6_existing_gap190.py'):
            base=ROOT/'stage/table6_remaining19_standard_hpo_bg8_20260912_v1'
            env['PYTHONPATH']=':'.join(map(str,(base,base/'TALENT',ROOT/'stage/table6_nonfoundation_all_benchmarks_20260823_v1/python_packages')))
        os.nice(19)
        subprocess.run(['ionice','-c3','-p',str(os.getpid())],check=True,capture_output=True)
        identity={'job':p['parent_job_id'],'step':os.environ['SLURM_STEP_ID'],'node':p['node'],
                  'plan_id':p['plan_id'],'gpu':gpu,'actual_cpu_ids':chosen,'inherited_cpu_ids':allowed,
                  'same_uid_rss_bytes':sum(existing),'original_visibility':original_visibility,
                  'child_visibility':{'ROCR_VISIBLE_DEVICES':env['ROCR_VISIBLE_DEVICES']},
                  'budget':env_budget,'epoch':time.time(),'state':'resource_gate_passed'}
        publish(stage/'node-resource-gate.json',identity)
        enable_subreaper()
        child=subprocess.Popen([p['python'],'-B',p['entry_script'],'--plan',str(path)],env=env,start_new_session=True)
        def forward(signum,frame):
            if child.poll() is None:
                try: os.killpg(child.pid,signum)
                except ProcessLookupError: pass
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1): signal.signal(sig,forward)
        try:
            code=child.wait(timeout=max(1,remaining+30))
        except BaseException:
            cleanup_or_hold(stage,identity)
            raise
        cleanup_or_hold(stage,identity)
        publish(stage/'node-terminal.json',{**identity,'state':'exited','exit_code':code,'finished_epoch':time.time()})
        if code: raise SystemExit(code)
        return {'state':'exited','exit_code':code}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('launch','node'));parser.add_argument('--plan',required=True)
    args=parser.parse_args();print(json.dumps(launch(args.plan) if args.mode=='launch' else node(args.plan)),flush=True)
