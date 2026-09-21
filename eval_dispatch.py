"""Bounded evaluation child steps in explicitly audited existing allocations.

Only our subprocess groups may be stopped. Parent allocations are never changed.
Idle physical UUIDs, not Slurm/DRM ordinal guesses, select evaluation devices.
Every checkpoint/dataset has a single atomic claim and immutable result.
"""
import argparse
import concurrent.futures
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

GIB = 1024 ** 3
IDLE_MAX = 64 * 1024 ** 2
PARENTS = {'196092','196093','200798','200797','204828','204827','204826','206117','206116','194259','194181','194180'}

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, obj, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('x') as h:
        json.dump(obj, h, indent=2, allow_nan=False)
        h.flush(); os.fsync(h.fileno())
    try:
        if immutable: os.link(tmp, path)
        else: os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()

def gpu(uuid_value):
    for p in Path('/sys/class/drm').glob('card[0-9]*/device'):
        if (p/'unique_id').exists() and (p/'unique_id').read_text().strip().lower() == uuid_value:
            return {'uuid':uuid_value, 'pci':p.resolve().name,
                    'busy':int((p/'gpu_busy_percent').read_text()), 'vram':int((p/'mem_info_vram_used').read_text())}
    raise RuntimeError('Physical GPU absent: '+uuid_value)

def check_idle(g):
    return g.get('busy') == 0 and 0 <= g.get('vram', -1) < IDLE_MAX

def build_plan(a):
    first, second = read(a.first), read(a.second)
    assert 10 <= second['epoch'] - first['epoch'], 'two distinct samples required'
    assert time.time() - second['epoch'] < 3600, 'capacity observations expired'
    old = {p['parent']:p for p in first['parents']}
    selections, skipped = [], []
    offset = 0
    for p in second['parents']:
        parent = p['parent']; q = old.get(parent,{})
        if parent not in PARENTS or p.get('error') or p.get('excluded') or q.get('error') or q.get('excluded'):
            skipped.append({'parent':parent,'reason':'allocation/probe unavailable'}); continue
        fields = p['job_fields']
        assert fields['NumNodes']=='1' and fields['UserId'].startswith('guangyi.chen(')
        assert fields['NodeList'] == q['job_fields']['NodeList'] == p['node']
        before = {g.get('uuid'):g for g in q['gpus']}
        idle = [g for g in p['gpus'] if check_idle(g) and check_idle(before.get(g.get('uuid'),{}))
                and before[g['uuid']]['pci'] == g['pci']]
        # RSS is deliberately conservative (shared pages may be counted twice).
        # Unlike host MemAvailable, this is also bounded by parent's reservation.
        rss = sum(x['rss'] for x in p['owned_processes'])
        match = re.search(r'(?:^|,)mem=([0-9.]+)([KMGTP])(?:,|$)',fields['AllocTRES'])
        assert match, fields
        mem = float(match[1]) * 1024 ** ('KMGTP'.index(match[2])+1)
        per_slot = 12*GIB if mem < 256*GIB else 100*GIB
        max_slots = 4 if mem < 256*GIB else 8
        slots = min(len(idle), max_slots, max(0, int((mem-rss-8*GIB)//per_slot)))
        if not slots:
            skipped.append({'parent':parent,'reason':'no safely idle GPU or no reserved RAM headroom',
                            'idle_cards':len(idle),'owned_rss_gib':rss/GIB}); continue
        chosen = idle[:slots]
        selections.append({'parent':parent,'node':p['node'],'gpus':chosen,'offset':offset,
            'threads_per_gpu':4,'rss_limit_gib':10 if mem < 256*GIB else 96,
            'step_memory_gib':int(slots*per_slot/GIB), 'parent_memory_gib':mem/GIB,
            'observed_owned_rss_gib':rss/GIB,'time_limit':'20:00:00',
            'sample_epochs':[q['epoch'],p['epoch']], 'all_idle_cards':len(idle),
            'reserved_spare_gpus':len(idle)-slots})
        offset += slots
    obj={'created_epoch':time.time(),'nodes':selections,'skipped':skipped,
         'actual_evaluation_gpus':offset,'parent_allocations_unchanged':True,
         'physical_gpu_policy':'zero busy and <64MiB in two samples, rechecked immediately before use',
         'RAM_policy':'within parent reservation, cap4x10GiB evaluator RSS on64GiB parents'}
    atomic(a.plan,obj,True); print(json.dumps(obj),flush=True)

def stop_own(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid, signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL);proc.wait(timeout=15)

def worker(cfg, g, ordinal, manifest_path, root, deadline):
    import psutil
    man=read(manifest_path)
    identity=f"{cfg['parent']}.{os.environ.get('SLURM_STEP_ID','unknown')}.{g['uuid']}"
    env=os.environ.copy()
    for key in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'):
        env.pop(key,None)
    env.update(ROCR_VISIBLE_DEVICES='GPU-'+g['uuid'], EXPECTED_GPU_UUID=g['uuid'],
        EXPECTED_GPU_PCI_BUS_ID=g['pci'], PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false')
    cpus=os.sched_getaffinity(0)
    assert len(cpus)>=cfg['threads_per_gpu']*len(cfg['gpus']), 'CPU affinity too small'
    worker_cpus=sorted(cpus)[(ordinal-cfg['offset'])*4:(ordinal-cfg['offset']+1)*4]
    startup=gpu(g['uuid'])
    if not check_idle(startup):
        atomic(root/'workers'/f'{identity}.json',{'state':'skipped_gpu_no_longer_idle','gpu':startup},True)
        return
    order=man['checkpoints']; start=ordinal%len(order)
    order=order[start:]+order[:start]
    # Small first, but never drop rows. The full test sequence is always retained.
    rows=sorted(man['rows'],key=lambda r:sum(x['size_bytes'] for x in r['input_files']))
    done=0; failures=0
    try:
        for ckpt in order:
            for row in rows:
                if time.time()>deadline: return
                key=f"step-{ckpt['step']}/row-{row['dataset_index']:03d}"
                result=root/'results'/f'{key}.json'; error=root/'errors'/f'{key}.json'
                claim=root/'claims'/f'{key}.json'
                if result.exists() or error.exists():continue
                claim.parent.mkdir(parents=True,exist_ok=True)
                owner={'worker':identity,'pid':os.getpid(),'node':socket.gethostname(),'started_epoch':time.time(),
                       'checkpoint_step':ckpt['step'],'dataset_index':row['dataset_index'],
                       'manifest_id':man['manifest_id']}
                try:
                    with claim.open('x') as h:json.dump(owner,h);h.flush();os.fsync(h.fileno())
                except FileExistsError:continue
                log=root/'logs'/f'{key}.log';log.parent.mkdir(parents=True,exist_ok=True)
                cmd=['taskset','-c',','.join(map(str,worker_cpus)),sys.executable,
                    str(Path(__file__).with_name('eval_one.py')),'--manifest',str(manifest_path),
                    '--checkpoint-step',str(ckpt['step']),'--dataset-index',str(row['dataset_index']),
                    '--output',str(result),'--threads','4']
                reason=None; proc=None; begin=time.time(); max_rss=0
                state={'state':'running','gpu':g,'checkpoint_step':ckpt['step'],'dataset':row['dataset'],
                       'completed':done,'failed':failures,'started_epoch':begin,'worker':identity}
                atomic(root/'workers'/f'{identity}.json',state)
                try:
                    # Check again between tasks: our previous child has exited,
                    # and an unrelated new workload must not be displaced.
                    if not check_idle(gpu(g['uuid'])):
                        reason='GPU became occupied between evaluator subprocesses';return
                    with log.open('x') as h:
                        proc=subprocess.Popen(cmd,env=env,stdout=h,stderr=subprocess.STDOUT,
                                              stdin=subprocess.DEVNULL,start_new_session=True)
                        last=0
                        while proc.poll() is None:
                            try:
                                p=psutil.Process(proc.pid)
                                rss=sum(x.memory_info().rss for x in [p]+p.children(recursive=True) if x.is_running())
                                max_rss=max(max_rss,rss)
                            except (psutil.NoSuchProcess,psutil.AccessDenied):rss=0
                            if rss>cfg['rss_limit_gib']*GIB:
                                reason='own evaluator exceeded reserved host RAM budget';stop_own(proc);break
                            if time.time()-begin>5400 or time.time()>deadline:
                                reason='bounded evaluator runtime reached';stop_own(proc);break
                            if time.time()-last>60:
                                atomic(root/'workers'/f'{identity}.json',{**state,'heartbeat_epoch':time.time(),'rss':rss})
                                last=time.time()
                            try:proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:pass
                    if proc.returncode==0 and result.exists():
                        obj=read(result)
                        assert obj.get('complete') is True and obj['checkpoint_step']==ckpt['step']
                        assert obj['dataset_index']==row['dataset_index']
                        done+=1
                    else:
                        failures+=1
                        atomic(error,{**owner,'complete':False,'reason':reason or 'evaluator exited',
                            'returncode':proc.returncode,'log':str(log),'max_rss_gib':max_rss/GIB,
                            'elapsed_s':time.time()-begin,'error_tail':log.read_text(errors='replace')[-6000:]},True)
                        # Systematic environment/protocol failures must not waste all tasks.
                        if failures>=3 and done==0:return
                except BaseException:
                    if proc is not None:stop_own(proc)
                    raise
                finally:
                    if claim.exists() and read(claim).get('worker')==identity:claim.unlink()
    finally:
        atomic(root/'workers'/f'{identity}.json',{'state':'worker_exited','worker':identity,
               'completed':done,'failed':failures,'ended_epoch':time.time(),'gpu':g})

def node_run(a):
    plan=read(a.plan); cfg=next(n for n in plan['nodes'] if n['parent']==a.parent)
    assert os.environ.get('SLURM_JOB_ID')==a.parent and socket.gethostname()==cfg['node']
    root=Path(a.manifest).parent
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    deadline=time.time()+19.5*3600
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cfg['gpus'])) as pool:
        futures=[pool.submit(worker,cfg,g,cfg['offset']+i,Path(a.manifest),root,deadline)
                 for i,g in enumerate(cfg['gpus'])]
        for f in futures:f.result()

def launch(a):
    root=Path(a.manifest).parent; plan=read(a.plan)
    assert len(read(a.manifest)['rows'])==224
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for cfg in plan['nodes']:
            if a.only_parent and cfg['parent']!=a.only_parent:continue
            receipt=root/'launches'/f"{cfg['parent']}.json"
            if receipt.exists():continue
            raw=subprocess.check_output(['scontrol','show','job',cfg['parent'],'-o'],text=True)
            fields=dict(re.findall(r'(\w+)=(\S+)',raw))
            assert fields['JobState']=='RUNNING' and fields['NodeList']==cfg['node']
            assert fields['UserId'].startswith('guangyi.chen(') and fields['NumNodes']=='1'
            assert re.search(r'(?:^|,)gres/gpu=8(?:,|$)',fields['AllocTRES'])
            cmd=['srun','--jobid='+cfg['parent'],'--overlap','--exact','-N1','-n1',
                '-c'+str(4*len(cfg['gpus'])),'--mem='+str(cfg['step_memory_gib'])+'G',
                '--gpus=8','--gpu-bind=none','--time='+cfg['time_limit'],'--unbuffered',
                '--job-name=regft50eval',sys.executable,str(Path(__file__).resolve()),'node',
                '--plan',str(Path(a.plan).resolve()),'--manifest',str(Path(a.manifest).resolve()),'--parent',cfg['parent']]
            log=root/'launches'/f"{cfg['parent']}.log";log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x') as h:
                p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            rec={'parent':cfg['parent'],'node':cfg['node'],'launcher_pid':p.pid,'command':cmd,
                 'physical_evaluation_gpus':[g['uuid'] for g in cfg['gpus']],
                 'cpu_threads':4*len(cfg['gpus']),'memory_gib':cfg['step_memory_gib'],
                 'started_epoch':time.time(),'log':str(log),'new_allocation':False,
                 'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
            atomic(receipt,rec,True);print(json.dumps(rec),flush=True)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('plan');s.add_argument('--first',required=True);s.add_argument('--second',required=True);s.add_argument('--plan',required=True)
    for mode in ('node','launch'):
        s=sub.add_parser(mode);s.add_argument('--manifest',required=True);s.add_argument('--plan',required=True)
        if mode=='node':s.add_argument('--parent',required=True)
        else:s.add_argument('--only-parent')
    a=p.parse_args();{'plan':build_plan,'node':node_run,'launch':launch}[a.mode](a)
if __name__=='__main__':main()
