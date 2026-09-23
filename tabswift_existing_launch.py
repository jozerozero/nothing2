"""Borrow one genuinely empty existing GPU through the video's cooperative lock.

No allocation submission, video signal, training edit, or canonical overwrite.
The inherited eight-device access is immediately narrowed to one physical UUID.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from eval_one import publish_new, require
from table6_restart_deadline import parse_duration, parse_fields, derive_deadline
import tabswift_existing_fp32 as worker

VIDEO_ROOT = Path('/vast/users/guangyi.chen/causal_group/jinyuan.hu/eec-bench/EECBench/experiments/holocine_generation_200_20260922')


def clean_environment():
    blocked = ('SLURM_', 'SBATCH_', 'SRUN_', 'PMI_', 'PMIX_', 'OMPI_')
    remove = {'PYTHONPATH', 'PYTHONHOME', 'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
              'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'}
    env = {k:v for k,v in os.environ.items() if not k.startswith(blocked) and k not in remove}
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', PYTHONUNBUFFERED='1')
    return env


def check_parent(plan):
    raw = subprocess.check_output(['scontrol', 'show', 'job', '-o', plan['parent_job_id']], text=True, timeout=30)
    f = parse_fields(raw)
    require(f.get('JobId') == '214135' and f.get('JobState') == 'RUNNING'
            and f.get('UserId', '').startswith('guangyi.chen(') and f.get('MinMemoryNode') == '2T',
            'Existing parent changed or does not have the verified memory contract')
    nodes = subprocess.check_output(['scontrol', 'show', 'hostnames', f['NodeList']], text=True, timeout=30).splitlines()
    require(plan['node'] in nodes and f.get('TresPerNode') == 'gres/gpu:mi210:8', 'GPU ownership changed')
    require(parse_duration(f['TimeLimit'])-parse_duration(f['RunTime']) > 7500, 'Parent expires too soon')
    return raw, f


def launch(path):
    plan, _ = worker.load_plan(path)
    raw, _ = check_parent(plan)
    stage = Path(plan['runtime_root'])
    command = ['srun', '--jobid='+plan['parent_job_id'], '--nodelist='+plan['node'],
               '--overlap', '--exact', '--nodes=1', '--ntasks=1', '--cpus-per-task=4',
               '--mem=64G', '--gres=gpu:mi210:8', '--gpu-bind=none', '--cpu-bind=cores',
               '--time=02:00:00', '--immediate=10', '--input=none', '--export=ALL',
               '--job-name=swfp32one', '--unbuffered', plan['worker_python'], '-B',
               str(Path(__file__).resolve()), 'node', '--plan', str(Path(path).resolve())]
    intent = {'plan_id':plan['plan_id'], 'epoch':time.time(), 'command':command,
              'controller_source':worker.identity(__file__), 'parent_control':raw,
              'actual_model_gpus':1, 'new_allocation':False, 'video_signals_sent':0}
    publish_new(stage/'launch-intent.json', intent)
    with (stage/'launch.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=clean_environment(), start_new_session=True)
    receipt = dict(intent, state='dispatched_not_yet_preflighted', controller_pid=child.pid)
    publish_new(stage/'launch-receipt.json', receipt)
    return receipt


def hardware(plan):
    d = Path('/sys/bus/pci/devices')/plan['gpu']['pci']
    require((d/'unique_id').read_text().strip().lower() == plan['gpu']['uuid'], 'Physical UUID/PCI mismatch')
    data = {'busy':int((d/'gpu_busy_percent').read_text()), 'vram':int((d/'mem_info_vram_used').read_text())}
    require(data['busy'] == 0 and data['vram'] < 128*2**20, 'GPU is not genuinely empty')
    return data


def owners(plan):
    import psutil
    matches = []
    for proc in psutil.process_iter(['pid', 'uids', 'cmdline']):
        try:
            if proc.pid == os.getpid() or proc.info['uids'].real != os.getuid():
                continue
            env = proc.environ()
            if ('GPU-'+plan['gpu']['uuid']) in env.get('ROCR_VISIBLE_DEVICES','').split(','):
                matches.append({'pid':proc.pid, 'cmd':proc.info['cmdline']})
        except psutil.NoSuchProcess:
            pass
    require(not matches, 'Physical GPU still assigned to a live worker: '+str(matches))
    return matches


def node(path):
    import psutil
    import ctypes
    node_started = time.monotonic()
    plan, _ = worker.load_plan(path)
    stage = Path(plan['runtime_root'])
    intent = json.loads((stage/'launch-intent.json').read_text())
    require(worker.q.verify_file(intent['controller_source']) == Path(__file__).resolve()
            and intent['plan_id'] == plan['plan_id'], 'Wrong controller source or runtime plan')
    require(socket.gethostname() == plan['node'] and os.environ.get('SLURM_JOB_ID') == plan['parent_job_id']
            and os.environ.get('SLURM_NTASKS') == '1' and os.environ.get('SLURM_PROCID') == '0'
            and os.environ.get('SLURM_STEP_ID', '').isdigit(), 'Wrong real single-rank child')
    raw, f = check_parent(plan)
    lease = VIDEO_ROOT/'gpu_leases'/(plan['node']+'_GPU-'+plan['gpu']['uuid']+'.lock')
    require(lease.parent.is_dir() and not lease.is_symlink(), 'Missing/unsafe cooperative GPU lease')
    fd = os.open(lease, os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.set_inheritable(fd, True)
    require(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0, 'Cannot enable child cleanup')
    child = None
    try:
        owners(plan); first = hardware(plan)
        allowed = sorted(os.sched_getaffinity(0))
        sample = psutil.cpu_percent(interval=1, percpu=True)
        available = [c for c in allowed if c < len(sample) and sample[c] < 50]
        require(len(available) >= 4, 'Fewer than four available allocated CPUs')
        selected = sorted(available, key=lambda c:(sample[c], c))[:4]
        os.sched_setaffinity(0, set(selected))
        require(psutil.virtual_memory().available >= 128*2**30, 'Insufficient node memory headroom')
        owners(plan); second = hardware(plan)
        started = time.monotonic()
        raw = subprocess.check_output(['scontrol', 'show', 'job', '-o', plan['parent_job_id']], text=True, timeout=30)
        finished = time.monotonic()
        budget = derive_deadline(raw, job_id=plan['parent_job_id'], query_started_monotonic=started,
            query_finished_monotonic=finished, observed_local_epoch=time.time(),
            expected_limit_seconds=parse_duration(f['TimeLimit']))
        seconds = min(node_started+6900-time.monotonic(), budget.safe_remaining_seconds)
        require(seconds > 600, 'Insufficient bounded runtime')
        env = dict(os.environ)
        for key in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL','PYTHONPATH','PYTHONHOME'):
            env.pop(key, None)
        env.update(budget.environment())
        env.update(ROCR_VISIBLE_DEVICES='GPU-'+plan['gpu']['uuid'], EXPECTED_GPU_UUID=plan['gpu']['uuid'],
            EXPECTED_GPU_PCI_BUS_ID=plan['gpu']['pci'], TABSWIFT_GPU_LEASE_FD=str(fd),
            JOB_BUDGET_END_EPOCH=str(time.time()+seconds), JOB_BUDGET_END_MONOTONIC=str(time.monotonic()+seconds),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
        own = worker.process_identity(os.getpid()); st = os.fstat(fd)
        gate = {'schema':'tabswift_existing_gpu_gate_v1', 'plan_id':plan['plan_id'],
            'job':plan['parent_job_id'], 'node':plan['node'], 'step':os.environ['SLURM_STEP_ID'],
            'uid':os.getuid(), 'created_monotonic':time.monotonic(), 'cpu_ids':sorted(selected), 'gpu':plan['gpu'],
            'release_method':'cooperative_gpu_lease', 'paused_video_owners':[], 'no_other_gpu_owners':True,
            'cooperative_lease':{'path':str(lease),'holder_pid':os.getpid(),'start_ticks':own['start_ticks'],
                'uid':os.getuid(),'fd':fd,'device':st.st_dev,'inode':st.st_ino},
            'hardware_samples':[first,second], 'video_processes_interrupted':0}
        publish_new(Path(plan['gate_path']), gate)
        os.nice(19)
        subprocess.run(['ionice','-c3','-p',str(os.getpid())], capture_output=True, check=True)
        child = subprocess.Popen([plan['worker_python'],'-B',str(Path(__file__).with_name('tabswift_existing_fp32.py')),
            'run','--plan',str(Path(path).resolve())],env=env,pass_fds=(fd,),start_new_session=True)
        def forward(signum, frame):
            if child.poll() is None:
                try: os.killpg(child.pid, signum)
                except ProcessLookupError: pass
        signal.signal(signal.SIGTERM, forward); signal.signal(signal.SIGINT, forward)
        try:
            code = child.wait(timeout=seconds+30)
        except subprocess.TimeoutExpired:
            forward(signal.SIGTERM, None)
            code = child.wait(timeout=20)
        publish_new(stage/'controller-terminal.json', {'exit_code':code, 'epoch':time.time(),
            'step':os.environ['SLURM_STEP_ID'], 'gpu':plan['gpu'], 'video_processes_interrupted':0})
        return {'exit_code':code, 'step':os.environ['SLURM_STEP_ID']}
    finally:
        mine = psutil.Process(os.getpid())
        for sig in (signal.SIGTERM, signal.SIGKILL):
            children = mine.children(recursive=True)
            for proc in reversed(children):
                try: proc.send_signal(sig)
                except psutil.NoSuchProcess: pass
            psutil.wait_procs(children, timeout=10)
            while True:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                    if not pid: break
                except ChildProcessError: break
            if not mine.children(recursive=True): break
        require(not mine.children(recursive=True), 'Owned child cleanup failed; inherited GPU lease retained')
        os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch','node'))
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    result=launch(args.plan) if args.mode=='launch' else node(args.plan)
    print(json.dumps(result),flush=True)
    if result.get('exit_code'): raise SystemExit(result['exit_code'])
