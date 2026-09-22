"""Read-only CPU/cgroup headroom samples for two explicitly owned 2TiB parents."""
import argparse, concurrent.futures, json, os, re, socket, subprocess, time
from pathlib import Path
import shared_gpu_capacity_probe as common

PARENTS = {'206116': 'auh7-1b-gpu-308', '206117': 'auh7-1b-gpu-306'}
ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')

def parent_path(parent, controllers, mounts):
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, names, relative = line.split(':', 2)
        if controllers and controllers not in names.split(','):
            continue
        if not controllers and names:
            continue
        parts = Path(relative).parts
        if 'job_' + parent not in parts:
            continue
        relative = Path(*parts[:parts.index('job_' + parent) + 1])
        for mount in mounts:
            left, right = mount.split(' - ', 1)
            lf, rf = left.split(), right.split()
            if (not controllers and rf[0] == 'cgroup2') or (controllers and rf[0] == 'cgroup' and controllers in rf[2].split(',')):
                mount_root = Path(lf[3]); mount_point = Path(lf[4])
                if relative.is_relative_to(mount_root):
                    return mount_point / relative.relative_to(mount_root)
    raise RuntimeError('Cannot prove exact parent cgroup for ' + str(controllers))

def expand(value):
    result = set()
    for item in value.strip().split(','):
        if not item: continue
        a, _, b = item.partition('-')
        result.update(range(int(a), int(b or a) + 1))
    return sorted(result)

def node(parent):
    common.require(parent in PARENTS and os.environ.get('SLURM_JOB_ID') == parent, 'wrong parent')
    common.require(socket.gethostname() == PARENTS[parent], 'wrong node')
    mounts = Path('/proc/self/mountinfo').read_text().splitlines()
    try:
        memory = parent_path(parent, None, mounts)
        current = int((memory/'memory.current').read_text())
        limit = int((memory/'memory.max').read_text())
        cpu_root = memory
        cpus = expand((cpu_root/'cpuset.cpus.effective').read_text())
        source = 'cgroup_v2'
    except RuntimeError:
        memory = parent_path(parent, 'memory', mounts)
        current = int((memory/'memory.usage_in_bytes').read_text())
        limit = int((memory/'memory.limit_in_bytes').read_text())
        cpu_root = parent_path(parent, 'cpuset', mounts)
        cpu_file = cpu_root/'cpuset.effective_cpus'
        if not cpu_file.exists(): cpu_file = cpu_root/'cpuset.cpus'
        cpus = expand(cpu_file.read_text())
        source = 'cgroup_v1'
    common.require(cpus and limit > current >= 0, 'invalid parent cgroup counters')
    import psutil
    samples = [psutil.cpu_percent(interval=1, percpu=True) for _ in range(2)]
    idle = [c for c in cpus if all(c < len(s) and s[c] < 25 for s in samples)]
    return {'parent_job_id': parent, 'node': PARENTS[parent], 'observed_epoch': time.time(),
            'parent_memory_current_bytes': current, 'parent_memory_limit_bytes': limit,
            'parent_memory_source': source,
            'available_memory_bytes': psutil.virtual_memory().available,
            'idle_cpu_ids': idle, 'parent_cpu_ids': cpus,
            'memory_cgroup': str(memory), 'cpuset_cgroup': str(cpu_root),
            'cpu_utilization_samples': samples, 'step_id': os.environ['SLURM_STEP_ID']}

def collect(parent):
    f = common.parsed_fields(common.run(['scontrol','show','job','-o',parent]))
    common.require(f.get('JobState') == 'RUNNING' and f.get('NumNodes') == '1' and f.get('NodeList') == PARENTS[parent], 'parent not live on expected node')
    common.require(f.get('UserId','').endswith('('+str(os.getuid())+')'), 'wrong owner')
    command = ['srun','--jobid='+parent,'--nodelist='+PARENTS[parent], '--overlap','--exact', '--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=1G','--gpus=0','--gpus-per-task=0','--gres=none','--time=00:02:00','--immediate=10','--input=none','--export=ALL','env','CUDA_VISIBLE_DEVICES=','HIP_VISIBLE_DEVICES=-1','ROCR_VISIBLE_DEVICES=-1','GPU_DEVICE_ORDINAL=-1','nice','-n','19',common.PYTHON,'-B',str(Path(__file__).resolve()),'--node',parent]
    text = common.run(command, timeout=150, env=common.probe_environment())
    lines = [v[len('AG_CAPACITY='):] for v in text.splitlines() if v.startswith('AG_CAPACITY=')]
    common.require(len(lines) == 1, 'no unique node sample')
    result = json.loads(lines[0]); result['controller_received_epoch'] = time.time()
    return result

def main():
    p = argparse.ArgumentParser(); p.add_argument('--node', choices=PARENTS); p.add_argument('--output',type=Path)
    p.add_argument('--launch-samples', nargs=2, type=Path)
    p.add_argument('--launch-v4-samples', nargs=2, type=Path)
    a = p.parse_args()
    if a.launch_v4_samples:
        import table6_ag_existing_v4 as worker
        rows = [{v['parent_job_id']:v for v in json.loads(path.read_text())['observations']} for path in a.launch_v4_samples]
        for parent,node_name in PARENTS.items():
            proof={'allow_cpu_sidecar':True,'parent_job_id':parent,'node':node_name,'observations':[row[parent] for row in rows]}
            worker.v3.validate_capacity(proof,parent,node_name,2)
        for parent,node_name in PARENTS.items():
            proof={'allow_cpu_sidecar':True,'parent_job_id':parent,'node':node_name,'observations':[row[parent] for row in rows]}
            path=worker.STAGE/('capacity-'+parent+'-22d.json')
            common.publish(path,proof)
            print(json.dumps(worker.launch(parent,node_name,path,'ag-'+parent+'-22d',worker.V3_STAGE/'launches'/('ag-'+parent+'-22d'))),flush=True)
    elif a.launch_samples:
        import table6_ag_existing_v3 as worker
        rows = [{v['parent_job_id']:v for v in json.loads(path.read_text())['observations']} for path in a.launch_samples]
        for parent,node_name in PARENTS.items():
            proof = {'allow_cpu_sidecar':True,'parent_job_id':parent,'node':node_name,
                     'observations':[row[parent] for row in rows]}
            worker.validate_capacity(proof,parent,node_name,4)
        for parent,node_name in PARENTS.items():
            proof = {'allow_cpu_sidecar':True,'parent_job_id':parent,'node':node_name,
                     'observations':[row[parent] for row in rows]}
            path = worker.STAGE/('capacity-'+parent+'-22d.json')
            common.publish(path,proof)
            receipt=worker.launch(parent,node_name,4,path,'ag-'+parent+'-22d')
            print(json.dumps(receipt),flush=True)
    elif a.node:
        print('AG_CAPACITY='+json.dumps(node(a.node)))
    else:
        common.require(a.output is not None and not a.output.exists(), 'new sample output required')
        common.publish(a.output.with_suffix('.intent.json'), {'parents':list(PARENTS),'epoch':time.time()})
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: rows=list(pool.map(collect,PARENTS))
        common.publish(a.output, {'observations':rows,'epoch':time.time()})
        print(json.dumps({'path':str(a.output),'samples':[{k:v for k,v in r.items() if k != 'cpu_utilization_samples'} for r in rows]}))
if __name__ == '__main__': main()
