"""Read-only node/GPU sampling inside the explicit user-owned allocations."""
import argparse,concurrent.futures,json,os,re,socket,subprocess,sys,time
from pathlib import Path

PARENTS=['196092','196093','200798','200797','204828','204827','204826','206117','206116','194259','194181','194180']

def node_snapshot():
    devices=[]
    for p in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
        if not (p/'unique_id').is_file():continue
        try:
            devices.append({'uuid':(p/'unique_id').read_text().strip().lower(),
              'drm':str(p),'pci':p.resolve().name,'busy':int((p/'gpu_busy_percent').read_text()),
              'vram':int((p/'mem_info_vram_used').read_text()),'vram_total':int((p/'mem_info_vram_total').read_text())})
        except (OSError,ValueError) as e:devices.append({'error':str(e),'drm':str(p)})
    import psutil
    procs=[]
    for p in psutil.process_iter(['pid','uids','memory_info','name','cmdline']):
        try:
            if p.info['uids'].real==os.getuid():
                procs.append({'pid':p.pid,'name':p.info['name'],'rss':p.info['memory_info'].rss,
                              'cmd':' '.join(p.info['cmdline'] or [])[:250]})
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    print(json.dumps({'parent':os.environ['SLURM_JOB_ID'],'node':socket.gethostname(),
         'epoch':time.time(),'gpus':devices,'owned_processes':procs,'available_ram':psutil.virtual_memory().available}))

def inspect_parent(parent):
    raw=subprocess.check_output(['scontrol','show','job',parent,'-o'],text=True)
    f=dict(re.findall(r'(\w+)=(\S+)',raw))
    if not (f.get('UserId','').startswith('guangyi.chen(') and f.get('JobState')=='RUNNING' and f.get('NumNodes')=='1'):
        return {'parent':parent,'excluded':'not running owned single node','raw':raw}
    # These specific allocations own all eight physical GPUs; never inspect/use
    # cards of a different user on a partial-node one-GPU allocation.
    if not re.search(r'(?:^|,)gres/gpu=8(?:,|$)',f.get('AllocTRES','')):
        return {'parent':parent,'excluded':'not all8 GPU allocation','raw':raw}
    cmd=['srun',f'--jobid={parent}','--overlap','--exact','-N1','-n1','-c1','--mem=1G',
         '--gres=none','--gpus=0','--time=00:02:00',sys.executable,str(Path(__file__).resolve()),'--node']
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    if p.returncode:return {'parent':parent,'error':p.stderr,'stdout':p.stdout,'job_fields':f}
    d=json.loads(p.stdout.strip().splitlines()[-1]);d['job_fields']=f
    d['steps']=subprocess.check_output(['squeue','--steps','-j',parent,'-h','-o','%i|%j|%M|%N'],text=True)
    return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--node',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.node:return node_snapshot()
    assert a.output and not a.output.exists()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(inspect_parent,PARENTS))
    a.output.write_text(json.dumps({'epoch':time.time(),'parents':rows},indent=2)+'\n')
    for r in rows:
        print(json.dumps({k:r.get(k) for k in ('parent','node','gpus','available_ram','excluded','error','steps')}),flush=True)
if __name__=='__main__':main()
