"""One-shot held submission for new, explicitly prepared single-node eval scripts."""
import argparse, hashlib, json, os, re, subprocess, time
from pathlib import Path
from table6_restart_ag import publish, require
from table6_restart_deadline import parse_fields

ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
CONTRACTS = {'tabfm':(64,256,'3-00:00:00'), 'tabswift':(64,256,'3-00:00:00'), 'gap190':(128,512,'02:00:00')}

def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=60)
    require(p.returncode == 0, f'{args[0]} rc={p.returncode}: {p.stderr[-2000:]}')
    return p.stdout.strip()

def context(path, family, name):
    require(not Path(path).is_symlink(), 'script symlink refused')
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT/'stage') and path.name == 'run.sh' and path.is_file(), 'unexpected script path')
    require(re.fullmatch(r'(?:tfm22d|tsw22d|t6g22d)', name), 'name outside this request')
    script = path.read_text()
    require(all(x in script for x in ('#SBATCH --nodes=1','#SBATCH --account=faculty-acc','#SBATCH --qos=bgqos','#SBATCH --partition=faculty','#SBATCH --no-requeue','#SBATCH --nice=0')), 'script resource contract missing')
    directives = dict(re.findall(r'^#SBATCH --([a-z-]+)=(\S+)$',script,re.M))
    require(directives.get('chdir') == str(path.parent), 'working directory mismatch')
    cpus, memory, limit = CONTRACTS[family]
    require(directives.get('mem') == str(memory)+'G', 'memory mismatch')
    require(directives.get('gpus') == '8', 'script must request eight GPUs')
    require('dependency' not in directives, 'independent job cannot have dependency')
    require(directives.get('time') in ({'72:00:00','3-00:00:00'} if family != 'gap190' else {'02:00:00'}), 'time mismatch')
    require(int(directives.get('ntasks','1')) * int(directives.get('cpus-per-task','1')) == cpus, 'CPU count mismatch')
    command(['bash','-n',str(path)])
    return path,script,directives,cpus,memory,limit

def verify(plan, held):
    path,script,dirs,cpus,mem,limit=context(plan['script'],plan['family'],plan['name'])
    require(hashlib.sha256(script.encode()).hexdigest() == plan['script_sha256'], 'script changed')
    job=plan['job_id']; f=parse_fields(command(['scontrol','show','job','-o',job]))
    expected={'JobId':job,'JobName':plan['name'],'Account':'faculty-acc','Partition':'faculty','QOS':'bgqos',
              'NumCPUs':str(cpus),'MinMemoryNode':str(mem)+'G','Nice':'0','Requeue':'0','Dependency':'(null)',
              'TimeLimit':limit,'WorkDir':str(path.parent),'Command':str(path),
              'NumTasks':dirs.get('ntasks','1'),'CPUs/Task':dirs.get('cpus-per-task','1')}
    for key,value in expected.items(): require(f.get(key)==value,f'{key}: {f.get(key)} != {value}')
    require(f.get('NumNodes') in ('1','1-1'),'not a single node')
    tres=dict(x.split('=',1) for x in f['ReqTRES'].split(','))
    require(tres.get('gres/gpu') == '8' and tres.get('cpu') == str(cpus),'not eight allocated GPUs')
    for key,field in (('output','StdOut'),('error','StdErr')):
        require(f.get(field)==dirs[key].replace('%j',job), 'log path mismatch')
    require(f.get('UserId','').endswith('('+str(os.getuid())+')'),'wrong owner')
    if held: require(f['JobState']=='PENDING' and f['Reason']=='JobHeldUser','job not held')
    spool=path.parent/('verified-spool-'+job+'.sh')
    require(not spool.is_symlink(), 'spool symlink refused')
    if not spool.exists(): command(['scontrol','write','batch_script',job,str(spool)])
    require(spool.read_text()==script,'spooled script differs')
    return f

def submit(path,family,name):
    path,script,dirs,_,_,_=context(path,family,name)
    ledger=path.parent/'node_submission.json'
    require(not ledger.exists(),'existing submission intent: inspect, do not retry')
    queue=command(['squeue','-u','guangyi.chen','-h','-o','%i|%j|%Z'])
    require(not any(row.split('|')[1]==name or row.split('|')[-1]==str(path.parent) for row in queue.splitlines()),'existing same-name/stage job')
    plan={'script':str(path),'family':family,'name':name,'script_sha256':hashlib.sha256(script.encode()).hexdigest(),
          'state':'submission_intent','epoch':time.time(),'commit':command(['git','-C',str(Path(__file__).parent),'rev-parse','HEAD'])}
    publish(ledger,plan)
    raw=command(['sbatch','--hold','--parsable','--job-name='+name,str(path)])
    job=raw.split(';')[0]; require(job.isdigit(),'unknown sbatch result; do not retry')
    plan.update(job_id=job,state='submitted_held'); publish(ledger,plan,replace=True)
    plan['verified_fields']=verify(plan,True);plan['state']='verified_held';publish(ledger,plan,replace=True)
    return plan

def release(path):
    ledger=Path(path).resolve().parent/'node_submission.json';plan=json.loads(ledger.read_text())
    require(plan['script'] == str(Path(path).resolve()) and not Path(path).is_symlink(), 'release script differs from ledger')
    require(plan['state']=='verified_held','not verified held')
    verify(plan,True)
    plan['state']='release_intent';publish(ledger,plan,replace=True)
    command(['scontrol','release',plan['job_id']])
    plan['verified_fields']=verify(plan,False);plan['state']='released';plan['released_epoch']=time.time()
    publish(ledger,plan,replace=True)
    return plan

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['submit','release']);p.add_argument('--script',required=True)
    p.add_argument('--family',choices=CONTRACTS);p.add_argument('--name');a=p.parse_args()
    result=submit(a.script,a.family,a.name) if a.mode=='submit' else release(a.script)
    print(json.dumps({k:v for k,v in result.items() if k!='verified_fields'},sort_keys=True))
if __name__=='__main__':main()
