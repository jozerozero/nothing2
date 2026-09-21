"""One initially-held TabFM job; immutable journal prevents duplicate submissions."""
import fcntl
import json
import re
import subprocess
import time
from pathlib import Path
from eval_one import publish_new, require, verify_file
from tabfm_prepare import OUT, STAGE, verify_manifest, identity
from classification32_submit import EXCLUDED_NODES, EXCLUDE

def command(argv):
    return subprocess.check_output(argv, text=True).strip()

def verify(job, script, raw, held):
    f = dict(re.findall(r'([^\s=]+)=([^\s]+)',raw))
    expected={'JobId':job,'JobName':'tabfm681','Partition':'faculty','Account':'faculty-acc',
      'QOS':'bgqos','NumTasks':'4','NumCPUs':'16','CPUs/Task':'4','MinMemoryNode':'256G',
      'Nice':'0','Requeue':'0','Dependency':'(null)', 'Command':str(script),
      'WorkDir':str(script.parent),'StdOut':str(OUT/'logs'/f'job-{job}.out'),
      'StdErr':str(OUT/'logs'/f'job-{job}.err')}
    for k,v in expected.items():
        require(f.get(k)==v,f'Submission contract mismatch {k}={f.get(k)} expected {v}')
    require(f.get('UserId','').startswith('guangyi.chen('),'Wrong owner')
    require(f.get('NumNodes') in ('1','1-1') and f.get('TimeLimit') in ('1-00:00:00','24:00:00'),
            'Must reserve one node24h')
    require('gres/gpu=4' in f.get('ReqTRES','').split(','),'Must reserve four GPUs')
    require(f.get('TresPerTask')=='cpu=4,gres/gpu=1' and f.get('NtasksPerN:B:S:C','').split(':')[0]=='4',
            'Wrong one-GPU/fourCPU perrank binding')
    require('single:1' in script.read_text(),'Single-GPU srun binding missing')
    if held:
        require(f.get('JobState')=='PENDING' and f.get('Reason')=='JobHeldUser','Job not held for contract check')
    for key in ('NodeList','SchedNodeList'):
        if f.get(key) not in (None,'(null)','None'):
            nodes=set(command(['scontrol','show','hostnames',f[key]]).splitlines())
            require(not nodes & EXCLUDED_NODES,'Excluded actual/planned node')
    require(f.get('ExcNodeList') not in (None,'(null)','None'),'Excluded-node request lost')
    return f

def main():
    manpath=OUT/'manifest.json'
    man=json.loads(manpath.read_text()); verify_manifest(man)
    repo=Path(__file__).resolve().parent
    script=repo/'tabfm_default_slurm.sh'
    pinned={Path(r['path']).resolve():r for r in man['worker_sources']}
    require(script in pinned,'Slurm script not pinned')
    verify_file(pinned[script])
    (OUT/'logs').mkdir(exist_ok=True)
    with (OUT/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=OUT/'submission_attempt.json'
        require(not journal.exists(),'An attempt exists; inspect scheduler, do not duplicate')
        active=command(['squeue','--me','-h','-o','%i|%j|%T'])
        require(not any(x.split('|')[1]=='tabfm681' for x in active.splitlines()),'Existing TabFM campaign active')
        args=['sbatch','--hold','--parsable','--job-name=tabfm681','--partition=faculty',
          '--account=faculty-acc','--qos=bgqos','--nodes=1','--ntasks=4','--ntasks-per-node=4',
          '--gpus-per-task=1','--cpus-per-task=4','--mem=256G','--time=1-00:00:00',
          '--no-requeue','--nice=0','--exclude='+EXCLUDE,'--chdir='+str(repo),
          '--output='+str(OUT/'logs/job-%j.out'),'--error='+str(OUT/'logs/job-%j.err'),
          '--export=ALL,PYTHONHASHSEED=0',str(script),str(manpath)]
        publish_new(journal,{'epoch':time.time(),'manifest_id':man['manifest_id'],'command':args,
                            'script':identity(script),'no_automatic_retry':True})
        job=command(args).split(';')[0];require(job.isdigit(),'Unrecognized sbatch response; do not retry')
        publish_new(OUT/'submitted_job.json',{'job_id':job,'epoch':time.time(),'manifest_id':man['manifest_id']})
        raw=command(['scontrol','show','job',job,'-o'])
        checked=verify(job,script,raw,held=True)
        publish_new(OUT/'verified_held_job.json',{'job_id':job,'raw':raw,'fields':checked,'epoch':time.time()})
        command(['scontrol','release',job])
        raw=command(['scontrol','show','job',job,'-o']);checked=verify(job,script,raw,held=False)
        publish_new(OUT/'released_job.json',{'job_id':job,'fields':checked,'raw':raw,'epoch':time.time()})
        print(json.dumps({'job_id':job,'state':checked.get('JobState'),'reason':checked.get('Reason'),
          'nodes':checked.get('NodeList'),'gpu_count':4,'task_scope':[457,224],
          'hierarchical_tasks':len(man['hierarchical_classification_indices']),
          'output_root':str(OUT),'preflight_required_before_formal_evaluation':True}),flush=True)

if __name__=='__main__':main()
