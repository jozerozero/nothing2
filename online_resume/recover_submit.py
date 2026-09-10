"""Validate terminal-owner evidence, preserve completed data, submit once held."""
import argparse, csv, fcntl, hashlib, json, os, subprocess, time
from pathlib import Path
from types import SimpleNamespace
import gpu_shard_group_scheduler_stage1 as scheduler
from recovery import ROOT, STAGE, OUT, OLD_WORK, NAME, POLICY, TAG, validate_shard, signatures

CK = ROOT/'checkpoints/e4_g5_support_condition_alpha_loops_20260907_v1/g36-g5scalpha-loop3-histe4-25k-v1'
OLD = CK/'e4g5sc3lr1-178786'
NEW = CK/'e4g5sc3lr2-181407'
def run(args):
    p = subprocess.run(args, text=True, capture_output=True, check=True)
    return p.stdout.strip()
def atomic(p, value):
    scheduler.atomic_json(p, value)
def validate():
    assert Path(__file__).resolve().parent == STAGE
    for line in (STAGE/'source.sha256').read_text().splitlines():
        digest,name = line.split('  ',1)
        assert hashlib.sha256((STAGE/name).read_bytes()).hexdigest() == digest, name
    for p in STAGE.glob('*.py'):compile(p.read_text(),str(p),'exec')
    for p in STAGE.glob('*.sh'):run(['bash','-n',str(p)])
    queue = run(['squeue','--me','-h','-o','%A|%j|%T'])
    conflicts = [line for line in queue.splitlines() if line.split('|')[1] in [NAME,'e4g5sc3r24gt1','e4g5sc3on24gt2']]
    assert not conflicts, conflicts
    accounting = run(['sacct','-j','181580','-n','-P','--format=JobID,State%30,End'])
    rows = [r.split('|') for r in accounting.splitlines()]
    assert any(r[0]=='181580' and r[1]=='FAILED' for r in rows)
    assert all(r[1].startswith(('FAILED','CANCELLED','COMPLETED','TIMEOUT')) for r in rows), rows
    assert 'JobState=RUNNING ' in run(['scontrol','show','job','181407','-o'])
    resume = json.loads((NEW/'resume_record.json').read_text())
    assert resume['old_job']==178786 and resume['step']==14500
    assert resume['source_checkpoint']==str(OLD/'step-14500.ckpt')
    assert resume['environment']['G5_LOOP_PASSES']=='3' and resume['all_checkpoint_states_verified_equal'] is True
    policy=json.loads(POLICY.read_text())
    a=SimpleNamespace(checkpoint_root=OLD,resume_checkpoint_root=NEW,output_root=OUT,claims_root=OUT/'.claims-v1',retained_steps=[8950],expected_dataset_names={n for s in policy['shards'] for n in s})
    os.environ['SLURM_JOB_ID']='preflight'
    claims={int(p.stem.split('-')[1]):json.loads(p.read_text()) for p in a.claims_root.glob('step-*.json')}
    assert sorted(claims)==list(range(8850,9101,50)), claims
    for step,claim in claims.items():
        cp,training=scheduler.checkpoint_for_step(a,step)
        assert str(claim['job_id'])=='181580' and claim['checkpoint']==str(cp)
        assert claim['training_job']==training and claim['loop_passes']==3 and claim['step']==step
    retained=[s for s in claims if scheduler.completed_step(a,s)]
    assert retained==[8950],retained
    assert sorted(p.name for p in OUT.glob('step-*'))==['step-8950','step-9100']
    assert sorted(p.name for p in (OUT/'step-9100').iterdir())==['talent_detailed.txt']
    reusable={}
    for p in sorted(OLD_WORK.glob('step-*/shard-*/shard_result.json')):
        step=int(p.parent.parent.name.split('-')[1]);i=int(p.parent.name.split('-')[1])
        assert step in claims
        cp,_=scheduler.checkpoint_for_step(a,step)
        reusable[f'{step}/{i}']={'path':str(p.parent),'signatures':validate_shard(p.parent,cp,step,i,policy)}
    assert len(reusable)==20, len(reusable)
    files=list(a.claims_root.glob('step-*.json'))+list((OUT/'step-8950').iterdir())+list((OUT/'step-9100').iterdir())
    for value in reusable.values():files.extend(Path(p) for p in value['signatures'])
    sig=signatures(set(files))
    assert all(time.time()-Path(p).stat().st_mtime>=12 for p in sig)
    return dict(failed_job=181580,training_job=181407,old_training_job=178786,retained_steps=retained,
                reclaim_steps=sorted(set(claims)-set(retained)),reusable_shards=reusable,
                signatures=sig,terminal_accounting=accounting,observed_epoch=time.time(),output_root=str(OUT),
                checkpoint_roots=[str(OLD),str(NEW)],stage=str(STAGE),job_name=NAME,
                resources=dict(qos='gtqos',partition='faculty',account='faculty-acc',nodes=4,gpus=24,tasks_per_node=6,cpus_per_task=4,memory_per_node='120G',group_count=6,nice=0,time_limit='3-00:00:00'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','submit']);mode=parser.parse_args().mode
    receipt=STAGE/'submission_state.json'
    assert not receipt.exists() and not (STAGE/'submission_intent.json').exists(), 'already attempted: inspect receipt/intent, never duplicate'
    audit=validate()
    if mode=='prepare':
        (ROOT/'logs'/TAG).mkdir(parents=True,exist_ok=True)
        p=subprocess.run(['sbatch','--test-only',str(STAGE/'slurm.sh')],text=True,capture_output=True)
        assert p.returncode==0,p.stdout+p.stderr
        audit['test_only']=p.stdout+p.stderr
        atomic(STAGE/'prepare_audit.json',audit)
        print(json.dumps(audit),flush=True)
        return
    prepared=json.loads((STAGE/'prepare_audit.json').read_text())
    assert 12<=time.time()-prepared['observed_epoch']<3600
    assert audit['signatures']==prepared['signatures'], 'source changed between scans'
    with (OUT/'.fp32-online-worksteal.lock').open('a+') as allocation_lock, (OUT/'.claim.lock').open('a+') as claim_lock:
        fcntl.flock(allocation_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(claim_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert signatures(Path(p) for p in audit['signatures'])==audit['signatures']
        with (STAGE/'submission_intent.json').open('x') as f:json.dump({'job_name':NAME,'failed_job':181580,'epoch':time.time()},f)
        archive=OUT/'.recovery-181580-20260911-r2';archive.mkdir()
        (archive/'claims').mkdir()
        for step in audit['reclaim_steps']:
            os.replace(OUT/'.claims-v1'/f'step-{step}.json',archive/'claims'/f'step-{step}.json')
        os.replace(OUT/'step-9100',archive/'step-9100-uncommitted')
        audit['archive']=str(archive)
        atomic(STAGE/'recovery_audit.json',audit)
        p=subprocess.run(['sbatch','--parsable','--hold',str(STAGE/'slurm.sh')],text=True,capture_output=True)
        if p.returncode:
            atomic(STAGE/'submission_failed.json',{'stdout':p.stdout,'stderr':p.stderr,'rc':p.returncode})
            raise RuntimeError(p.stdout+p.stderr)
        job=p.stdout.strip().split(';')[0];assert job.isdigit(),p.stdout
        audit.update(evaluation_job=int(job),submitted_epoch=time.time(),registration_complete=True)
        atomic(receipt,audit)
        print('__L3R_SUBMITTED__'+job,flush=True)
    run(['scontrol','release',job])
    audit['released_epoch']=time.time();atomic(receipt,audit)
    print(run(['scontrol','show','job',job,'-o']),flush=True)
    print('__L3R_RECEIPT__'+json.dumps(audit),flush=True)

if __name__=='__main__':main()
