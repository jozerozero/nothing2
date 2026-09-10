"""Deploy-time checks and exactly-once submission for the authorized evaluator."""
import argparse, csv, hashlib, json, math, os, subprocess, time
from pathlib import Path

ROOT=Path("/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1")
STAGE=Path("/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/e4_g5sc_loop3_resume181407_eval_fp32_online_24gpu_gt_step50_20260910_v1/online_resume")
OLD=Path("/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/checkpoints/e4_g5_support_condition_alpha_loops_20260907_v1/g36-g5scalpha-loop3-histe4-25k-v1/e4g5sc3lr1-178786")
NEW=Path("/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/checkpoints/e4_g5_support_condition_alpha_loops_20260907_v1/g36-g5scalpha-loop3-histe4-25k-v1/e4g5sc3lr2-181407")
OUT=Path("/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/evaluation/e4_g5sc_loop3_resume181407_fp32_online_24gpu_gt_step50_20260910_v1/E4_G5SC_LOOP3/lineage-178786-181407")
LEGACY=ROOT/'evaluation/e4_g5sc_loop3_178786_fp32_online_12gpu_gt_step50_20260908_v1/E4_G5SC_LOOP3/train-178786'
LOG=ROOT/'logs/e4_g5sc_loop3_resume181407_eval_fp32_online_24gpu_gt_step50_20260910_v1'
NAME='e4g5sc3r24gt1'
def run(args):
    p=subprocess.run(args,check=True,text=True,capture_output=True)
    return p.stdout.strip()
def atomic(path,x):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,indent=2)+'\n')
    os.replace(tmp,path)
def validate():
    assert Path(__file__).resolve().parent==STAGE
    for line in (STAGE/'source.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert hashlib.sha256((STAGE/name).read_bytes()).hexdigest()==digest,name
    for file in STAGE.glob('*.py'):compile(file.read_text(),str(file),'exec')
    for file in STAGE.glob('*.sh'):run(['bash','-n',str(file)])
    record=json.loads((NEW/'resume_record.json').read_text())
    assert record['old_job']==178786 and record['step']==14500
    assert record['source_checkpoint']==str(OLD/'step-14500.ckpt')
    assert record['environment']['G5_LOOP_PASSES']=='3'
    assert record['all_checkpoint_states_verified_equal'] is True
    assert (OLD/'step-8850.ckpt').stat().st_size>100_000_000
    assert (NEW/'step-14550.ckpt').stat().st_size>100_000_000
    model_root=ROOT/'stage/e4_g5_support_condition_alpha_loops_base_20260907_v1/source'
    assert (model_root/'src').is_dir()
    assert OUT != LEGACY and not OUT.is_symlink()
    active=run(['squeue','--me','-h','-o','%A|%j|%T|%q'])
    conflicts=[x for x in active.splitlines() if x.split('|')[1] in ('e4g5sc3on24gt2',NAME,'e4g5sc3on12gt1')]
    assert not conflicts,conflicts
    terminal=run(['sacct','-X','-j','181416,180468','-n','-P','--format=JobID,State%30'])
    states={x.split('|')[0]:x.split('|')[1] for x in terminal.splitlines()}
    assert states['181416'].startswith('CANCELLED') and states['180468'].startswith('CANCELLED'),states
    assert 'JobState=RUNNING ' in run(['scontrol','show','job','181407','-o'])
    complete=[]
    for p in LEGACY.glob('step-*/talent_detailed.txt'):
        step=int(p.parent.name.split('-')[1])
        with p.open(newline='') as f: rows=list(csv.DictReader(f,delimiter='\t'))
        vals=[float(r['accuracy']) for r in rows]
        assert len(rows)==len({r['dataset'] for r in rows})==178 and all(math.isfinite(v) and 0<=v<=1 for v in vals)
        rc=json.loads((p.parent/'gpu_shard_receipt.json').read_text())
        assert rc['explicit_fp32'] is True and rc['clf_use_amp'] is False and rc['clf_use_fa3'] is False
        assert rc['dataset_count']==rc['unique_dataset_count']==178 and rc['stable_scan_sec']>=12
        assert time.time()-p.stat().st_mtime>=12
        complete.append(step)
    assert sorted(complete)==list(range(450,8801,50)), 'legacy results changed; rescan before submit'
    assert not (OUT/'.claims-v1').exists() and not list(OUT.glob('step-*')), 'new output already has work'
    return {'terminal_jobs':states,'frozen_panels':len(complete),'old_unvalidated_steps':114,'resume_from':14550,'new_target_panels':324,'training_job':181407,'old_training_job':178786,'checkpoint_roots':[str(OLD),str(NEW)],'output_root':str(OUT)}

mode=argparse.ArgumentParser()
mode.add_argument('mode',choices=['prepare','submit'])
args=mode.parse_args()
receipt=STAGE/'submission_state.json'
assert not receipt.exists(), 'already submitted; inspect receipt instead of resubmitting'
audit=validate()
if args.mode=='prepare':
    LOG.mkdir(parents=True,exist_ok=True)
    test=subprocess.run(['sbatch','--test-only',str(STAGE/'slurm.sh')],capture_output=True,text=True)
    assert test.returncode==0,test.stdout+test.stderr
    audit.update(test_only=test.stdout+test.stderr,prepared_epoch=time.time(),stage=str(STAGE))
    atomic(STAGE/'prepare_audit.json',audit)
    print(json.dumps(audit))
else:
    assert (STAGE/'prepare_audit.json').is_file()
    assert time.time()-json.loads((STAGE/'prepare_audit.json').read_text())['prepared_epoch']<3600
    assert LOG.is_dir()
    # Exclusive intent file protects against accidental retry after an uncertain response.
    intent=STAGE/'submission_intent.json'
    with intent.open('x') as f:
        json.dump({'created_epoch':time.time(),'job_name':NAME,'authorized_cancelled_job':181416},f)
    p=subprocess.run(['sbatch','--parsable',str(STAGE/'slurm.sh')],capture_output=True,text=True)
    if p.returncode:
        atomic(STAGE/'submission_failed.json',{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
        raise RuntimeError(p.stdout+p.stderr)
    job=p.stdout.strip().split(';')[0]
    assert job.isdigit(),p.stdout
    audit.update(evaluation_job=int(job),job_name=NAME,submitted_epoch=time.time(),stage=str(STAGE),slurm_file='slurm.sh',resources={'qos':'gtqos','partition':'faculty','account':'faculty-acc','nodes':4,'gpus':24,'tasks_per_node':6,'cpus_per_task':4,'memory_per_node':'120G','group_count':6,'nice':0,'time_limit':'3-00:00:00'})
    atomic(receipt,audit)
    print(json.dumps(audit),flush=True)
    print(run(['scontrol','show','job',job,'-o']))

