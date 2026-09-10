"""Validate both lineages, then submit exactly two authorized jobs and release them."""
import argparse, csv, hashlib, json, math, os, subprocess, time
from pathlib import Path
from contract import ROOT,STAGE,OLD,NEW,OUT,LEGACY,BASE,LOG,RECEIPT,SPECS

def run(args):
    p=subprocess.run(args,check=True,text=True,capture_output=True)
    return p.stdout.strip()

def atomic(path,data):
    tmp=path.with_name(path.name+f".tmp-{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(data,f,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def signature(path):
    s=path.stat()
    return [s.st_size,s.st_mtime_ns]

def panel(path):
    with path.open(newline="") as f:rows=list(csv.DictReader(f,delimiter="\t"))
    values={r["dataset"]:float(r["accuracy"]) for r in rows}
    assert len(rows)==len(values)==178 and all(math.isfinite(v) and 0<=v<=1 for v in values.values()),str(path)
    return values

def validate():
    assert Path(__file__).resolve().parent==STAGE
    for line in (STAGE/"source.sha256").read_text().splitlines():
        digest,name=line.split("  ",1)
        assert hashlib.sha256((STAGE/name).read_bytes()).hexdigest()==digest,name
    for p in STAGE.glob("*.py"):compile(p.read_text(),str(p),"exec")
    for p in STAGE.glob("*.sh"):run(["bash","-n",str(p)])
    record=json.loads((NEW/"resume_record.json").read_text())
    assert record["old_job"]==177623 and record["step"]==13000
    assert record["source_checkpoint"]==str(OLD/"step-13000.ckpt")
    assert record["environment"]["G5_LOOP_PASSES"]=="4"
    assert record["all_checkpoint_states_verified_equal"] is True
    assert (OLD/"step-5150.ckpt").stat().st_size>100_000_000
    assert (OLD/"step-13000.ckpt").stat().st_size>100_000_000
    assert (NEW/"step-13050.ckpt").stat().st_size>100_000_000
    model=ROOT/"stage/e4_g5_support_condition_alpha_loops_base_20260907_v1/source"
    assert (model/"src").is_dir()
    assert OUT != LEGACY and not OUT.is_symlink()
    active=run(["squeue","--me","-h","-o","%A|%j|%T|%q"])
    conflicts=[line for line in active.splitlines() if line.split("|")[1] in
        ("e4g5sc4on20gt2","e4g5sc4on12gt1",*[s["job_name"] for s in SPECS.values()])]
    assert not conflicts,conflicts
    terminal=run(["sacct","-X","-j","181417,180467,178889","-n","-P","--format=JobID,State%30"])
    states={x.split("|")[0]:x.split("|")[1] for x in terminal.splitlines()}
    assert all(states[x].startswith("CANCELLED") for x in ("181417","180467","178889")),states
    assert "JobState=RUNNING " in run(["scontrol","show","job","180825","-o"])
    frozen=[];signatures={}
    for p in LEGACY.glob("step-*/talent_detailed.txt"):
        step=int(p.parent.name.split("-")[1])
        rp=p.parent/"gpu_shard_receipt.json";cp=LEGACY/".claims-v1"/f"step-{step}.json"
        bp=BASE/f"step-{step}"/"talent_detailed.txt"
        paths=[p,rp,cp,bp]
        before={str(x):signature(x) for x in paths}
        assert all(time.time()-x.stat().st_mtime>=12 for x in paths)
        candidate,baseline=panel(p),panel(bp)
        assert set(candidate)==set(baseline)
        receipt=json.loads(rp.read_text());claim=json.loads(cp.read_text())
        assert receipt["dataset_count"]==receipt["unique_dataset_count"]==178
        assert receipt["shard_count"]==4 and receipt["model_tag"]==f"step-{step}"
        assert receipt["explicit_fp32"] is True and receipt["clf_use_amp"] is False and receipt["clf_use_fa3"] is False
        assert receipt["stable_scan_sec"]>=12
        assert claim["step"]==step and int(claim["job_id"]) in (178889,180467)
        assert claim["checkpoint"]==str(OLD/f"step-{step}.ckpt")
        assert math.isfinite(receipt["average_accuracy"]) and abs(sum(candidate.values())/178-receipt["average_accuracy"])<1e-12
        assert before=={str(x):signature(x) for x in paths}
        signatures.update(before);frozen.append(step)
    assert sorted(frozen)==list(range(50,5101,50)),"legacy results changed; rescan"
    assert not (OUT/".claims-v1").exists() and not list(OUT.glob("step-*")),"new namespace contains work"
    assert set(panel(BASE/"step-5150"/"talent_detailed.txt"))==set(panel(LEGACY/"step-5100"/"talent_detailed.txt"))
    return dict(terminal_jobs=states,frozen_content_validated_panels=102,frozen_signatures=signatures,
        old_unvalidated_panels=158,resume_from_step=13050,new_target_panels=398,start_step=5150,
        training_job=180825,old_training_job=177623,checkpoint_roots=[str(OLD),str(NEW)],output_root=str(OUT),
        checkpoint_routing={"5150:50:13000":str(OLD),"13050:50:25000":str(NEW)},
        legacy_claims_untouched=True,protocol={"dataset_count":178,"loop_passes":4,"explicit_fp32":True,
        "amp":False,"fa3":False,"n_estimators":32,"baseline_training":142620,"baseline_evaluation":144859})

def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["prepare","submit"])
    args=p.parse_args()
    assert not RECEIPT.exists(),"already submitted; inspect receipt"
    assert not (STAGE/"submission_intent.json").exists(),"uncertain prior attempt; inspect queue before retry"
    audit=validate()
    if args.mode=="prepare":
        LOG.mkdir(parents=True,exist_ok=True)
        tests={}
        for qos,spec in SPECS.items():
            test=subprocess.run(["sbatch","--test-only",str(STAGE/spec["slurm_file"])],text=True,capture_output=True)
            assert test.returncode==0,test.stdout+test.stderr
            tests[qos]=test.stdout+test.stderr
        audit.update(prepared_epoch=time.time(),test_only=tests,stage=str(STAGE))
        atomic(STAGE/"prepare_audit.json",audit)
        print(json.dumps({k:v for k,v in audit.items() if k!="frozen_signatures"}),flush=True)
        return
    prepared=json.loads((STAGE/"prepare_audit.json").read_text())
    elapsed=time.time()-prepared["prepared_epoch"]
    assert 12<=elapsed<3600,elapsed
    assert audit["frozen_signatures"]==prepared["frozen_signatures"],"legacy signatures changed since prepare"
    audit.pop("frozen_signatures")
    with (STAGE/"submission_intent.json").open("x") as f:
        json.dump({"created_epoch":time.time(),"authorized_cancelled_job":181417,"authorized_qos_gpus":{"gtqos":8,"stqos":4}},f)
    audit.update(stage=str(STAGE),jobs={},registration_complete=False,created_epoch=time.time())
    atomic(RECEIPT,audit)
    # Both jobs stay held until their identities have been atomically registered.
    for qos,spec in SPECS.items():
        r=subprocess.run(["sbatch","--hold","--parsable",str(STAGE/spec["slurm_file"])],text=True,capture_output=True)
        if r.returncode:
            atomic(STAGE/"submission_failed.json",{"qos":qos,"stdout":r.stdout,"stderr":r.stderr,"partial_jobs":audit["jobs"]})
            raise RuntimeError(r.stdout+r.stderr)
        job=r.stdout.strip().split(";")[0]
        assert job.isdigit(),r.stdout
        audit["jobs"][qos]=dict(spec,evaluation_job=int(job),submitted_epoch=time.time(),partition="faculty",
            tasks_per_node=4,cpus_per_task=4,memory_per_node="120G",nice=0,requeue=False,dependency=None,gpu_binding="single:1")
        atomic(RECEIPT,audit)
        print("registered "+qos+" "+job,flush=True)
    audit["registration_complete"]=True
    atomic(RECEIPT,audit)
    for qos,job in audit["jobs"].items():
        run(["scontrol","release",str(job["evaluation_job"])])
        job["released_epoch"]=time.time();atomic(RECEIPT,audit)
    print(json.dumps(audit),flush=True)
    for job in audit["jobs"].values():print(run(["scontrol","show","job",str(job["evaluation_job"]),"-o"]),flush=True)

if __name__=="__main__":main()
