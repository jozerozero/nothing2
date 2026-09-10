import argparse,csv,json,multiprocessing,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import gpu_shard_group_scheduler_stage1 as s
import contract

JOBS={"999001":dict(contract.SPECS["gtqos"],evaluation_job=999001),"999002":dict(contract.SPECS["stqos"],evaluation_job=999002)}
def namespace(root,steps):
    return argparse.Namespace(checkpoint_root=root/"old",resume_checkpoint_root=root/"new",output_root=root/"out",
        claims_root=root/"out/.claims-v1",lock_path=root/"out/.claim.lock",steps=steps,checkpoint_stable_sec=60,allowed_jobs=JOBS)
def worker_claim(root,index,q):
    qos="gtqos" if index%2==0 else "stqos";job="999001" if qos=="gtqos" else "999002"
    a=namespace(Path(root),list(range(5150,6551,50)))
    with patch.dict(os.environ,{"SLURM_JOB_ID":job,"SLURM_JOB_QOS":qos,"SLURM_JOB_NAME":JOBS[job]["job_name"]}),patch.object(s,"registered_jobs",return_value=JOBS),patch.object(s,"completed_step",return_value=False),patch.object(s,"checkpoint_is_stable",return_value=True):
        q.put(s.claim_checkpoint(a,index,0))

class Tests(unittest.TestCase):
    def test_boundaries(self):
        a=namespace(Path("/temp"),[])
        for step in (5150,13000):self.assertEqual(s.checkpoint_for_step(a,step),(a.checkpoint_root/f"step-{step}.ckpt",177623))
        for step in (13050,25000):self.assertEqual(s.checkpoint_for_step(a,step),(a.resume_checkpoint_root/f"step-{step}.ckpt",180825))
        for step in (5100,13025,1525,25050):
            with self.assertRaises(ValueError):s.checkpoint_for_step(a,step)
        self.assertEqual(len(range(5150,13001,50)),158)
        self.assertEqual(len(range(13050,25001,50)),240)
    def test_concurrent_cross_qos_claims(self):
        with tempfile.TemporaryDirectory() as td:
            ctx=multiprocessing.get_context("fork");q=ctx.Queue()
            procs=[ctx.Process(target=worker_claim,args=(td,i,q)) for i in range(12)]
            for p in procs:p.start()
            claims=[q.get(timeout=15) for _ in procs]
            for p in procs:p.join(15);self.assertEqual(p.exitcode,0)
            self.assertEqual(len({c["step"] for c in claims}),12)
            self.assertEqual({c["job_id"] for c in claims},set(JOBS))
            self.assertTrue(all(c["loop_passes"]==4 and c["training_job"]==177623 for c in claims))
            self.assertFalse((Path(td)/"old").exists())
    def fixture(self,root):
        a=namespace(root,[5150])
        p=a.output_root/"step-5150/talent_detailed.txt";p.parent.mkdir(parents=True)
        p.write_text("dataset\taccuracy\n"+"".join(f"d{i}\t0.5\n" for i in range(178)))
        a.claims_root.mkdir()
        cp=a.claims_root/"step-5150.json"
        claim=dict(checkpoint=str((a.checkpoint_root/"step-5150.ckpt").resolve()),training_job=177623,loop_passes=4,job_id="999002",qos="stqos",job_name=JOBS["999002"]["job_name"])
        cp.write_text(json.dumps(claim))
        rp=p.parent/"gpu_shard_receipt.json"
        rc=dict(dataset_count=178,unique_dataset_count=178,explicit_fp32=True,clf_use_amp=False,clf_use_fa3=False,shard_count=4,model_tag="step-5150",stable_scan_sec=12)
        rp.write_text(json.dumps(rc))
        for path in (p,rp,cp):os.utime(path,(time.time()-20,time.time()-20))
        return a,p,rp,cp
    def test_other_qos_completed_accepted(self):
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{"SLURM_JOB_ID":"999001"}):
            a,*_=self.fixture(Path(td));self.assertTrue(s.completed_step(a,5150))
    def test_unknown_producer_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            a,p,rp,cp=self.fixture(Path(td));c=json.loads(cp.read_text());c["job_id"]="181417";cp.write_text(json.dumps(c))
            with self.assertRaises(AssertionError):s.completed_step(a,5150)
    def test_partial_merge_waits(self):
        with tempfile.TemporaryDirectory() as td:
            a,p,rp,cp=self.fixture(Path(td));rp.unlink();self.assertFalse(s.completed_step(a,5150))
    def test_fresh_merge_waits(self):
        with tempfile.TemporaryDirectory() as td:
            a,p,rp,cp=self.fixture(Path(td));os.utime(rp,None);self.assertFalse(s.completed_step(a,5150))
    def test_nan_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            a,p,rp,cp=self.fixture(Path(td));p.write_text(p.read_text().replace("0.5","nan",1));self.assertFalse(s.strict_panel(p))
    def test_completed_skipped_and_stale_claim_not_reclaimed(self):
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{"SLURM_JOB_ID":"999001","SLURM_JOB_QOS":"gtqos","SLURM_JOB_NAME":JOBS["999001"]["job_name"]}),patch.object(s,"registered_jobs",return_value=JOBS),patch.object(s,"completed_step",side_effect=lambda a,step:step==5150),patch.object(s,"checkpoint_is_stable",return_value=True):
            a=namespace(Path(td),[5150,5200,5250]);a.claims_root.mkdir(parents=True)
            stale=a.claims_root/"step-5200.json";stale.write_text('{"old":"claim"}')
            c=s.claim_checkpoint(a,0,0);self.assertEqual(c["step"],5250)
            self.assertEqual(stale.read_text(),'{"old":"claim"}')
    def test_registry_requires_both_jobs(self):
        with tempfile.TemporaryDirectory() as td:
            receipt=Path(td)/"state.json"
            with patch.object(contract,"RECEIPT",receipt):
                receipt.write_text(json.dumps({"registration_complete":True,"jobs":{"gtqos":JOBS["999001"],"stqos":JOBS["999002"]}}))
                self.assertEqual(set(contract.registered_jobs()),set(JOBS))
                receipt.write_text(json.dumps({"registration_complete":False,"jobs":{}}))
                with self.assertRaises(AssertionError):contract.registered_jobs()
    def test_slurm_resources_and_independent_allocation_locks(self):
        root=Path(__file__).parent
        for kind,tasks,nodes,days,qos,account in (("gt",8,2,3,"gtqos","faculty-acc"),("st",4,1,1,"stqos","test-acc")):
            text=(root/f"slurm_{kind}.sh").read_text()
            for token in (f"--qos={qos}",f"--account={account}",f"--nodes={nodes}",f"--ntasks={tasks}","--ntasks-per-node=4","--gpus-per-task=1","--cpus-per-task=4","--mem=120G",f"--time={days}-00:00:00","--nice=0","--no-requeue","--gpu-bind=single:1"):
                self.assertIn(token,text)
            self.assertIn('.evaluator-${SLURM_JOB_NAME}.lock',text)
            self.assertNotIn(".fp32-online-worksteal.lock",text)
        worker=(root/"run_group_worker.sh").read_text()
        self.assertIn("torch.cuda.device_count()",worker)
        self.assertIn("ROCR_VISIBLE_DEVICES",worker)
        self.assertIn('--group-count "${GROUP_COUNT}"',worker)
        self.assertIn('G5_LOOP_PASSES"] == "4"',worker)

if __name__=="__main__":unittest.main()
