import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
import time
import multiprocessing
from unittest.mock import patch

import gpu_shard_group_scheduler_stage1 as scheduler

def parallel_claim(a, index, q):
    with patch.dict(os.environ, {'SLURM_JOB_ID':'999999'}):
        q.put(scheduler.claim_checkpoint(a,index,0))


class ResumeTests(unittest.TestCase):
    def fixture(self, root, producer='181580'):
        root=root.resolve()
        a=argparse.Namespace(checkpoint_root=root/'old',resume_checkpoint_root=root/'new',output_root=root/'out',claims_root=root/'out/.claims-v1',lock_path=root/'out/.claim.lock',retained_steps=[8950],steps=[8950],checkpoint_stable_sec=60,expected_dataset_names={f'd{i}' for i in range(178)})
        panel=a.output_root/'step-8950'/'talent_detailed.txt'
        panel.parent.mkdir(parents=True)
        panel.write_text('dataset\taccuracy\n'+''.join(f'd{i}\t0.8\n' for i in range(178)))
        cp,_=scheduler.checkpoint_for_step(a,8950)
        claim=dict(checkpoint=str(cp),training_job=178786,loop_passes=3,job_id=producer)
        scheduler.atomic_json(a.claims_root/'step-8950.json',claim)
        receipt=dict(dataset_count=178,unique_dataset_count=178,explicit_fp32=True,clf_use_amp=False,clf_use_fa3=False,shard_count=4,model_tag='step-8950',stable_scan_sec=12,n_estimators=32,outer_batch=8,n_jobs=1,kv_cache=False)
        return a,panel,receipt

    def mature(self, directory):
        for p in directory.iterdir():os.utime(p,(time.time()-60,time.time()-60))

    def test_panel_without_receipt_is_incomplete_not_fatal(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ,{'SLURM_JOB_ID':'999999'}):
            a,panel,rc=self.fixture(Path(td));self.mature(panel.parent)
            for _ in range(50):self.assertFalse(scheduler.completed_step(a,8950))
            self.assertIsNone(scheduler.claim_checkpoint(a,0,1))
            self.assertEqual(json.loads((a.claims_root/'step-8950.json').read_text())['job_id'],'181580')
            scheduler.atomic_json(panel.parent/'gpu_shard_receipt.json',rc)
            self.assertFalse(scheduler.completed_step(a,8950))
            (panel.parent/'talent_summary.txt').write_text('completed')
            self.assertFalse(scheduler.completed_step(a,8950))
            self.mature(panel.parent)
            self.assertTrue(scheduler.completed_step(a,8950))

    def test_old_producer_only_accepted_on_retained_steps(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ,{'SLURM_JOB_ID':'999999'}):
            a,panel,rc=self.fixture(Path(td))
            scheduler.atomic_json(panel.parent/'gpu_shard_receipt.json',rc)
            (panel.parent/'talent_summary.txt').write_text('completed');self.mature(panel.parent)
            self.assertTrue(scheduler.completed_step(a,8950))
            a.retained_steps=[]
            with self.assertRaises(AssertionError):scheduler.completed_step(a,8950)

    def test_unknown_producer_and_bad_protocol_fail_closed(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ,{'SLURM_JOB_ID':'999999'}):
            a,panel,rc=self.fixture(Path(td),producer='12345')
            scheduler.atomic_json(panel.parent/'gpu_shard_receipt.json',rc)
            (panel.parent/'talent_summary.txt').write_text('completed');self.mature(panel.parent)
            with self.assertRaises(AssertionError):scheduler.completed_step(a,8950)
            cp=a.claims_root/'step-8950.json';claim=json.loads(cp.read_text());claim['job_id']='999999';scheduler.atomic_json(cp,claim)
            rc['clf_use_amp']=True;scheduler.atomic_json(panel.parent/'gpu_shard_receipt.json',rc);self.mature(panel.parent)
            with self.assertRaises(AssertionError):scheduler.completed_step(a,8950)

    def test_exact_dataset_identity(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ,{'SLURM_JOB_ID':'999999'}):
            a,panel,rc=self.fixture(Path(td))
            scheduler.atomic_json(panel.parent/'gpu_shard_receipt.json',rc)
            (panel.parent/'talent_summary.txt').write_text('completed');self.mature(panel.parent)
            a.expected_dataset_names={'wrong'}
            with self.assertRaises(AssertionError):scheduler.completed_step(a,8950)

    def test_atomic_claims_across_eight_processes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            a=argparse.Namespace(checkpoint_root=root/'old',resume_checkpoint_root=root/'new',output_root=root/'out',claims_root=root/'out/.claims-v1',lock_path=root/'out/.claim.lock',retained_steps=[],steps=list(range(8850,9250,50)),checkpoint_stable_sec=0)
            a.checkpoint_root.mkdir()
            for s in a.steps:
                with (a.checkpoint_root/f'step-{s}.ckpt').open('wb') as f:f.truncate(100_000_001)
            ctx=multiprocessing.get_context('fork');q=ctx.Queue()
            ps=[ctx.Process(target=parallel_claim,args=(a,i,q)) for i in range(8)]
            for p in ps:p.start()
            values=[q.get(timeout=10) for _ in ps]
            for p in ps:p.join(10);self.assertEqual(p.exitcode,0)
            self.assertEqual(sorted(x['step'] for x in values),a.steps)
            self.assertEqual(len(list(a.claims_root.glob('*.json'))),8)

    def test_reusable_shard_is_validated_and_not_overwritten(self):
        import recovery
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();source=root/'work/step-8850/shard-0';source.mkdir(parents=True)
            cp=root/'ckpt/step-8850.ckpt';policy={'shards':[['a','b'],[],[],[]]}
            record=dict(checkpoint=str(cp),model_tag='step-8850',shard_index=0,dataset_count=2,explicit_fp32=True,clf_use_amp=False,clf_use_fa3=False,n_estimators=32,outer_batch=8,n_jobs=1,kv_cache=False)
            scheduler.atomic_json(source/'shard_result.json',record)
            (source/'step-8850').mkdir();panel=source/'step-8850/talent_detailed.txt';panel.write_text('dataset\taccuracy\ttime_s\na\t0.8\t1\nb\t0.9\t2\n')
            sig=recovery.validate_shard(source,cp,8850,0,policy)
            receipt={'reusable_shards':{'8850/0':dict(path=str(source),signatures=sig)}}
            target=root/'new/work/step-8850/shard-0'
            with patch.object(recovery,'OLD_WORK',root/'work'):
                recovery.install_reusable_shard(receipt,8850,0,target,cp,policy)
                self.assertTrue(target.is_symlink());self.assertEqual(target.resolve(),source)
                recovery.install_reusable_shard(receipt,8850,0,target,cp,policy)
                self.assertEqual(recovery.validate_shard(source,cp,8850,0,policy),sig)
                panel.write_text(panel.read_text().replace('0.8','nan'))
                with self.assertRaises(AssertionError):recovery.install_reusable_shard(receipt,8850,0,target,cp,policy)

    def test_receipt_disappears_between_stat_and_open(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ,{'SLURM_JOB_ID':'999999'}):
            a,panel,rc=self.fixture(Path(td))
            scheduler.atomic_json(panel.parent/'gpu_shard_receipt.json',rc)
            (panel.parent/'talent_summary.txt').write_text('completed');self.mature(panel.parent)
            original=Path.read_text
            def disappearing(path,*args,**kwargs):
                if path.name=='gpu_shard_receipt.json':raise FileNotFoundError(path)
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',disappearing):self.assertFalse(scheduler.completed_step(a,8950))

    def test_boundary(self):
        a=argparse.Namespace(checkpoint_root=Path('/old'), resume_checkpoint_root=Path('/new'))
        for step in (8850, 14500):
            self.assertEqual(scheduler.checkpoint_for_step(a,step),(Path(f'/old/step-{step}.ckpt'),178786))
        for step in (14550, 25000):
            self.assertEqual(scheduler.checkpoint_for_step(a,step),(Path(f'/new/step-{step}.ckpt'),181407))
        for step in (8800,14525,14675,25050):
            with self.assertRaises(ValueError): scheduler.checkpoint_for_step(a,step)

    def test_unique_claims_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            a=argparse.Namespace(checkpoint_root=root/'old',resume_checkpoint_root=root/'new',output_root=root/'out',claims_root=root/'out/.claims-v1',lock_path=root/'out/.claim.lock',steps=[8850,14500,14550],checkpoint_stable_sec=60)
            with patch.dict(os.environ,{'SLURM_JOB_ID':'999999'}), patch.object(scheduler,'completed_step',return_value=False), patch.object(scheduler,'checkpoint_is_stable',return_value=True):
                claims=[scheduler.claim_checkpoint(a,i,0) for i in range(3)]
                self.assertIsNone(scheduler.claim_checkpoint(a,3,0))
            self.assertEqual([x['step'] for x in claims],[8850,14500,14550])
            self.assertEqual([x['training_job'] for x in claims],[178786,178786,181407])
            self.assertTrue(all(x['loop_passes']==3 and x['job_id']=='999999' for x in claims))
            self.assertEqual(len(list(a.claims_root.glob('*.json'))),3)
            self.assertFalse((root/'old').exists())
            self.assertFalse((root/'new').exists())

    def test_completed_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            a=argparse.Namespace(checkpoint_root=root/'old',resume_checkpoint_root=root/'new',output_root=root/'out',claims_root=root/'out/.claims-v1',lock_path=root/'out/.claim.lock',steps=[8850,8900],checkpoint_stable_sec=60)
            with patch.dict(os.environ,{'SLURM_JOB_ID':'999999'}), patch.object(scheduler,'completed_step',side_effect=lambda args,s:s==8850), patch.object(scheduler,'checkpoint_is_stable',return_value=True):
                claim=scheduler.claim_checkpoint(a,0,0)
            self.assertEqual(claim['step'],8900)
            self.assertFalse((a.claims_root/'step-8850.json').exists())

    def test_slurm_contract(self):
        text=(Path(__file__).parent/'slurm.sh').read_text()
        for token in ['--qos=bgqos','--nodes=4','--ntasks=32','--ntasks-per-node=8','--gpus-per-task=1','--cpus-per-task=4','--mem=120G','--time=3-00:00:00','--nice=0','--no-requeue','--gpu-bind=single:1']:
            self.assertIn(token,text)
        worker=(Path(__file__).parent/'run_group_worker.sh').read_text()
        self.assertIn('--group-count 8',worker)
        self.assertIn('--resume-checkpoint-root',worker)
        self.assertIn('torch.cuda.device_count()',worker)
        self.assertIn('ROCR_VISIBLE_DEVICES',worker)
        self.assertNotIn("if not os.environ.get(\"CUDA_VISIBLE_DEVICES\")",worker)

if __name__=='__main__': unittest.main()
