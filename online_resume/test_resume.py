import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gpu_shard_group_scheduler_stage1 as scheduler


class ResumeTests(unittest.TestCase):
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
        for token in ['--qos=gtqos','--nodes=4','--ntasks=24','--ntasks-per-node=6','--gpus-per-task=1','--cpus-per-task=4','--mem=120G','--time=3-00:00:00','--nice=0','--no-requeue','--gpu-bind=single:1']:
            self.assertIn(token,text)
        worker=(Path(__file__).parent/'run_group_worker.sh').read_text()
        self.assertIn('--group-count 6',worker)
        self.assertIn('--resume-checkpoint-root',worker)
        self.assertIn('torch.cuda.device_count()',worker)
        self.assertIn('ROCR_VISIBLE_DEVICES',worker)
        self.assertNotIn("if not os.environ.get(\"CUDA_VISIBLE_DEVICES\")",worker)

if __name__=='__main__': unittest.main()
