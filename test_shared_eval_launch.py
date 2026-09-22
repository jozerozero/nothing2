import os
import unittest
from unittest.mock import patch
import shared_eval_launch as m


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.plan={'parent_job_id':'196093','node':'auh7-1b-gpu-197'}
        self.raw=('JobId=196093 JobState=RUNNING NumNodes=1 NodeList=auh7-1b-gpu-197 '
                  'UserId=guangyi.chen(1000) AllocTRES=cpu=64,mem=64G,node=1,gres/gpu=8 '
                  'TimeLimit=3-00:00:00 RunTime=2-06:00:00')
    def test_parent(self):
        self.assertEqual(m.check_parent(self.raw,self.plan)['JobId'],'196093')
    def test_foreign_parent(self):
        for old,new in [('JobId=196093','JobId=196092'),('guangyi.chen','different'),
                        ('JobState=RUNNING','JobState=PENDING'),('NumNodes=1','NumNodes=8'),
                        ('mem=64G','mem=32G'),('gres/gpu=8','gres/gpu=4'),
                        ('RunTime=2-06:00:00','RunTime=2-23:00:00')]:
            with self.subTest(old=old), self.assertRaises((RuntimeError,ValueError)):
                m.check_parent(self.raw.replace(old,new),self.plan)
    def test_environment_not_spoofed(self):
        with patch.dict(os.environ,{'SLURM_JOB_ID':'bad','SRUN_CPUS_PER_TASK':'99',
                                    'ROCR_VISIBLE_DEVICES':'bad','PYTHONPATH':'bad',
                                    'SBATCH_NODES':'8'},clear=True):
            env=m.clean_environment()
        self.assertFalse(any(k.startswith(('SLURM_','SRUN_','SBATCH_')) for k in env))
        self.assertNotIn('ROCR_VISIBLE_DEVICES',env)
        self.assertNotIn('PYTHONPATH',env)
        self.assertEqual(env['PYTHONNOUSERSITE'],'1')


if __name__=='__main__': unittest.main()
