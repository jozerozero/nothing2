import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import tabfm_prepare as prepare
import tabfm_submit as submit

class TestFreezeSubmit(unittest.TestCase):
    def test_stable_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'source.py';p.write_text('print(1)\n')
            record=prepare.identity(p)
            self.assertEqual(record,prepare.identity(p))
            self.assertEqual(submit.verify_file(record),p.resolve())
            p.write_text('print(2)\n')
            with self.assertRaises(RuntimeError):submit.verify_file(record)

    def test_manifest_digest(self):
        value={'x':1};value['manifest_id']=prepare.object_digest(value)
        prepare.verify_manifest(value)
        value['x']=2
        with self.assertRaises(RuntimeError):prepare.verify_manifest(value)

    def test_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'batch.sh';script.write_text('#!/bin/bash\nsrun --gpu-bind=single:1\n')
            f={'JobId':'123','JobName':'tabfm681','Partition':'faculty','Account':'faculty-acc',
              'QOS':'bgqos','NumTasks':'4','NumCPUs':'16','CPUs/Task':'4','MinMemoryNode':'256G',
              'Nice':'0','Requeue':'0','Dependency':'(null)','Command':str(script),'WorkDir':str(script.parent),
              'StdOut':str(submit.OUT/'logs/job-123.out'),'StdErr':str(submit.OUT/'logs/job-123.err'),
              'UserId':'guangyi.chen(2012)','NumNodes':'1','TimeLimit':'1-00:00:00',
              'ReqTRES':'cpu=16,mem=256G,node=1,gres/gpu=4','TresPerTask':'cpu=4,gres/gpu=1',
              'NtasksPerN:B:S:C':'4:0:*:*','JobState':'PENDING','Reason':'JobHeldUser',
              'NodeList':'(null)','SchedNodeList':'(null)','ExcNodeList':'auh7-1b-gpu-193'}
            raw=lambda:' '.join(k+'='+v for k,v in f.items())
            submit.verify('123',script,raw(),True)
            f['NumTasks']='8'
            with self.assertRaises(RuntimeError):submit.verify('123',script,raw(),True)

if __name__=='__main__':unittest.main()
