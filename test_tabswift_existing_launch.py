import tempfile
import unittest
import os
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch, Mock

import tabswift_existing_launch as launch


class LauncherTests(unittest.TestCase):
    def test_zombie_is_not_a_live_gpu_owner(self):
        proc=Mock(pid=999999,info={'uids':SimpleNamespace(real=os.getuid()),'cmdline':['exited']})
        proc.status.return_value='zombie'
        proc.environ.side_effect=AssertionError('Must not read zombie environment')
        fake=SimpleNamespace(process_iter=lambda _: [proc],STATUS_ZOMBIE='zombie',STATUS_DEAD='dead',
                             NoSuchProcess=ProcessLookupError)
        with patch.dict(sys.modules,{'psutil':fake}):
            self.assertEqual(launch.owners({'gpu':{'uuid':'x'}}),[])

    def test_no_implicit_video_release(self):
        self.assertEqual(launch.release_idle_video({}, {}, Path('/unused')),0)

    def test_reject_wrong_video_target(self):
        with self.assertRaises(RuntimeError):
            launch.release_idle_video({'node':'auh7-1b-gpu-241','gpu':{'uuid':'88c2b6b231e4a217'}},
                {'release_idle_video':{'pid':3905554}},Path('/unused'))

    def test_typed_allocation_and_single_model_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = dict(parent_job_id='214135', node='auh7-1b-gpu-241',
                        runtime_root=tmp, worker_python='/verified/python', plan_id='test')
            with patch.object(launch.worker, 'load_plan', return_value=(plan, None)), \
                 patch.object(launch, 'check_parent', return_value=('verified', {})), \
                 patch.object(launch, 'publish_new'), \
                 patch.object(launch.subprocess, 'Popen', return_value=Mock(pid=42)) as popen:
                receipt = launch.launch(Path(tmp)/'plan.json')
            command = popen.call_args.args[0]
            self.assertIn('--gres=gpu:mi210:8', command)
            self.assertNotIn('--gpus=8', command)
            self.assertIn('--overlap', command)
            self.assertIn('--time=02:00:00', command)
            self.assertEqual(receipt['actual_model_gpus'], 1)
            self.assertFalse(receipt['new_allocation'])
            self.assertEqual(receipt['video_signals_sent'], 0)


if __name__ == '__main__':
    unittest.main()
