import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import tabswift_existing_launch as launch


class LauncherTests(unittest.TestCase):
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
