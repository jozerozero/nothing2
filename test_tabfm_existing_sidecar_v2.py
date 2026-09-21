"""CPU-only v2 affinity tests; never change the test runner's real affinity."""
import hashlib
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import tabfm_existing_sidecar_v2 as v2


class CpuBindingTests(unittest.TestCase):
    def setup_masks(self, allowed=None, frames=None, actual=None):
        self.allowed = set(range(32, 96)) if allowed is None else set(allowed)
        self.selected = {36, 37, 70, 95}
        if frames is None:
            frames = [[90.] * 128, [90.] * 128]
            for number, cpu in enumerate(sorted(self.selected)):
                frames[0][cpu] = number
                frames[1][cpu] = number + 1
        self.psutil = SimpleNamespace(cpu_percent=Mock(side_effect=frames))
        self.mask = set(self.allowed)
        def set_mask(pid, mask):
            self.assertEqual(pid, 0)
            self.mask = set(mask) if actual is None else set(actual)
        self.setter = Mock(side_effect=set_mask)
        for patcher in (patch.dict(sys.modules, {"psutil": self.psutil}),
                        patch.object(os, "sched_getaffinity", side_effect=lambda pid: set(self.mask), create=True),
                        patch.object(os, "sched_setaffinity", self.setter, create=True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_two_samples_then_exact_four_cpu_affinity(self):
        self.setup_masks()
        record = v2.bind_idle_cpu_cores()
        self.assertEqual(self.mask, self.selected)
        self.assertEqual(record["actual_cpu_ids_after"], sorted(self.selected))
        self.assertEqual(record["observed_allowed_cpu_count"], 64)
        self.assertEqual(record["inherited_cpu_access_count"], 64)
        self.assertEqual(record["actual_model_cpu_count"], 4)
        self.assertEqual(self.psutil.cpu_percent.call_count, 2)
        self.psutil.cpu_percent.assert_called_with(interval=.5, percpu=True)

    def test_refuse_old_restricted_four_cpu_mask(self):
        self.setup_masks(allowed=range(4))
        with self.assertRaisesRegex(RuntimeError, "parent64 CPU access"):
            v2.bind_idle_cpu_cores()
        self.setter.assert_not_called()
        self.psutil.cpu_percent.assert_not_called()

    def test_idle_in_only_one_frame_not_eligible(self):
        frames = [[90.] * 128, [90.] * 128]
        for cpu in (36, 37, 70, 95):
            frames[0][cpu] = frames[1][cpu] = 1.
        frames[1][95] = 25.
        self.setup_masks(frames=frames)
        with self.assertRaisesRegex(RuntimeError, "idle in both"):
            v2.bind_idle_cpu_cores()
        self.setter.assert_not_called()

    def test_missing_cpu_sample_fails_closed(self):
        self.setup_masks(frames=[[0.] * 64, [0.] * 64])
        with self.assertRaisesRegex(RuntimeError, "incomplete or invalid"):
            v2.bind_idle_cpu_cores()
        self.setter.assert_not_called()

    def test_nonfinite_sample_fails_closed(self):
        frames = [[0.] * 128, [0.] * 128]
        frames[1][70] = float("nan")
        self.setup_masks(frames=frames)
        with self.assertRaisesRegex(RuntimeError, "incomplete or invalid"):
            v2.bind_idle_cpu_cores()
        self.setter.assert_not_called()

    def test_affinity_readback_mismatch_fails_closed(self):
        self.setup_masks(actual={1, 2, 3, 4})
        with self.assertRaisesRegex(RuntimeError, "mask differs"):
            v2.bind_idle_cpu_cores()

    def test_step_inherits64_but_worker_threads_stay4(self):
        plan = {"deadline_epoch": time.time() + 20000, "sidecar_source": {"path": str(Path(v2.__file__).resolve())}}
        command = v2.step_command(plan, {"worker_python": "/python"}, "/plan.json")
        self.assertIn("--cpus-per-task=64", command)
        self.assertNotIn("--cpus-per-task=4", command)
        self.assertIn("--mem=40G", command)
        self.assertIn("--ntasks=1", command)
        env = v2.lane_environment(plan, {"SLURM_JOB_ID": "196092", "SLURM_NTASKS": "1", "SLURM_PROCID": "0",
                                        "SLURM_LOCALID": "0", "SLURM_STEP_ID": "419"})
        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "4")
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], "GPU-8b8b827ace9944c1")

    def test_deployed_v1_source_untouched(self):
        source = Path(v2.__file__).with_name("tabfm_existing_sidecar.py")
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         "f284341dc3c5e6d4071faf6c9c78442568f948151bfd72acc5607d4c44f8603b")


if __name__ == "__main__":
    unittest.main()
