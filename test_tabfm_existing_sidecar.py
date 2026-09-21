"""CPU-only safety contracts for the new one-GPU operational sidecar."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import tabfm_existing_sidecar as s
import test_tabfm_default_dispatch as fixtures


class SidecarTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        now = time.time()
        self.plan = {"parent_job_id": s.PARENT, "node": s.NODE, "gpus": [dict(s.GPU)], "sidecar_id": "test-sidecar",
            "max_lanes": 1, "cpu_count": 4, "mem_gib": 40, "parent_gpu_count": 8, "parent_mem_gib": 64,
            "all8_gpu_reservation_verified": True, "resources_available_verified": True, "free_cpu_cores": 8,
            "startup_other_rss_bytes": 14 * s.GIB, "deadline_epoch": now + 20000, "probe_epoch": now - 1,
            "external_idle_probes": [{**s.GPU, "epoch": now - ago, "gpu_busy_percent": 0, "used_vram_bytes": 13 << 20}
                                     for ago in (20, 1)],
            "sidecar_source": {"path": str(Path(s.__file__).resolve()),
                               "sha256": hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest()},
            "campaign_path": str(self.base / "campaign.json")}
        self.plan["plan_id"] = s.frozen.digest(self.plan)
        self.planpath = self.base / "plan.json"
        self.planpath.write_text(json.dumps(self.plan))
        self.man = {"output_root": str(self.base / "new_campaign"), "worker_python": "/fake/python", "manifest_id": "frozen-id",
                    "per_task_timeout_seconds": 7200, "per_task_rss_limit_bytes": 48 * s.GIB}
        self.safe = {"own_tree_rss_bytes": 2 * s.GIB, "same_uid_rss_bytes": 16 * s.GIB,
                     "other_same_uid_rss_bytes": 14 * s.GIB}

    def test_plan_identity_freshness_and_exact_authorized_scope(self):
        with patch.object(s.frozen, "load_campaign", return_value=(self.man, [])):
            plan, _, _ = s.load_plan(self.planpath, fresh=True)
            self.assertEqual(plan["gpus"], [s.GPU])
            bad = dict(self.plan, parent_job_id="206116")
            bad["plan_id"] = s.frozen.digest({k: v for k, v in bad.items() if k != "plan_id"})
            self.planpath.write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError, "Only authorized"):
                s.load_plan(self.planpath)

    def test_real_single_rank_environment_and_uuid_mask(self):
        env = {"SLURM_JOB_ID": s.PARENT, "SLURM_NTASKS": "1", "SLURM_PROCID": "0", "SLURM_LOCALID": "0",
               "SLURM_STEP_ID": "13", "ROCR_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "HIP_VISIBLE_DEVICES": "3",
               "CUDA_VISIBLE_DEVICES": "3", "PYTHONPATH": "old"}
        clean = s.lane_environment(self.plan, env)
        self.assertEqual(clean["ROCR_VISIBLE_DEVICES"], "GPU-" + s.GPU["uuid"])
        self.assertEqual(clean["EXPECTED_GPU_PCI_BUS_ID"], s.GPU["pci"])
        self.assertNotIn("HIP_VISIBLE_DEVICES", clean)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", clean)
        self.assertNotIn("PYTHONPATH", clean)
        self.assertEqual(clean["SLURM_NTASKS"], "1")
        with self.assertRaises(RuntimeError):
            s.lane_environment(self.plan, dict(env, SLURM_NTASKS="4"))

    def test_parent_capacity_and_expiry_are_checked(self):
        end = datetime.datetime.fromtimestamp(self.plan["deadline_epoch"] + 600, datetime.timezone.utc).replace(tzinfo=None).isoformat()
        raw = f"JobId=196092 JobState=RUNNING NumNodes=1 NodeList={s.NODE} UserId=guangyi.chen(123) AllocTRES=cpu=64,mem=64G,gres/gpu=8 EndTime={end}"
        self.assertEqual(s.verify_parent(self.plan, raw)["JobId"], s.PARENT)
        for bad in (raw.replace("mem=64G", "mem=32G"), raw.replace("gres/gpu=8", "gres/gpu=1"), raw.replace("RUNNING", "COMPLETED")):
            with self.assertRaises(RuntimeError):
                s.verify_parent(self.plan, bad)

    def test_operational_caps_do_not_change_frozen_budget(self):
        s.guard_snapshot(self.safe, startup=True)
        for changes, startup in (({"own_tree_rss_bytes": 32 * s.GIB + 1}, False),
                                 ({"same_uid_rss_bytes": 60 * s.GIB + 1}, False),
                                 ({"other_same_uid_rss_bytes": 20 * s.GIB + 1}, True)):
            with self.assertRaises(s.ResourceGuardError):
                s.guard_snapshot(dict(self.safe, **changes), startup=startup)
        self.assertEqual(self.man["per_task_rss_limit_bytes"], 48 * s.GIB)

    def test_watchdog_fails_closed_on_any_inspection_error(self):
        events = []
        original = s.frozen.process_rss
        with patch.object(s, "rss_snapshot", side_effect=PermissionError("cannot inspect")):
            with s.operational_watchdog(events):
                with self.assertRaises(s.ResourceGuardError):
                    s.frozen.process_rss(123)
        self.assertIs(s.frozen.process_rss, original)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["inspection_exception"], "PermissionError")

    def test_watchdog_retains_trigger_and_restores_sampler(self):
        events = []
        with patch.object(s, "rss_snapshot", return_value=dict(self.safe, own_tree_rss_bytes=33 * s.GIB)):
            with s.operational_watchdog(events):
                with self.assertRaises(s.ResourceGuardError):
                    s.frozen.process_rss(123)
        self.assertIn("32GiB", events[0]["reason"])

    def test_isolated_smoke_routes_original_manifest(self):
        derived = s.smoke_manifest(self.plan, self.man)
        self.assertEqual(derived["manifest_id"], self.man["manifest_id"])
        self.assertNotEqual(derived["output_root"], self.man["output_root"])
        self.assertEqual(self.man["output_root"], str(self.base / "new_campaign"))
        with patch.object(s, "rss_snapshot", return_value=self.safe), patch.object(s.frozen, "launch", return_value=True) as launch:
            self.assertTrue(s.guarded_launch(self.plan, derived, {}, {}, [], smoke=True))
        self.assertEqual(launch.call_args.args[1], Path(self.plan["campaign_path"]))

    def test_step_requests_only_one_lane_no_allocation_mutation(self):
        command = s.step_command(self.plan, self.man, self.planpath)
        for expected in ("--jobid=196092", "--ntasks=1", "--cpus-per-task=4", "--mem=40G", "--gpus=8", "--gpu-bind=none"):
            self.assertIn(expected, command)
        self.assertNotIn("sbatch", command)
        self.assertNotIn("scancel", command)
        self.assertEqual(command[-2:], ["--mode", "lane"])

    def test_internal_idle_probe_rejects_occupied_gpu(self):
        s.check_idle({"gpu_busy_percent": 0, "used_vram_bytes": 13 << 20})
        for busy, memory in ((1, 13 << 20), (0, 128 << 20)):
            with self.assertRaises(RuntimeError):
                s.check_idle({"gpu_busy_percent": busy, "used_vram_bytes": memory})

    def test_guard_failure_kills_only_own_worker_and_preserves_failed_receipt(self):
        fixture = fixtures.DispatcherTests("test_fresh_child_validated_before_publication")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        events = []
        def stop(child):
            child.returncode = -15
        with patch.object(s.frozen.subprocess, "Popen", fixture.fake_popen(fixture.result())), \
                patch.object(s, "rss_snapshot", return_value=dict(self.safe, own_tree_rss_bytes=33 * s.GIB)), \
                patch.object(s.frozen, "stop_child", side_effect=stop) as killer:
            with s.operational_watchdog(events):
                self.assertFalse(s.frozen.launch(fixture.man, fixture.campaign, fixture.tasks[0], fixture.owner))
        killer.assert_called_once_with(fixture.last_child)
        output = s.frozen.read(s.frozen.task_path(fixture.man, "results", fixture.tasks[0]))
        self.assertFalse(output["complete"])
        self.assertIn("resource_guard", output["reason"])
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
