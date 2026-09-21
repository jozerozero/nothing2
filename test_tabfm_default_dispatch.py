"""CPU-only contracts; no GPU, remote commands, subprocess submission or network."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import tabfm_default_dispatch as d


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "new_tabfm681"
        self.root.mkdir()
        self.worker = self.base / "fake_worker.py"
        self.worker.write_text("# Frozen fake worker\n")
        self.man = {"output_root": str(self.root), "worker_python": sys.executable,
                    "worker_script": self.identity(self.worker),
                    "per_task_rss_limit_bytes": 48 * 1024**3, "per_task_timeout_seconds": 7200}
        self.tasks = []
        for kind, count in d.COUNTS.items():
            rows = [{"dataset_index": i, "dataset": f"{kind}-{i}", "suite": "fake",
                     "task_kind": kind, "work_size": i + 1, "input_fingerprint": f"input-{i}"}
                    for i in range(count)]
            if kind == "classification":
                for i, row in enumerate(rows):
                    row.update(train_rows=i + 2, test_rows=2, features=3, classes=12 if i == 2 else 2)
            data = {"rows": rows}
            data["manifest_id"] = d.digest(data)
            path = self.base / f"{kind}.json"
            path.write_text(json.dumps(data))
            self.man[kind + "_manifest"] = self.identity(path)
            self.tasks.extend({"task_kind": kind, "dataset_index": i, "dataset": row["dataset"],
                               "row": row, "data_manifest_id": data["manifest_id"]} for i, row in enumerate(rows))
        self.man["manifest_id"] = d.digest(self.man)
        self.campaign = self.root / "manifest.json"
        self.campaign.write_text(json.dumps(self.man))
        self.owner = {"rank": 0, "job": "123", "step": "2", "node": "safe-node",
                      "uuid": "aabb", "pci": "0000:01:00.0"}
        self.environment = {"SLURM_JOB_ID": "123", "SLURM_PROCID": "0", "SLURM_LOCALID": "0",
                            "SLURM_NTASKS": "4", "SLURM_STEP_ID": "2", "ROCR_VISIBLE_DEVICES": "3"}

    @staticmethod
    def identity(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def result(self, task=None):
        task = task or self.tasks[0]
        metrics = {"accuracy": .5} if task["task_kind"] == "classification" else {"rmse": 1., "mae": .5, "r2": -.3}
        return {"complete": True, "status": "complete", "manifest_id": self.man["manifest_id"],
                "task_kind": task["task_kind"], "dataset_index": task["dataset_index"], "dataset": task["dataset"],
                "data_manifest_id": task["data_manifest_id"], "input_fingerprint": task["row"]["input_fingerprint"],
                "worker_source_sha256": self.man["worker_script"]["sha256"], "full_test_split": True,
                "data_audit": {"full_test_split": True}, "ensemble_audit": {"actual_ensemble_count": 32},
                "actual_ensemble_count": 32, "metrics": metrics,
                "physical_gpu": {"uuid": self.owner["uuid"], "pci_bus_id": self.owner["pci"]}}

    def gpu_records(self):
        records = [dict(self.owner, manifest_id=self.man["manifest_id"], rank=rank,
                        uuid=f"aab{rank}", pci=f"0000:0{rank+1}:00.0", step="0") for rank in range(4)]
        for record in records:
            d.atomic(self.man, self.root / "preflight" / "123" / f"rank-{record['rank']}.json", record)
        return records

    def test_frozen_scope_and_sha(self):
        man, tasks = d.load_campaign(self.campaign)
        self.assertEqual(man, self.man)
        self.assertEqual(len(tasks), 681)
        self.assertEqual([t["dataset_index"] for t in tasks[-224:]], list(range(224)))
        self.worker.write_text("# Changed\n")
        with self.assertRaisesRegex(RuntimeError, "Frozen SHA"):
            d.load_campaign(self.campaign)

    def test_manifest_digest_required(self):
        man = copy.deepcopy(self.man)
        man["per_task_timeout_seconds"] = 1
        self.campaign.write_text(json.dumps(man))
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            d.load_campaign(self.campaign)

    def test_atomic_claims_cannot_overwrite_or_escape(self):
        self.assertTrue(d.claim(self.man, self.tasks[0], self.owner))
        self.assertFalse(d.claim(self.man, self.tasks[0], dict(self.owner, rank=1)))
        self.assertEqual(d.read(d.task_path(self.man, "claims", self.tasks[0]))["rank"], 0)
        with self.assertRaises(RuntimeError):
            d.atomic(self.man, self.base / "outside.json", {})
        with self.assertRaises(FileExistsError):
            d.atomic(self.man, d.task_path(self.man, "claims", self.tasks[0]), {})

    def test_smoke_selection_all_protocols_full_rows(self):
        selected = d.smoke_tasks(self.man, self.tasks)
        self.assertEqual([(t["task_kind"], t["dataset_index"]) for t in selected],
                         [("classification", 0), ("classification", 2), ("regression", 0), ("regression", 1)])
        self.assertIs(selected[0]["row"], self.tasks[0]["row"])

    def test_visibility_cleaned_and_multi_mask_rejected(self):
        env = dict(self.environment, HIP_VISIBLE_DEVICES="0", CUDA_VISIBLE_DEVICES="0", GPU_DEVICE_ORDINAL="0")
        clean = d.normalize_visibility(env)
        self.assertEqual(clean["ROCR_VISIBLE_DEVICES"], "3")
        self.assertNotIn("HIP_VISIBLE_DEVICES", clean)
        with self.assertRaises(RuntimeError):
            d.normalize_visibility(dict(env, ROCR_VISIBLE_DEVICES="0,1"))

    def test_preflight_distinct_physical_ids(self):
        self.gpu_records()
        with patch.dict(os.environ, self.environment):
            self.assertEqual(len(d.check_preflight(self.man)), 4)
            path = self.root / "preflight" / "123" / "rank-3.json"
            bad = d.read(path)
            bad["uuid"] = "aab0"
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError, "distinct physical GPUs"):
                d.check_preflight(self.man)

    def test_binding_rejects_different_gpu(self):
        self.gpu_records()
        with patch.dict(os.environ, self.environment), patch.object(d, "gpu_record", return_value=self.owner):
            with self.assertRaisesRegex(RuntimeError, "binding differs"):
                d.binding(self.man)

    def test_worker_environment_freezes_expected_gpu(self):
        with patch.dict(os.environ, dict(self.environment, PYTHONPATH="old", PYTHONHOME="old",
                                         CUDA_VISIBLE_DEVICES="0", HIP_VISIBLE_DEVICES="0")):
            env = d.worker_environment(self.man, self.owner)
        self.assertEqual(env["EXPECTED_GPU_UUID"], self.owner["uuid"])
        self.assertEqual(env["EXPECTED_GPU_PCI_BUS_ID"], self.owner["pci"])
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], "3")
        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        for key in ("PYTHONPATH", "PYTHONHOME", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
            self.assertNotIn(key, env)
        self.assertTrue(Path(env["TMPDIR"]).is_relative_to(self.root))

    def test_success_validation_rejects_wrong_incomplete_or_nonfinite(self):
        path = self.root / "validate.json"
        result = self.result()
        for changes in ({"manifest_id": "old"}, {"complete": False}, {"full_test_split": False},
                        {"worker_source_sha256": "bad"}, {"metrics": {"accuracy": float("nan")}},
                        {"input_fingerprint": "changed"}, {"actual_ensemble_count": 0}):
            path.write_text(json.dumps(dict(result, **changes)))
            with self.assertRaises(RuntimeError):
                d.valid_result(path, self.man, self.tasks[0])
        path.write_text(json.dumps(result))
        self.assertEqual(d.valid_result(path, self.man, self.tasks[0]), result)

    def fake_popen(self, result, returncode=0):
        outer = self
        class Process:
            pid = 999999
            def __init__(self, command, **kwargs):
                self.returncode = None
                self.command, self.kwargs = command, kwargs
                outer.last_child = self
            def poll(self):
                return self.returncode
            def wait(self, timeout=None):
                output = Path(self.command[self.command.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(result))
                self.returncode = returncode
                return returncode
        return Process

    def test_fresh_child_validated_before_publication(self):
        with patch.object(d.subprocess, "Popen", self.fake_popen(self.result())), patch.object(d, "process_rss", return_value=4096):
            self.assertTrue(d.launch(self.man, self.campaign, self.tasks[0], self.owner))
        child = self.last_child
        self.assertTrue(child.kwargs["start_new_session"])
        self.assertEqual(child.command[-2:], ["--threads", "4"])
        self.assertIn("/worker_outputs/", child.command[child.command.index("--output") + 1])
        self.assertEqual(d.read(d.task_path(self.man, "results", self.tasks[0]))["status"], "complete")
        self.assertEqual(len(list((self.root / "attempts").glob("*.json"))), 1)

    def test_nonzero_exit_cannot_publish_apparent_success(self):
        with patch.object(d.subprocess, "Popen", self.fake_popen(self.result(), 3)), patch.object(d, "process_rss", return_value=0):
            self.assertFalse(d.launch(self.man, self.campaign, self.tasks[0], self.owner))
        self.assertFalse(d.read(d.task_path(self.man, "results", self.tasks[0]))["complete"])
        self.assertEqual(len(list((self.root / "errors").glob("*.json"))), 1)

    def test_rss_budget_terminates_process_and_records_failure(self):
        def terminate(child):
            child.returncode = -15
        with patch.object(d.subprocess, "Popen", self.fake_popen(self.result())), \
                patch.object(d, "process_rss", return_value=48 * 1024**3 + 1), \
                patch.object(d, "stop_child", side_effect=terminate) as stop:
            self.assertFalse(d.launch(self.man, self.campaign, self.tasks[0], self.owner))
            stop.assert_called_once()
        self.assertEqual(d.read(d.task_path(self.man, "results", self.tasks[0]))["reason"], "rss_budget_exceeded")

    def test_timeout_budget_without_sleep(self):
        def terminate(child):
            child.returncode = -15
        with patch.object(d.subprocess, "Popen", self.fake_popen(self.result())), \
                patch.object(d, "process_rss", return_value=0), \
                patch.object(d.time, "monotonic", side_effect=[0., 7201.]), \
                patch.object(d, "stop_child", side_effect=terminate):
            self.assertFalse(d.launch(self.man, self.campaign, self.tasks[0], self.owner))
        self.assertEqual(d.read(d.task_path(self.man, "results", self.tasks[0]))["reason"], "task_timeout")

    def test_smoke_gate_needs_all_four_current_job_successes(self):
        records = self.gpu_records()
        selected = d.smoke_tasks(self.man, self.tasks)
        with patch.dict(os.environ, self.environment):
            with self.assertRaises(FileNotFoundError):
                d.check_smoke(self.man, self.tasks, publish=True)
            for rank, task in enumerate(selected):
                self.assertTrue(d.claim(self.man, task, records[rank], smoke=True))
                d.atomic(self.man, d.task_path(self.man, "smoke", task), self.result(task))
            d.check_smoke(self.man, self.tasks, publish=True)
            self.assertEqual(len(d.check_smoke(self.man, self.tasks)["tasks"]), 4)
            with self.assertRaises(FileExistsError):
                d.check_smoke(self.man, self.tasks, publish=True)

    def test_one_failure_continues_queue_no_implicit_retry(self):
        with patch.object(d, "binding", return_value=self.owner), patch.object(d, "check_smoke"), \
                patch.object(d, "launch", side_effect=[False, True]) as launch:
            result = d.dispatch(self.man, self.campaign, self.tasks[:2])
        self.assertEqual(launch.call_count, 2)
        self.assertEqual(result["attempts"], {"failed": 1, "success": 1})
        self.assertFalse(d.claim(self.man, self.tasks[0], self.owner))

    def test_status_does_not_equate_claim_or_error_with_complete(self):
        d.claim(self.man, self.tasks[0], self.owner)
        d.atomic(self.man, d.task_path(self.man, "results", self.tasks[1]), {"complete": False})
        summary = d.status(self.man, self.tasks)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["counts"]["classification"]["claimed_without_result"], 1)
        self.assertEqual(summary["counts"]["classification"]["failed_or_invalid"], 1)

    def test_shell_resources_and_manifest_not_spool_path(self):
        shell = Path(d.__file__).with_name("tabfm_default_slurm.sh")
        text = shell.read_text()
        for token in ("--nodes=1", "--ntasks=4", "--gpus-per-task=1", "--cpus-per-task=4", "--mem=256G",
                      "--time=24:00:00", "--qos=bgqos", "--gpu-bind=single:1", "--no-requeue", "--nice=0"):
            self.assertIn(token, text)
        self.assertNotIn("BASH_SOURCE", text)
        self.assertIn('["worker_script"]["path"]', text)
        self.assertIn('${1:-${TABFM_CAMPAIGN:-}}', text)
        self.assertNotIn("sleep", text)
        self.assertLess(text.index(" check-smoke "), text.index(" run "))
        subprocess.run(["bash", "-n", str(shell)], check=True)


if __name__ == "__main__":
    unittest.main()
