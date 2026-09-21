"""CPU-only dispatch safety tests: no Slurm/GPU/remote calls or real subprocesses."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import classification32_campaign as campaign
import classification32_dispatch as dispatch


class FinishedProcess:
    pid = 7654321
    def __init__(self, code=0):
        self.returncode = code
    def poll(self):
        return self.returncode


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.env = {"SLURM_JOB_ID": "123456", "SLURM_STEP_ID": "3",
                    "SLURM_PROCID": "0", "CLASS32_SHARD": "0"}
        for p in (patch.object(dispatch, "ROOT", self.root), patch.object(campaign, "ROOT", self.root),
                  patch.dict(os.environ, self.env), patch.object(dispatch, "CHILD", None),
                  patch.dict(sys.modules, {"psutil": SimpleNamespace(Process=Mock(),
                      NoSuchProcess=ProcessLookupError, AccessDenied=PermissionError)})):
            p.start()
            self.addCleanup(p.stop)
        self.row = {"dataset_index": 0, "dataset": "small", "train_rows": 12,
                    "test_rows": 5, "features": 2}
        self.man = {"manifest_id": "frozen-manifest", "rows": [self.row], "smoke_indices": [0],
                    "models": {"tabiclv1": {"python": "/not/executed/python", "env": {}}},
                    "per_task_rss_limit_bytes": 1024**3, "per_task_timeout_seconds": 600}
        self.owner = {"job": "123456", "step": "3", "rank": 0,
                      "node": "test-node", "uuid": "abc0", "pci": "0000:01:00.0"}

    def publish_preflight(self, duplicate_uuid=False, duplicate_pci=False):
        records = []
        for i in range(4):
            record = {"rank": i, "job": self.owner["job"], "node": self.owner["node"],
                      "uuid": f"abc{i}", "pci": f"0000:0{i+1}:00.0",
                      "manifest_id": self.man["manifest_id"]}
            if duplicate_uuid and i == 3:
                record["uuid"] = "abc0"
            if duplicate_pci and i == 3:
                record["pci"] = "0000:01:00.0"
            campaign.atomic(self.root / "preflight" / self.owner["job"] / f"rank-{i}.json", record)
            records.append(record)
        return records

    def test_visibility_four_ranks_preserve_scheduler_masks_strip_aliases(self):
        for rank in range(4):
            env = {"SLURM_NTASKS": "4", "SLURM_PROCID": str(rank), "SLURM_LOCALID": str(rank),
                   "ROCR_VISIBLE_DEVICES": str(rank + 4), "CUDA_VISIBLE_DEVICES": str(rank),
                   "HIP_VISIBLE_DEVICES": str(rank), "GPU_DEVICE_ORDINAL": str(rank), "KEEP": "yes"}
            before = dict(env)
            got = dispatch.normalize_visibility(env)
            self.assertEqual(env, before)
            self.assertEqual(got["ROCR_VISIBLE_DEVICES"], str(rank + 4))
            self.assertEqual(got["KEEP"], "yes")
            self.assertFalse(any(k in got for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")))
            old = json.loads(got["CLASS32_ORIGINAL_VISIBILITY"])
            self.assertEqual(old["CUDA_VISIBLE_DEVICES"], str(rank))

    def test_visibility_accepts_single_physical_uuid(self):
        env = {"SLURM_NTASKS": "4", "SLURM_PROCID": "2", "SLURM_LOCALID": "2",
               "ROCR_VISIBLE_DEVICES": "GPU-0123-abcdef"}
        self.assertEqual(dispatch.normalize_visibility(env)["ROCR_VISIBLE_DEVICES"], "GPU-0123-abcdef")

    def test_visibility_rejects_missing_or_multiple_or_ambiguous_masks(self):
        env = {"SLURM_NTASKS": "4", "SLURM_PROCID": "0", "SLURM_LOCALID": "0"}
        for value in ("", "0,1", "all", "-1", "0 1", "GPU-a,GPU-b"):
            with self.subTest(mask=value), self.assertRaisesRegex(RuntimeError, "Exactly one"):
                dispatch.normalize_visibility(dict(env, ROCR_VISIBLE_DEVICES=value))

    def test_visibility_rejects_wrong_task_layout(self):
        env = {"SLURM_NTASKS": "4", "SLURM_PROCID": "0", "SLURM_LOCALID": "0", "ROCR_VISIBLE_DEVICES": "0"}
        for changed in ({"SLURM_NTASKS": "8"}, {"SLURM_PROCID": "4"}, {"SLURM_LOCALID": "1"}):
            with self.subTest(changed=changed), self.assertRaisesRegex(RuntimeError, "four-rank"):
                dispatch.normalize_visibility({**env, **changed})

    def pass_gate(self, model="tabiclv1"):
        campaign.atomic(self.root / "gates" / (model + ".passed.json"),
                        {"manifest_id": self.man["manifest_id"], "indices": [0]})

    def run_dispatch(self, validator=None):
        with patch.object(campaign, "manifest_load", return_value=self.man), \
             patch.object(dispatch, "binding", return_value=self.owner), \
             patch.object(campaign, "MODELS", ("tabiclv1",)), \
             patch.object(campaign, "valid_result", validator or Mock()), \
             patch.object(dispatch, "launch", return_value=True) as launched:
            dispatch.dispatch()
            return launched

    def test_atomic_claim_unique_under_concurrency(self):
        def attempt(i):
            return dispatch.claim(self.man, "tabiclv1", self.row, dict(self.owner, pid=i))
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(attempt, range(24)))
        self.assertEqual(sum(results), 1)
        path = self.root / "claims/tabiclv1/row-000.json"
        receipt = path.read_bytes()
        self.assertFalse(attempt(999))
        self.assertEqual(path.read_bytes(), receipt)
        self.assertEqual(campaign.read(path)["manifest_id"], self.man["manifest_id"])
        self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_claims_are_per_model_and_per_dataset(self):
        self.assertTrue(dispatch.claim(self.man, "tabiclv1", self.row, self.owner))
        self.assertTrue(dispatch.claim(self.man, "tabiclv2", self.row, self.owner))
        self.assertTrue(dispatch.claim(self.man, "tabiclv1", dict(self.row, dataset_index=1), self.owner))

    def test_preflight_accepts_four_distinct_physical_gpus(self):
        expected = self.publish_preflight()
        self.assertEqual(dispatch.check_preflight(self.man), expected)

    def test_preflight_rejects_duplicate_uuid(self):
        self.publish_preflight(duplicate_uuid=True)
        with self.assertRaisesRegex(RuntimeError, "Four distinct"):
            dispatch.check_preflight(self.man)

    def test_preflight_rejects_duplicate_pci(self):
        self.publish_preflight(duplicate_pci=True)
        with self.assertRaisesRegex(RuntimeError, "Four distinct"):
            dispatch.check_preflight(self.man)

    def test_binding_must_match_own_rank_not_merely_allowed_gpu_set(self):
        records = self.publish_preflight()
        wrong = dict(self.owner, uuid=records[1]["uuid"], pci=records[1]["pci"])
        with patch.object(dispatch, "gpu_record", return_value=wrong):
            with self.assertRaises(RuntimeError):
                dispatch.binding(self.man)
        self.assertFalse((self.root / "bindings").exists())

    def test_binding_accepts_own_rank_and_publishes_receipt(self):
        self.publish_preflight()
        with patch.object(dispatch, "gpu_record", return_value=self.owner):
            self.assertEqual(dispatch.binding(self.man), self.owner)
        self.assertTrue((self.root / "bindings/123456/3/rank-0.json").is_file())

    def test_no_smoke_gate_means_no_claim_and_no_launch(self):
        launched = self.run_dispatch()
        launched.assert_not_called()
        self.assertFalse((self.root / "claims").exists())
        done = campaign.read(self.root / "worker_done/123456/rank-0.json")
        self.assertEqual(done["passed_models"], [])
        self.assertIn("not campaign completion", done["reason"])

    def test_smoke_pass_gate_is_revalidated_before_claim(self):
        self.pass_gate()
        validator = Mock(side_effect=RuntimeError("invalid smoke receipt"))
        with self.assertRaisesRegex(RuntimeError, "invalid smoke"):
            self.run_dispatch(validator)
        self.assertFalse((self.root / "claims").exists())

    def test_existing_formal_result_is_validated_and_never_overwritten(self):
        self.pass_gate()
        output = self.root / "results/tabiclv1/row-000.json"
        campaign.atomic(output, {"sentinel": "original result"})
        before = output.read_bytes()
        validator = Mock()
        launched = self.run_dispatch(validator)
        launched.assert_not_called()
        self.assertEqual(validator.call_count, 2)  # Smoke gate and formal result.
        self.assertEqual(output.read_bytes(), before)
        self.assertFalse((self.root / "claims").exists())

    def test_failed_method_smoke_has_no_pass_gate_or_later_dataset(self):
        self.man["smoke_indices"] = [0, 1]
        self.man["rows"].append(dict(self.row, dataset_index=1))
        with patch.object(campaign, "manifest_load", return_value=self.man), \
             patch.object(dispatch, "binding", return_value=self.owner), \
             patch.object(dispatch, "launch", return_value=False) as launched:
            dispatch.smoke(2)
        self.assertEqual(launched.call_count, 1)
        self.assertTrue((self.root / "gates/tabiclv1.failed.json").is_file())
        self.assertFalse((self.root / "gates/tabiclv1.passed.json").exists())

    def test_all_smoke_datasets_required_for_pass_gate(self):
        self.man["smoke_indices"] = [0, 1]
        self.man["rows"].append(dict(self.row, dataset_index=1))
        with patch.object(campaign, "manifest_load", return_value=self.man), \
             patch.object(dispatch, "binding", return_value=self.owner), \
             patch.object(dispatch, "launch", return_value=True) as launched:
            dispatch.smoke(2)
        self.assertEqual(launched.call_count, 2)
        gate = campaign.read(self.root / "gates/tabiclv1.passed.json")
        self.assertEqual(gate["indices"], [0, 1])
        self.assertFalse((self.root / "gates/tabiclv1.failed.json").exists())

    def test_exit_zero_without_output_is_not_success(self):
        with patch.object(dispatch.subprocess, "Popen", return_value=FinishedProcess()), \
             patch.object(campaign, "valid_result") as validated:
            result = dispatch.launch(self.man, "tabiclv1", self.row, smoke=True)
        self.assertFalse(result)
        validated.assert_not_called()
        errors = list((self.root / "errors").glob("*.json"))
        self.assertEqual(len(errors), 1)
        self.assertFalse(campaign.read(errors[0])["success"])
        self.assertIsNone(dispatch.CHILD)

    def test_invalid_output_is_recorded_method_failure_not_uncaught_error(self):
        output = self.root / "smoke/tabiclv1/row-000.json"
        campaign.atomic(output, {"invalid": True})
        before = output.read_bytes()
        with patch.object(dispatch.subprocess, "Popen", return_value=FinishedProcess()), \
             patch.object(campaign, "valid_result", side_effect=RuntimeError("actual32 missing")):
            result = dispatch.launch(self.man, "tabiclv1", self.row, smoke=True)
        self.assertFalse(result)
        errors = list((self.root / "errors").glob("*.json"))
        self.assertEqual(len(errors), 1)
        self.assertIn("actual32 missing", errors[0].read_text())
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
