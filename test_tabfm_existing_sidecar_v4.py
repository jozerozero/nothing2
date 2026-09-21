"""Low-memory scheduling does not reduce native context or ensemble budgets."""
import ast
import copy
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import tabfm_existing_sidecar_v4 as v4
import tabfm_sidecar_prepare as prepare
import test_tabfm_sidecar_prepare as old_tests


def task(kind="classification", **changes):
    row = {"train_rows": 2000, "test_rows": 48, "features": 100, "classes": 10, "suite": "fake"}
    row.update(changes)
    return {"task_kind": kind, "dataset_index": 0, "dataset": kind + "-0", "row": row}


class SmallLaneTests(unittest.TestCase):
    def test_exact_thresholds_eligible_without_changing_rows(self):
        item = task()
        original = copy.deepcopy(item)
        self.assertTrue(v4.task_eligibility(item)["eligible"])
        self.assertEqual(item, original)
        self.assertTrue(v4.task_eligibility(task("regression"))["eligible"])

    def test_large_and_hierarchy_formal_tasks_ineligible(self):
        for changes in ({"test_rows": 49}, {"features": 101}, {"classes": 11}):
            self.assertFalse(v4.task_eligibility(task(**changes))["eligible"])

    def test_missing_invalid_dimensions_cannot_use_work_size_guess(self):
        item = task("regression")
        item["row"] = {"work_size": 1}
        self.assertFalse(v4.task_eligibility(item)["eligible"])
        for value in (None, True, 10., 0, -1):
            self.assertFalse(v4.task_eligibility(task(features=value))["eligible"])
        self.assertFalse(v4.task_eligibility(task(classes=None))["eligible"])

    def test_regression_shape_overlay_never_changes_original_worker_task(self):
        item = task("regression")
        item["row"] = {"work_size": 99999, "input_fingerprint": "frozen-input"}
        original = copy.deepcopy(item)
        overlay = {"0": {"train_rows": 830, "test_rows": 208, "features": 10,
                         "receipt": {"path": "/pinned-original.json", "sha256": "frozen"}}}
        record = v4.task_eligibility(item, overlay)
        self.assertTrue(record["eligible"])
        self.assertEqual(item, original)
        self.assertEqual(record["shape_source"], "verified_pinned_original_step22175_receipt")
        self.assertEqual(record["shape_receipt"], overlay["0"]["receipt"])

    def test_audit_includes_all_eligible_and_excluded_rows(self):
        items = [task(), task("regression", features=None), task(classes=11)]
        audit = v4.eligibility_audit({"plan_id": "new"}, {"manifest_id": "frozen"}, items)
        self.assertEqual(audit["eligible_counts"], {"classification": 1, "regression": 0})
        self.assertEqual(len(audit["records"]), 3)
        self.assertTrue(audit["full_original_support_and_test_unchanged"])
        self.assertTrue(audit["smoke_exempt_from_size_filter_but_same16GiB_operational_guard"])

    def test_lower_memory_operational_guards_only(self):
        safe = {"own_tree_rss_bytes": 16 * v4.GIB, "same_uid_rss_bytes": 56 * v4.GIB,
                "other_same_uid_rss_bytes": 40 * v4.GIB}
        v4.guard_snapshot(safe, startup=True)
        for changes, startup in (({"own_tree_rss_bytes": 16 * v4.GIB + 1}, False),
                                 ({"same_uid_rss_bytes": 60 * v4.GIB + 1}, False),
                                 ({"other_same_uid_rss_bytes": 40 * v4.GIB + 1}, True)):
            with self.assertRaises(v4.ResourceGuardError):
                v4.guard_snapshot(dict(safe, **changes), startup=startup)

    def test_step20g_still_one_gpu_runtime_four_threads(self):
        command = v4.step_command({"deadline_epoch": time.time() + 20000,
            "sidecar_source": {"path": str(Path(v4.__file__))}}, {"worker_python": "/python"}, "/plan")
        self.assertIn("--mem=20G", command)
        self.assertIn("--cpus-per-task=64", command)
        self.assertIn("--ntasks=1", command)

    def test_four_smokes_use_original_tasks_not_formal_filter(self):
        source = Path(v4.__file__).read_text()
        self.assertIn("selected = frozen.smoke_tasks(man, tasks)", source)
        self.assertLess(source.index("for task in selected:"), source.index("for task in sorted(eligible_tasks"))
        self.assertIn('"--threads", "4"', Path(v4.frozen.__file__).read_text())

    def test_preparer_v4_only_relaxes_startup_for_smaller_child(self):
        item = old_tests.snapshot(1790025913)
        item["owned_processes"][0]["rss"] = 30 * v4.GIB
        with patch.object(prepare, "SIDECAR_SCRIPT", "tabfm_existing_sidecar_v3.py"):
            with self.assertRaises(RuntimeError):
                prepare.check_parent(item)
        with patch.object(prepare, "SIDECAR_SCRIPT", "tabfm_existing_sidecar_v4.py"):
            self.assertEqual(prepare.check_parent(item)[1], 30 * v4.GIB)
            item["owned_processes"][0]["rss"] = 41 * v4.GIB
            with self.assertRaises(RuntimeError):
                prepare.check_parent(item)

    def test_preparer_pins_overlay_and_both_runtime_helpers(self):
        man = {"output_root": str(prepare.OUT)}
        original_tasks = [task("regression")]
        overlay = {"rows": {}, "skipped": [{"dataset_index": 0, "reason": "missing"}]}
        build = Mock(return_value=overlay)
        fixture = old_tests.PrepareTests("test_complete_plan_has_operational_limits")
        with patch.object(prepare, "SIDECAR", "existing196092-single-20260922-v4"), \
                patch.object(prepare, "SIDECAR_SCRIPT", "tabfm_existing_sidecar_v4.py"), \
                patch.object(v4.frozen, "load_campaign", return_value=(man, original_tasks)), \
                patch.dict(sys.modules, {"tabfm_regression_shapes": SimpleNamespace(build=build)}):
            plan = fixture.make_plan(old_tests.snapshot(1790025913), old_tests.snapshot(1790025943), 1790025950)
        build.assert_called_once_with(man, original_tasks)
        self.assertEqual(plan["mem_gib"], 20)
        self.assertEqual(plan["cpu_count"], 4)
        self.assertEqual(plan["inherited_cpu_count"], 64)
        self.assertEqual(plan["small_task_limits"], v4.SMALL_TASK_LIMITS)
        self.assertEqual(plan["regression_shape_overlay"], overlay)
        self.assertEqual(len(plan["source_records"]), 2)

    def test_previous_versions_and_shared_helper_are_unchanged(self):
        root = Path(v4.__file__).parent
        expected = {"tabfm_existing_sidecar.py": "f284341dc3c5e6d4071faf6c9c78442568f948151bfd72acc5607d4c44f8603b",
            "tabfm_existing_sidecar_v2.py": "aa0a3e4d0b0615057124ec7907462c6f836bc500a82e0be532db08d5d7db49f5",
            "tabfm_existing_sidecar_v3.py": "1e6a9694df0540549d1e558e3c016ab359f3cf991d85a4230a6e08dd73816998",
            "tabfm_local_tmp.py": "1c07095857ecbb5f90e453c10342ef33053196a03d118604c695d70fbe015ef4",
            "tabfm_default_dispatch.py": "7f3c7672ff0100247edf0b150fa7cf683d5543e8ab28284105470177d20ab30b"}
        for name, sha in expected.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), sha)


if __name__ == "__main__":
    unittest.main()
