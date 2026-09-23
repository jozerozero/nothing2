"""Pure transformation tests: no remote access, GPU, submission or source edits."""
import copy
from pathlib import Path
import unittest

from eval_one import object_digest
import tabswift_prepare_fp32_v2 as prepare


def record(name):
    return {"path": "/isolated/repo/" + name, "sha256": name + "-fake-test-hash",
            "size_bytes": 123, "mtime_ns": 100}


def old_manifest(variant="official16"):
    value = {
        "schema": 1, "name": f"tabswift_{variant}_standard681_20260922_v1",
        "created_epoch": 1, "output_root": f"/isolated/evaluation/tabswift_{variant}_standard681_20260922_v1",
        "membership_count": 681, "classification_count": 457, "regression_count": 224,
        "classification_manifest": record("classification.json"), "regression_manifest": record("regression.json"),
        "classification_raw_inputs": {"0": [record("raw-input")]},
        "weights": {"shared": record("swift.ckpt")}, "smoke_tasks": [1, 2, 3, 4],
        "worker_python": "/isolated/venv/bin/python", "worker_script": record("tabswift_one.py"),
        "worker_sources": [record("tabswift_one.py"), record("eval_one.py")],
        "per_task_rss_limit_bytes": 48 * (1 << 30), "per_task_timeout_seconds": 7200,
        "protocol": {"variant": variant,
            "n_estimators": {"classification": 16, "regression": 16} if variant == "official16" else {"classification": 32, "regression": 8},
            "strict_actual_count": variant == "budget32x8", "batch_size": 16, "random_state": 42,
            "use_amp": True, "norm_methods": ["none", "power"], "pca_dim": 100,
            "softmax_temperature": 0.9, "regression_targets": "original frozen transform", "finetuning": False},
    }
    value["manifest_id"] = object_digest(value)
    return value


def old_plan():
    value = {"schema": 1, "name": "old", "created_epoch": 1,
        "output_root": "/isolated/stage/tabswift_standard681_20260922_v1",
        "membership_count": 1362, "protocol_variants": list(prepare.VARIANTS),
        "source_records": [record("tabswift_one.py"), record("eval_one.py")],
        "campaign_manifests": [record("official16.json"), record("budget32x8.json")],
        "worker_python": "/isolated/venv/bin/python",
        "allocation": {"nodes": 1, "gpus": 4, "hours": 24},
        "dispatch_policy": "unchanged", "bootstrap_receipt": record("bootstrap.json")}
    value["plan_id"] = object_digest(value)
    return value


class PrecisionPrepareTest(unittest.TestCase):
    additions = [record(name) for name in prepare.NEW_SOURCE_NAMES]

    def transform(self, original):
        variant = original["protocol"]["variant"]
        return prepare.transform_manifest(original, record("old-manifest.json"), record("old-plan.json"),
            Path(f"/isolated/evaluation/tabswift_{variant}_fp32_standard681_20260923_v2"),
            self.additions, created_epoch=2)

    def test_both_variants_preserve_all_nonprecision_science(self):
        for variant in prepare.VARIANTS:
            old = old_manifest(variant); unchanged = copy.deepcopy(old)
            new = self.transform(old)
            self.assertEqual(old, unchanged)
            self.assertEqual(new["protocol"]["n_estimators"], old["protocol"]["n_estimators"])
            self.assertEqual(new["protocol"]["precision"], "fp32")
            self.assertIs(new["protocol"]["use_amp"], False)
            self.assertIn("unconditional autocast", new["protocol"]["precision_note"])
            for key in old.keys() - prepare.MANIFEST_MUTABLE:
                self.assertEqual(new[key], old[key])
            self.assertNotEqual(new["manifest_id"], old["manifest_id"])
            self.assertEqual(new["worker_script"], self.additions[1])
            self.assertEqual(new["worker_sources"][:2], old["worker_sources"])

    def test_old_manifest_digest_tamper_fails_closed(self):
        old = old_manifest(); old["protocol"]["random_state"] = 99
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            self.transform(old)

    def test_estimator_budget_cannot_be_altered(self):
        old = old_manifest(); old["protocol"]["n_estimators"]["regression"] = 8
        old["manifest_id"] = object_digest({k: v for k, v in old.items() if k != "manifest_id"})
        with self.assertRaisesRegex(RuntimeError, "estimator counts"):
            self.transform(old)

    def test_old_source_identity_cannot_be_replaced(self):
        with self.assertRaisesRegex(RuntimeError, "collides"):
            prepare.merge_sources([record("eval_one.py")], [record("eval_one.py")])

    def test_repeated_migration_rejected(self):
        old = self.transform(old_manifest())
        with self.assertRaisesRegex(RuntimeError, "already migrated"):
            self.transform(old)

    def test_plan_never_carries_actionable_allocation(self):
        old = old_plan(); unchanged = copy.deepcopy(old)
        records = [{**record("manifest.json"), "path": f"/isolated/evaluation/tabswift_{v}_fp32_standard681_20260923_v2/manifest.json"}
                   for v in prepare.VARIANTS]
        new = prepare.transform_plan(old, record("old-plan.json"), records, self.additions,
                                     Path("/isolated/stage") / prepare.NAME, created_epoch=2)
        self.assertEqual(old, unchanged)
        self.assertNotIn("allocation", new)
        self.assertEqual(new["legacy_allocation_reference"], old["allocation"])
        self.assertIs(new["existing_allocations_only"], True)
        self.assertIs(new["new_allocation_submission_allowed"], False)
        self.assertEqual(new["plan_id"], object_digest({k: v for k, v in new.items() if k != "plan_id"}))
        for key in old.keys() - prepare.PLAN_MUTABLE:
            self.assertEqual(new[key], old[key])

    def test_plan_variant_order_must_match(self):
        records = [{**record("manifest.json"), "path": f"/isolated/evaluation/tabswift_{v}_fp32_standard681_20260923_v2/manifest.json"}
                   for v in reversed(prepare.VARIANTS)]
        with self.assertRaisesRegex(RuntimeError, "order/name"):
            prepare.transform_plan(old_plan(), record("old-plan.json"), records, self.additions,
                                   Path("/isolated/stage") / prepare.NAME, created_epoch=2)

    def test_new_manifest_must_not_reuse_old_output(self):
        old = old_manifest()
        with self.assertRaisesRegex(RuntimeError, "isolated"):
            prepare.transform_manifest(old, record("old-manifest.json"), record("old-plan.json"),
                old["output_root"], self.additions, created_epoch=2)


if __name__ == "__main__":
    unittest.main()
