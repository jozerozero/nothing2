"""CPU-only original-Mitra contract tests; never load data, weights, or a GPU."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pfn_mitra_one as worker


class PFNMitraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_path = HERE.parent / "evaluation_manifest.json"
        cls.reference_path = (HERE.parent / "rank_comparison_20260921T171633" /
                              "results/step-22175/row-196.json")
        cls.manifest, cls.row, cls.checkpoint = worker.common.load_manifest(cls.manifest_path, 22175, 196)
        cls.source = json.loads(cls.reference_path.read_text())

    def with_source(self, source, **changes):
        with tempfile.TemporaryDirectory(prefix="test-pfn-mitra-") as temporary:
            path = Path(temporary) / "reference.json"
            path.write_text(json.dumps(source))
            return worker.load_source_result(path, changes.get("manifest", self.manifest),
                                             changes.get("row", self.row),
                                             changes.get("checkpoint", self.checkpoint))

    def test_pfn_original_receipt_accepts(self):
        self.assertEqual(self.with_source(self.source), self.source)

    def test_non_pfn_rejects(self):
        row = {**self.row, "suite": "talent"}
        with self.assertRaisesRegex(RuntimeError, "Only PFN28"):
            self.with_source(self.source, row=row)

    def test_wrong_original_identity_rejects(self):
        for key, value in (("checkpoint_step", 22225), ("complete", False),
                           ("manifest_id", "wrong"), ("input_fingerprint", "wrong"),
                           ("dataset_index", 197), ("row_id", "wrong"),
                           ("dataset_protocol_fingerprint", "wrong")):
            source = {**self.source, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "mismatch"):
                self.with_source(source)

    def test_wrong_checkpoint_lineage_rejects(self):
        source = deepcopy(self.source)
        source["checkpoint"]["finetune_step"] = 50
        with self.assertRaisesRegex(RuntimeError, "unfinetuned"):
            self.with_source(source)

    def test_missing_loop3_verification_rejects(self):
        source = {**self.source, "actual_forward_block_calls": [[1] * 12]}
        with self.assertRaisesRegex(RuntimeError, "Loop3"):
            self.with_source(source)

    def test_full_data_and_target_parity_accepts(self):
        worker.verify_data_parity(deepcopy(self.source["data_audit"]), self.source,
                                  self.source["target_transform_source"], self.source["target_transform"])

    def test_any_raw_hash_or_row_count_change_rejects(self):
        for key in ("canonical_support_frame_sha256", "canonical_test_frame_sha256",
                    "support_targets_sha256", "test_targets_sha256", "support_rows", "test_rows"):
            audit = deepcopy(self.source["data_audit"])
            audit[key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "data audit differs"):
                worker.verify_data_parity(audit, self.source, self.source["target_transform_source"],
                                          self.source["target_transform"])

    def test_split_row_identity_change_rejects(self):
        audit = deepcopy(self.source["data_audit"])
        audit["split"]["official_train_rowids_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "data audit differs"):
            worker.verify_data_parity(audit, self.source, self.source["target_transform_source"],
                                      self.source["target_transform"])

    def test_target_transform_source_and_values_reject(self):
        for record, transform in (({}, self.source["target_transform"]),
                                  (self.source["target_transform_source"], {"kind": "wrong"})):
            with self.subTest(record=record), self.assertRaisesRegex(RuntimeError, "target transform"):
                worker.verify_data_parity(self.source["data_audit"], self.source, record, transform)

    def test_runtime_recipe_strict(self):
        cfg = SimpleNamespace(hyperparams={"max_epochs": 0, "max_samples_support": 8192,
                  "max_samples_query": 1024, "precision": "bfloat16", "dim_output": 1,
                  "n_ensembles": 1, "grad_scaler_enabled": False})
        self.assertEqual(worker.check_trainer_cfg(cfg), cfg.hyperparams)
        for key, value in (("max_epochs", 1), ("max_samples_support", 4096),
                           ("max_samples_query", 512), ("precision", "float32"),
                           ("dim_output", 1000), ("n_ensembles", 8), ("grad_scaler_enabled", True)):
            changed = SimpleNamespace(hyperparams={**cfg.hyperparams, key: value})
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                worker.check_trainer_cfg(changed)

    def test_guard_no_training_and_optimizer_forbidden(self):
        class Trainer:
            def __init__(self, cfg, model):
                self.cfg, self.model = cfg, model
                self.optimizer = SimpleNamespace(step=lambda: None)
            def train(self):
                return "preprocessing-only"
        si = SimpleNamespace(TrainerFinetune=Trainer)
        cfg = SimpleNamespace(hyperparams={"max_epochs": 0, "max_samples_support": 8192,
                  "max_samples_query": 1024, "precision": "bfloat16", "dim_output": 1,
                  "n_ensembles": 1, "grad_scaler_enabled": False})
        trainer = worker.guarded_trainer(si)(cfg, SimpleNamespace(dim_output=1, use_flash_attn=False))
        self.assertEqual(trainer.train(), "preprocessing-only")
        with self.assertRaisesRegex(RuntimeError, "optimizer.step"):
            trainer.optimizer.step()

    def test_publish_never_overwrites(self):
        with tempfile.TemporaryDirectory(prefix="test-pfn-mitra-publish-") as temporary:
            path = Path(temporary) / "result.json"
            worker.common.publish_new(path, {"old": True})
            with self.assertRaises(FileExistsError):
                worker.common.publish_new(path, {"old": False})
            self.assertEqual(json.loads(path.read_text()), {"old": True})

    def test_reused_loader_and_historical_adapter_pins(self):
        self.assertEqual(worker.sha256_file(HERE / "eval_one.py"), worker.RAW_LOADER_SHA256)
        historical = HERE.parents[1] / "mitra_all_classification_regression_20260823_v3/mitra_common.py"
        self.assertEqual(worker.sha256_file(historical), worker.HISTORICAL_ADAPTER_SHA256)


if __name__ == "__main__":
    unittest.main()
