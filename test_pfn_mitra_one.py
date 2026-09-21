"""CPU-only original-Mitra contract tests; never load data, weights, or a GPU."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


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


class LoadedHipBindingTests(unittest.TestCase):
    def test_mapped_segments_and_symlink_resolve_one_runtime(self):
        with tempfile.TemporaryDirectory(prefix="test-pfn-mitra-hip-") as temporary:
            library = Path(temporary) / "libamdhip64.so.6.0"
            library.write_bytes(b"fake; never dlopen in unit tests")
            symlink = Path(temporary) / "libamdhip64.so"
            symlink.symlink_to(library.name)
            maps = (f"000-001 r-xp 0000 00:00 1 {library}\n"
                    f"002-003 r--p 0000 00:00 1 {library}\n"
                    f"004-005 r--p 0000 00:00 1 {symlink}\n"
                    "006-007 rw-p 0000 00:00 0 [heap]\n")
            self.assertEqual(worker.select_loaded_hip_library(maps), library.resolve())

    def test_no_loaded_runtime_fails_without_system_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            worker.select_loaded_hip_library("000-001 rw-p 0000 00:00 0 [heap]\n")

    def test_multiple_loaded_hip_versions_rejected(self):
        with tempfile.TemporaryDirectory(prefix="test-pfn-mitra-hip-") as temporary:
            paths = [Path(temporary) / f"libamdhip64.so.{v}" for v in (6, 7)]
            for path in paths:
                path.write_bytes(b"fake")
            maps = "\n".join(f"000-001 r-xp 0000 00:00 1 {path}" for path in paths)
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                worker.select_loaded_hip_library(maps)

    def test_relative_and_deleted_runtime_rejected(self):
        for path in ("libamdhip64.so.6", "/tmp/libamdhip64.so.6 (deleted)"):
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, "absolute library path"):
                worker.select_loaded_hip_library(f"000-001 r-xp 0000 00:00 1 {path}")

    def test_gpu_visibility_rejected_before_reading_maps(self):
        for available, count in ((False, 0), (True, 2)):
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available,
                                                         device_count=lambda: count))
            with self.subTest(available=available, count=count), \
                    patch.object(Path, "read_text", side_effect=AssertionError("no maps read")), \
                    self.assertRaisesRegex(RuntimeError, "Exactly one"):
                worker.gpu_identity(torch)

    def fake_torch(self):
        smoke = SimpleNamespace(sum=lambda: SimpleNamespace(cpu=lambda: 16))
        return SimpleNamespace(version=SimpleNamespace(hip="6.4.test"), float32="FP32",
            ones=Mock(return_value=smoke),
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                init=Mock(), set_device=Mock(), get_device_name=lambda _: "AMD test",
                get_device_properties=lambda _: SimpleNamespace(total_memory=100)))

    def binding_contexts(self, torch, returned_pci=b"0000:47:00.0", returned_uuid="test-uuid"):
        def lookup(buffer, _size, device):
            self.assertEqual(device, 0)
            buffer.value = returned_pci
            return 0
        fake_library = SimpleNamespace(hipDeviceGetPCIBusId=Mock(side_effect=lookup))
        def read(path, *args, **kwargs):
            if str(path) == "/proc/self/maps":
                return "mock maps"
            if str(path) == "/sys/bus/pci/devices/0000:47:00.0/unique_id":
                return returned_uuid + "\n"
            raise AssertionError(f"Unexpected file read: {path}")
        return (patch.dict(worker.os.environ, {"EXPECTED_GPU_UUID": "GPU-test-uuid",
                    "EXPECTED_GPU_PCI_BUS_ID": "0000:47:00.0"}),
                patch.object(Path, "read_text", read),
                patch.object(worker, "select_loaded_hip_library", return_value=Path("/torch/lib/libamdhip64.so.6")),
                patch.object(worker.ctypes, "CDLL", return_value=fake_library),
                patch.object(worker.os, "RTLD_NOLOAD", 4, create=True))

    def test_loaded_runtime_noload_and_physical_compute_audit(self):
        torch = self.fake_torch()
        environment, files, select, cdll, noload = self.binding_contexts(torch)
        with environment, files, select, cdll as load, noload:
            audit = worker.gpu_identity(torch)
            load.assert_called_once_with("/torch/lib/libamdhip64.so.6", mode=4 | worker.os.RTLD_LOCAL)
        self.assertEqual(audit["hip_runtime_library"], "/torch/lib/libamdhip64.so.6")
        self.assertEqual(audit["uuid"], "test-uuid")
        torch.cuda.init.assert_called_once_with()
        torch.cuda.set_device.assert_called_once_with(0)
        torch.ones.assert_called_once_with((4, 4), dtype="FP32", device="cuda:0")

    def test_physical_gpu_mismatch_rejected_before_compute(self):
        torch = self.fake_torch()
        environment, files, select, cdll, noload = self.binding_contexts(torch, returned_uuid="wrong-uuid")
        with environment, files, select, cdll, noload, self.assertRaisesRegex(RuntimeError, "Physical GPU mismatch"):
            worker.gpu_identity(torch)
        torch.ones.assert_not_called()


if __name__ == "__main__":
    unittest.main()
