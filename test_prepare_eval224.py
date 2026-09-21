"""Synthetic, local-only manifest tests: no real data, weights, model or GPU."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("synthetic_prepare_eval224", HERE / "prepare_eval224.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)

REG_COUNTS = {"talent": 100, "BCCO": 50, "CTR23": 33, "TabArena": 13, "PFN": 28}
CLASS_COUNTS = {"talent": 200, "BCCO": 106, "OpenML-CC18": 62, "PFN": 29, "TabArena": 33, "TabZilla": 27}
STEPS = tuple(range(22176, 22226))


def upstream_digest(value):
    """Independent copy of the upstream prepare.py JSON encoding contract."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fingerprint(row):
    return upstream_digest({key: row.get(key) for key in
                            ("task_kind", "dataset", "suite", "format", "input_files",
                             "target_feature", "class_labels")})


def seal(plan):
    plan["plan_id"] = upstream_digest({key: value for key, value in plan.items() if key != "plan_id"})
    return plan


class SyntheticRun:
    def __init__(self, root):
        self.root = root
        self.checkpoints = root / "finetune"
        self.checkpoints.mkdir()
        self.source_checkpoint = self.write("baseline/step-22175.ckpt", b"synthetic baseline; never deserialize")
        self.raw = self.write("data/source.arff", b"synthetic data; metadata only")
        self.split = self.write("data/official-split.arff", b"synthetic split; metadata only")
        raw_record, split_record = self.metadata(self.raw), self.metadata(self.split)
        rows = []
        for kind, counts in (("classification", CLASS_COUNTS), ("regression", REG_COUNTS)):
            for suite, count in counts.items():
                for index in range(count):
                    dataset = f"{suite}__{kind}__{index:03d}"
                    row = {"task_kind": kind, "suite": suite, "dataset": dataset,
                           "row_id": f"{kind}::{dataset}", "source_path": str(self.raw),
                           "format": "talent_npy" if suite == "talent" else "bcco_csv" if suite == "BCCO" else "openml_arff",
                           "input_files": [deepcopy(raw_record)]}
                    if kind == "regression" and suite == "PFN":
                        row.update(target_feature="target", official_split_path=str(self.split))
                        row["input_files"].append(deepcopy(split_record))
                    row["input_fingerprint"] = fingerprint(row)
                    rows.append(row)
        self.plan = seal({"schema": 1, "rows": rows, "data_blocked": [],
                          "classification_memberships": 457, "regression_memberships": 224})
        self.plan_path = self.write_json("plan.json", self.plan)
        self.sizes = {}
        for step in STEPS:
            path = self.write(f"finetune/step-{step}.ckpt", f"synthetic checkpoint {step}".encode())
            self.sizes[path.name] = path.stat().st_size
        self.contract = {"source_step": 22175, "optimizer_updates": 50, "shared_model": True,
                         "inspect_only": False, "validation_or_test_loaded": False,
                         "checkpoint_sha256": hashlib.sha256(self.source_checkpoint.read_bytes()).hexdigest()}
        self.complete = {"complete": True, "source_step": 22175, "optimizer_updates": 50,
                         "checkpoint_count": 50, "final_step": 22225, "checkpoint_sizes": self.sizes}
        self.contract_path = self.write_json("finetune/contract.json", self.contract)
        self.complete_path = self.write_json("finetune/complete.json", self.complete)
        self.eval_data = self.write("helpers/eval_data.py", self.loader_code())
        self.vendor = root / "vendor"
        bindings = {}
        for name in ("standard_loader.py", "regression_suite_worker.py", "official_talent_regression_worker.py",
                     "talent_regression_contract.py", "prepare_benchmark_memberships.py"):
            code = b"raise AssertionError('vendor must not be imported during metadata preparation')\n"
            if name == "official_talent_regression_worker.py":
                code += b"class RegressionTargetTransform:\n    pass\n"
            path = self.write("vendor/" + name, code)
            bindings[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.vendor_manifest = self.write_json("vendor_manifest.json", bindings)
        self.native = root / "native/src"
        self.write("native/src/tabicl/__init__.py", b"raise AssertionError('native model must never be imported')\n")
        self.inference = self.write_json("inference.json", {"loop": 3, "protocol": "synthetic-native-inference"})

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def write_json(self, relative, value):
        return self.write(relative, (json.dumps(value, sort_keys=True) + "\n").encode())

    @staticmethod
    def metadata(path):
        info = path.stat()
        return {"path": str(path), "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns}

    @staticmethod
    def loader_code(diagnostic=False):
        header = "PROTOCOL = 'mitra-standard681-data-v1'\n"
        if not diagnostic:
            return (header + "raise AssertionError('loader must not be imported without verify_data')\n"
                    "def load(row):\n    raise AssertionError('unexpected data load')\n").encode()
        return (header + "def load(row):\n"
                "    audit = dict(protocol=PROTOCOL, protocol_validation=True, row_id=row['row_id'],\n"
                "                 task_kind='regression', input_fingerprint=row['input_fingerprint'],\n"
                "                 support_subsampling_in_loader=False, full_test_split=True,\n"
                "                 validation_holdout_in_loader=False, test_labels_used_for_fit_or_routing=False)\n"
                "    return None, None, None, None, audit\n").encode()

    def kwargs(self, **changes):
        values = dict(source_plan=self.plan_path, checkpoint_dir=self.checkpoints,
                      finetune_contract=self.contract_path, finetune_complete=self.complete_path,
                      eval_data=self.eval_data, vendor_dir=self.vendor, vendor_manifest=self.vendor_manifest,
                      native_source=self.native, inference_protocol=self.inference)
        values.update(changes)
        return values

    def build(self, **changes):
        return prepare.build_manifest(**self.kwargs(**changes))

    def save_plan(self):
        self.write_json("plan.json", seal(self.plan))


class PrepareEval224Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="test-eval224-")
        self.addCleanup(temporary.cleanup)
        self.fixture = SyntheticRun(Path(temporary.name).resolve())
        # A future accidental external command is a test failure, never a real job.
        forbidden = patch("subprocess.run", side_effect=AssertionError("external commands forbidden"))
        forbidden.start()
        self.addCleanup(forbidden.stop)
        no_children = patch("subprocess.Popen", side_effect=AssertionError("child processes forbidden"))
        no_children.start()
        self.addCleanup(no_children.stop)

    def regression(self, plan=None, suite=None):
        return next(row for row in (plan or self.fixture.plan)["rows"]
                    if row["task_kind"] == "regression" and (suite is None or row["suite"] == suite))

    def test_digest_matches_upstream_not_compact_json(self):
        value = {"z": [3, 2], "a": "x"}
        self.assertEqual(prepare.object_digest(value), upstream_digest(value))
        compact = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertNotEqual(prepare.object_digest(value), compact)

    def test_select_exact_224_quotas_order_and_deepcopy(self):
        plan = self.fixture.plan
        before = deepcopy(plan)
        rows = prepare.select_regression_rows(plan)
        self.assertEqual(len(rows), 224)
        self.assertEqual(Counter(row["suite"] for row in rows), REG_COUNTS)
        self.assertEqual([row["row_id"] for row in rows],
                         [row["row_id"] for row in plan["rows"] if row["task_kind"] == "regression"])
        rows[0]["input_files"][0]["size_bytes"] = 99999
        self.assertEqual(plan, before)

    def test_plan_id_tampering_is_rejected(self):
        plan = deepcopy(self.fixture.plan)
        plan["plan_id"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "plan_id"):
            prepare.select_regression_rows(plan)

    def test_resigned_missing_membership_and_wrong_suite_quotas_rejected(self):
        cases = []
        missing = deepcopy(self.fixture.plan)
        missing["rows"].pop()
        cases.append((missing, "standard681"))
        wrong_reg = deepcopy(self.fixture.plan)
        self.regression(wrong_reg)["suite"] = "BCCO"
        cases.append((wrong_reg, "suite counts"))
        wrong_class = deepcopy(self.fixture.plan)
        wrong_class["rows"][0]["suite"] = "BCCO"
        cases.append((wrong_class, "standard457"))
        for plan, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                prepare.select_regression_rows(seal(plan))

    def test_resigned_duplicate_row_or_dataset_rejected(self):
        for field, message in (("row_id", "duplicate source membership"), ("dataset", "duplicate regression dataset")):
            plan = deepcopy(self.fixture.plan)
            rows = [row for row in plan["rows"] if row["task_kind"] == "regression"]
            rows[1][field] = rows[0][field]
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, message):
                prepare.select_regression_rows(seal(plan))

    def test_resigned_blocked_data_and_row_rejected(self):
        for top_level in (True, False):
            plan = deepcopy(self.fixture.plan)
            if top_level:
                plan["data_blocked"] = [{"row_id": "regression::blocked", "reason": "missing"}]
            else:
                self.regression(plan)["blocked_reason"] = "missing official data"
            with self.subTest(top_level=top_level), self.assertRaisesRegex(RuntimeError, "blocked"):
                prepare.select_regression_rows(seal(plan))

    def test_pfn_explicit_target_split_and_frozen_membership_required(self):
        for case in ("target", "split", "relative", "unfrozen"):
            plan = deepcopy(self.fixture.plan)
            row = self.regression(plan, "PFN")
            if case == "target":
                row.pop("target_feature")
            elif case == "split":
                row.pop("official_split_path")
            elif case == "relative":
                row["official_split_path"] = "split.arff"
            else:
                row["input_files"] = row["input_files"][:1]
            row["input_fingerprint"] = fingerprint(row)
            with self.subTest(case=case), self.assertRaisesRegex(RuntimeError, "PFN"):
                prepare.select_regression_rows(seal(plan))

    def test_input_fingerprint_and_metadata_shape_rejected(self):
        for case, message in (("digest", "input_fingerprint"), ("size", "metadata"), ("duplicate", "duplicate input")):
            plan = deepcopy(self.fixture.plan)
            row = self.regression(plan)
            if case == "digest":
                row["input_fingerprint"] = "0" * 64
            elif case == "size":
                row["input_files"][0]["size_bytes"] = 0
            else:
                row["input_files"].append(deepcopy(row["input_files"][0]))
            with self.subTest(case=case), self.assertRaisesRegex(RuntimeError, message):
                prepare.select_regression_rows(seal(plan))

    def test_build_exact_steps_and_hash_checkpoint_bytes_once(self):
        original_open = Path.open
        checkpoint_reads = Counter()
        def tracked_open(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if path.parent == self.fixture.checkpoints and path.suffix == ".ckpt" and mode == "rb":
                checkpoint_reads[path.name] += 1
            return original_open(path, *args, **kwargs)
        original_plan = self.fixture.plan_path.read_bytes()
        with patch.object(Path, "open", new=tracked_open):
            manifest = self.fixture.build()
        self.assertEqual(checkpoint_reads, {f"step-{step}.ckpt": 1 for step in STEPS})
        self.assertEqual([row["step"] for row in manifest["checkpoints"]], list(STEPS))
        self.assertEqual([row["finetune_step"] for row in manifest["checkpoints"]], list(range(1, 51)))
        self.assertEqual((manifest["membership_count"], manifest["checkpoint_count"], manifest["evaluation_unit_count"]),
                         (224, 50, 11200))
        self.assertEqual(manifest["suite_counts"], REG_COUNTS)
        self.assertEqual(manifest["baseline_checkpoint_count"], 0)
        self.assertFalse(manifest["models_loaded_at_prepare"])
        self.assertFalse(manifest["data_files_copied"])
        self.assertEqual(manifest["results_created_at_prepare"], 0)
        self.assertEqual(manifest["manifest_id"], upstream_digest({k: v for k, v in manifest.items() if k != "manifest_id"}))
        self.assertEqual(self.fixture.plan_path.read_bytes(), original_plan)

    def test_checkpoint_set_rejects_missing_or_extra_step(self):
        missing = self.fixture.checkpoints / "step-22176.ckpt"
        missing.rename(missing.with_suffix(".not-a-checkpoint"))
        with self.assertRaisesRegex(RuntimeError, "checkpoint set"):
            self.fixture.build()
        missing.with_suffix(".not-a-checkpoint").rename(missing)
        self.fixture.write("finetune/step-22175.ckpt", b"unexpected source inside ft50 directory")
        with self.assertRaisesRegex(RuntimeError, "checkpoint set"):
            self.fixture.build()

    def test_checkpoint_size_and_completion_membership_mismatch_rejected(self):
        complete = deepcopy(self.fixture.complete)
        complete["checkpoint_sizes"]["step-22176.ckpt"] += 1
        self.fixture.write_json("finetune/complete.json", complete)
        with self.assertRaisesRegex(RuntimeError, "checkpoint size"):
            self.fixture.build()
        complete["checkpoint_sizes"].pop("step-22176.ckpt")
        self.fixture.write_json("finetune/complete.json", complete)
        with self.assertRaisesRegex(RuntimeError, "completion checkpoint membership"):
            self.fixture.build()

    def test_contract_and_completion_require_exact_50_updates(self):
        cases = [("contract", "optimizer_updates", 49), ("contract", "shared_model", False),
                 ("contract", "inspect_only", True), ("contract", "validation_or_test_loaded", True),
                 ("complete", "checkpoint_count", 49), ("complete", "final_step", 22224),
                 ("complete", "complete", False)]
        for which, key, value in cases:
            with self.subTest(which=which, key=key):
                changed = deepcopy(getattr(self.fixture, which))
                changed[key] = value
                self.fixture.write_json(f"finetune/{which}.json", changed)
                try:
                    with self.assertRaisesRegex(RuntimeError, "contract|finetuning run"):
                        self.fixture.build()
                finally:
                    self.fixture.write_json(f"finetune/{which}.json", getattr(self.fixture, which))

    def test_contract_cannot_be_borrowed_from_another_directory(self):
        elsewhere = self.fixture.write_json("another-run/contract.json", self.fixture.contract)
        with self.assertRaisesRegex(RuntimeError, "alongside"):
            self.fixture.build(finetune_contract=elsewhere)

    def test_optional_baseline_is_true_source_and_not_a_51st_finetune(self):
        manifest = self.fixture.build(source_checkpoint=self.fixture.source_checkpoint)
        self.assertEqual((manifest["checkpoint_count"], manifest["finetuned_checkpoint_count"],
                          manifest["baseline_checkpoint_count"], manifest["evaluation_unit_count"]), (51, 50, 1, 11424))
        self.assertEqual(manifest["finetuned_evaluation_unit_count"], 11200)
        self.assertEqual(manifest["checkpoints"][0]["step"], 22175)
        self.assertEqual(manifest["checkpoints"][0]["kind"], "source_baseline")
        wrong = self.fixture.write("baseline/not-the-source.ckpt", b"different checkpoint")
        with self.assertRaisesRegex(RuntimeError, "true step22175"):
            self.fixture.build(source_checkpoint=wrong)

    def test_changed_input_metadata_is_rejected(self):
        self.fixture.raw.write_bytes(b"changed length")
        with self.assertRaisesRegex(RuntimeError, "input metadata changed"):
            self.fixture.build()

    def test_checkpoint_mutation_after_hashing_is_rejected(self):
        original_identity = prepare.file_identity
        target = self.fixture.checkpoints / "step-22176.ckpt"
        def change_after_hash(path, **kwargs):
            result = original_identity(path, **kwargs)
            if Path(path) == target:
                target.write_bytes(b"changed after immutable hash; now a different size")
            return result
        with patch.object(prepare, "file_identity", side_effect=change_after_hash):
            with self.assertRaisesRegex(RuntimeError, "checkpoint changed after hashing"):
                self.fixture.build()

    def test_vendor_digest_and_loader_protocol_rejected(self):
        helper = self.fixture.vendor / "standard_loader.py"
        before = helper.read_bytes()
        helper.write_bytes(before + b"# changed\n")
        with self.assertRaisesRegex(RuntimeError, "vendor code changed"):
            self.fixture.build()
        helper.write_bytes(before)
        self.fixture.eval_data.write_bytes(b"PROTOCOL = 'wrong'\ndef load(row):\n    pass\n")
        with self.assertRaisesRegex(RuntimeError, "data-loader API/protocol"):
            self.fixture.build()

    def test_target_transform_requires_named_class(self):
        wrong = self.fixture.write("helpers/wrong_transform.py", b"class DifferentTransform:\n    pass\n")
        with self.assertRaisesRegex(RuntimeError, "RegressionTargetTransform"):
            self.fixture.build(target_transform=wrong)

    def test_empty_native_package_initializer_allowed_but_empty_checkpoint_rejected(self):
        initializer = self.fixture.native / "tabicl/__init__.py"
        initializer.write_bytes(b"")
        manifest = self.fixture.build()
        record = manifest["native_source"]["files"]["tabicl/__init__.py"]
        self.assertEqual(record["size_bytes"], 0)
        self.assertEqual(record["sha256"], hashlib.sha256(b"").hexdigest())
        (self.fixture.checkpoints / "step-22176.ckpt").write_bytes(b"")
        with self.assertRaisesRegex(RuntimeError, "empty immutable input"):
            self.fixture.build()

    def test_verify_data_uses_synthetic_helper_for_all_224(self):
        self.fixture.eval_data.write_bytes(self.fixture.loader_code(diagnostic=True))
        manifest = self.fixture.build(verify_data=True)
        self.assertIn("all224_diagnostic_cpu_load_passed", manifest["data_validation"])
        self.assertEqual(len([row for row in manifest["rows"] if row["diagnostic_data_audit"]["protocol_validation"]]), 224)
        self.assertFalse(manifest["models_loaded_at_prepare"])

    def test_file_identity_hashes_nonempty_file_and_rejects_symlink(self):
        path = self.fixture.source_checkpoint
        record = prepare.file_identity(path)
        self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(record["size_bytes"], path.stat().st_size)
        empty = self.fixture.write("empty.bin", b"")
        with self.assertRaisesRegex(RuntimeError, "empty"):
            prepare.file_identity(empty)
        link = self.fixture.root / "link.ckpt"
        link.symlink_to(path)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            prepare.file_identity(link)

    def test_publish_is_no_clobber_even_for_identical_manifest(self):
        manifest = self.fixture.build()
        output = self.fixture.root / "published/manifest.json"
        receipt = prepare.publish_manifest(output, manifest)
        original = output.read_bytes()
        self.assertEqual(json.loads(original), manifest)
        self.assertEqual(receipt["manifest_sha256"], hashlib.sha256(original).hexdigest())
        for candidate in (manifest, {**manifest, "membership_count": 1}):
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                prepare.publish_manifest(output, candidate)
            self.assertEqual(output.read_bytes(), original)

    def test_publish_rejects_tampering_and_dangling_symlink(self):
        manifest = self.fixture.build()
        manifest["checkpoint_count"] = 1
        output = self.fixture.root / "invalid-manifest.json"
        with self.assertRaisesRegex(RuntimeError, "content identity"):
            prepare.publish_manifest(output, manifest)
        self.assertFalse(output.exists())
        output.symlink_to(self.fixture.root / "nonexistent")
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            prepare.publish_manifest(output, manifest)
        self.assertTrue(output.is_symlink())

    def test_publish_race_preserves_competing_file(self):
        manifest = self.fixture.build()
        output = self.fixture.root / "publication-race.json"
        def publish_first(source, destination):
            Path(destination).write_bytes(b"competing publisher must survive")
            raise FileExistsError(str(destination))
        with patch.object(prepare.os, "link", side_effect=publish_first), self.assertRaises(FileExistsError):
            prepare.publish_manifest(output, manifest)
        self.assertEqual(output.read_bytes(), b"competing publisher must survive")
        self.assertEqual(list(self.fixture.root.glob(".publication-race.json.*")), [])


if __name__ == "__main__":
    unittest.main()
