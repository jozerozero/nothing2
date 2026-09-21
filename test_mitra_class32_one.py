"""CPU-only synthetic tests; no real weights, GPU, training, or remote calls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mitra_class32_one as worker


def cfg():
    return SimpleNamespace(seed=0, hyperparams={"n_ensembles": 32, "max_epochs": 0,
        "max_samples_support": 8192, "max_samples_query": 1024, "precision": "bfloat16",
        "dim_output": 10, "grad_scaler_enabled": False, "shuffle_classes": False,
        "shuffle_features": False, "use_random_transforms": False, "random_mirror_x": True})


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
    def any(self):
        return self.values.any()


class FakeModel:
    def __init__(self):
        self.dim_output, self.use_flash_attn, self.hooks = 10, False, []
    def register_forward_hook(self, fn):
        self.hooks.append(fn)
        return SimpleNamespace(remove=lambda: self.hooks.remove(fn))
    def forward(self, xs, ys, xt, logits):
        args = (FakeTensor(xs[None]), FakeTensor(ys[None]), FakeTensor(xt[None]),
            FakeTensor(np.zeros((1, xs.shape[1]), dtype=bool)),
            FakeTensor(np.zeros((1, len(xs)), dtype=bool)), FakeTensor(np.zeros((1, len(xt)), dtype=bool)))
        for fn in self.hooks:
            fn(self, args, FakeTensor(logits[None]))


class FakeTrainer:
    def __init__(self, member, rng, support_rows):
        self.member, self.rng, self.model, self.cfg = member, rng, FakeModel(), cfg()
        self.audit_rng_entry, self.audit_preprocess_rng = f"support-rng-{member}", f"global-rng-{member}"
        self.audit_rng_after_fit = f"after-fit-{member}"
        self.audit_feature_mirror = [[1, -1]]
        self.audit_fit_calls, self.audit_support_rows, self.audit_optimizer_steps = 1, support_rows, 0
        self.double_forward, self.wrong_shape = False, False
    def predict(self, xs, ys, xt):
        indices = self.rng.choice(len(xs), size=min(len(xs), 8192), replace=False)
        logits = np.tile(np.linspace(-1, 1, 10, dtype=np.float32), (len(xt), 1))
        logits[:, self.member % 10] += np.float32(self.member / 32)
        if self.wrong_shape:
            logits = logits[:, :1]
        self.model.forward(xs[indices], ys[indices], xt, logits)
        if self.double_forward:
            self.model.forward(xs[indices], ys[indices], xt, logits)
        return logits


class FakeOOM(RuntimeError):
    pass


TORCH = SimpleNamespace(OutOfMemoryError=FakeOOM, cuda=SimpleNamespace(empty_cache=lambda: None))


class FakeEstimator:
    def __init__(self):
        self.n_estimators, self.seed, self.fine_tune, self.fine_tune_steps = 32, 0, False, 0
        self.trainers = []
        self.skip_last, self.bad_softmax, self.oom_once = False, False, False
        self.fit_count = 0
    def fit(self, xs, ys, X_val, y_val):
        assert xs is X_val and ys is y_val
        self.X, self.y = xs, ys
        rng = np.random.RandomState(0)
        self.trainers = [FakeTrainer(i, rng, len(ys)) for i in range(32)]
        self.fit_count += 1
        return self
    def predict_proba(self, xt):
        if self.oom_once:
            self.oom_once = False
            raise FakeOOM("synthetic query OOM")
        selected = self.trainers[:-1] if self.skip_last else self.trainers
        probs = []
        for trainer in selected:
            logits = trainer.predict(self.X, self.y, xt)[:, :len(np.unique(self.y))]
            probs.append(np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))
        return sum(probs) / len(probs) + float(self.bad_softmax)


def metadata(path):
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": worker.digest_file(path)}


class Class32Tests(unittest.TestCase):
    def arrays(self, classes=3, query_rows=21):
        ys = np.tile(np.arange(classes, dtype=np.int64), 10)
        xs = np.arange(len(ys) * 2, dtype=np.float32).reshape(-1, 2)
        query = np.arange(query_rows * 2, dtype=np.float32).reshape(-1, 2)
        return xs, ys, query

    def fitted(self, classes=3, query_rows=21):
        xs, ys, query = self.arrays(classes, query_rows)
        native = FakeEstimator().fit(xs, ys, X_val=xs, y_val=ys)
        return native, query

    def test_native32_softmax_average_and_class_slicing(self):
        for classes in (2, 3, 10):
            native, query = self.fitted(classes)
            probability, records = worker.audited_probability_chunk(native, query, np)
            self.assertEqual(probability.shape, (len(query), classes))
            np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
            self.assertEqual(len(records), 32)
            self.assertTrue(all(r["model_output_shape"] == [1, len(query), 10] for r in records))
            self.assertTrue(all(r["logits_shape"] == [len(query), 10] for r in records))
            self.assertTrue(all(not t.model.hooks for t in native.trainers))

    def test_native31_partial_oom_fit_cannot_pass(self):
        native, _query = self.fitted()
        native.trainers.pop()
        with self.assertRaisesRegex(RuntimeError, "Exactly32"):
            worker.fitted_member_records(native, np)

    def test_cap_downgrade_and_any_optimizer_update_rejected(self):
        native, _query = self.fitted()
        native.trainers[9].cfg.hyperparams["max_samples_support"] = 4096
        with self.assertRaisesRegex(RuntimeError, "max_samples_support"):
            worker.fitted_member_records(native, np)
        native, _query = self.fitted()
        native.trainers[31].audit_optimizer_steps = 1
        with self.assertRaisesRegex(RuntimeError, "no-update"):
            worker.fitted_member_records(native, np)

    def test_32_distinct_objects_but_shared_native_rng(self):
        native, _query = self.fitted()
        native.trainers[1] = native.trainers[0]
        with self.assertRaisesRegex(RuntimeError, "distinct native"):
            worker.fitted_member_records(native, np)
        native, _query = self.fitted()
        native.trainers[2].rng = np.random.RandomState(0)
        with self.assertRaisesRegex(RuntimeError, "advancing RNG"):
            worker.fitted_member_records(native, np)

    def test_skipped_member_or_duplicate_model_forward_rejected(self):
        native, query = self.fitted()
        native.skip_last = True
        with self.assertRaisesRegex(RuntimeError, "32 independently"):
            worker.audited_probability_chunk(native, query, np)
        native, query = self.fitted()
        native.trainers[3].double_forward = True
        with self.assertRaisesRegex(RuntimeError, "duplicate classifier forward"):
            worker.audited_probability_chunk(native, query, np)

    def test_regression_shape_and_wrong_probability_mean_rejected(self):
        native, query = self.fitted()
        native.trainers[3].wrong_shape = True
        with self.assertRaisesRegex(RuntimeError, "output shape"):
            worker.audited_probability_chunk(native, query, np)
        native, query = self.fitted()
        native.bad_softmax = True
        with self.assertRaisesRegex(RuntimeError, "arithmetic mean"):
            worker.audited_probability_chunk(native, query, np)

    def test_full_query_chunk_coverage_with_duplicate_rows(self):
        xs, ys, _query = self.arrays()
        audited = worker.AuditedClassifier(FakeEstimator(), np, TORCH).fit(xs, ys)
        probability, chunk = audited.predict_full(np.zeros((2053, 2), dtype=np.float32))
        self.assertEqual(probability.shape, (2053, 3))
        self.assertEqual(chunk, 1024)
        audit = audited.ensemble_audits[0]
        self.assertTrue(audit["actual32_verified"])
        self.assertEqual(audit["minimum_contributions_per_test_row"], 32)
        self.assertEqual(audit["maximum_contributions_per_test_row"], 32)
        self.assertEqual([(x["start"], x["stop"]) for x in audit["successful_prediction_chunks"]],
                         [(0, 1024), (1024, 2048), (2048, 2053)])

    def test_historical_query_oom_backoff_does_not_double_count(self):
        xs, ys, query = self.arrays(query_rows=1500)
        native = FakeEstimator()
        native.oom_once = True
        audited = worker.AuditedClassifier(native, np, TORCH).fit(xs, ys)
        _, chunk = audited.predict_full(query)
        self.assertEqual(chunk, 512)
        audit = audited.ensemble_audits[0]
        self.assertEqual(len(audit["discarded_oom_attempts"]), 1)
        self.assertEqual(audit["minimum_contributions_per_test_row"], 32)
        self.assertEqual(audit["maximum_contributions_per_test_row"], 32)

    def test_unchanged_hierarchy_has32_members_at_every_node(self):
        path = HERE.parents[1] / "mitra_all_classification_regression_20260823_v3/hierarchical_mitra.py"
        helper = worker.common.import_path("_test_class32_hierarchy", path)
        xs, ys, query = self.arrays(classes=12, query_rows=15)
        audited = worker.AuditedClassifier(FakeEstimator(), np, TORCH)
        probability, tree = helper.hierarchical_predict_proba(audited, xs, ys, query,
            branch_factor=10, predict_fn=lambda fitted, query: fitted.predict_full(query))
        self.assertEqual(probability.shape, (15, 12))
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        self.assertEqual(tree["nodes_fitted"], 3)
        self.assertEqual(len(audited.ensemble_audits), 3)
        self.assertTrue(all(a["actual_ensemble_count"] == 32 and a["actual32_verified"]
                            and a["test_rows"] == len(query) for a in audited.ensemble_audits))

    def test_config32_seed0_no_training_fixed_recipe(self):
        worker.check_cfg(cfg())
        for key, value in (("n_ensembles", 8), ("max_epochs", 1), ("dim_output", 1),
                           ("precision", "float32"), ("shuffle_classes", True)):
            changed = cfg()
            changed.hyperparams[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                worker.check_cfg(changed)
        changed = cfg()
        changed.seed = 42
        with self.assertRaisesRegex(RuntimeError, "seed0"):
            worker.check_cfg(changed)

    def test_cache_numeric_loading_and_object_rejection(self):
        with tempfile.TemporaryDirectory(prefix="test-mitra-class32-cache-") as directory:
            path = Path(directory) / "cache.npz"
            xs, ys, xt = self.arrays(classes=3, query_rows=6)
            np.savez(path, X_train=xs, y_train=ys[:, None], X_test=xt, y_test=np.arange(6) % 3)
            record = metadata(path)
            row = {"cache": record, "input_fingerprint": worker.common.object_digest(record),
                   "train_rows": len(ys), "test_rows": 6, "features": 2, "classes": 3}
            result = worker.load_cache(row, np)
            self.assertEqual(result[1].shape, ys.shape)
            self.assertTrue(result[-1]["full_test_split"])
            np.savez(path, X_train=xs.astype(object), y_train=ys, X_test=xt, y_test=np.arange(6) % 3)
            row["cache"] = metadata(path)
            with self.assertRaises(ValueError):
                worker.load_cache(row, np)

    def test_plan_seal_and_exact_historical_cache_binding(self):
        with tempfile.TemporaryDirectory(prefix="test-mitra-class32-plan-") as directory:
            root = Path(directory)
            cache, weights, benchmark = root / "cache.npz", root / "weights.json", root / "benchmark.json"
            cache.write_bytes(b"metadata-only cache")
            weights.write_text("{}")
            rows = [{"dataset_index": i, "dataset": f"test{i}", "suite": "talent", "cache": metadata(cache)} for i in range(457)]
            for row in rows:
                row["input_fingerprint"] = worker.common.object_digest(row["cache"])
            benchmark.write_text(json.dumps({"complete": True, "rows": [
                {"dataset": row["dataset"], "suite": "talent", "cache_path": str(cache)} for row in rows]}))
            plan = {"membership_count": 457, "rows": rows, "n_estimators": 32, "seed": 0,
                    "weights_manifest": metadata(weights), "class_benchmark_manifest": metadata(benchmark)}
            plan["manifest_id"] = worker.common.object_digest(plan)
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            self.assertEqual(worker.load_plan(path, 243)[1]["dataset"], "test243")
            plan["seed"] = 42
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(RuntimeError, "content identity"):
                worker.load_plan(path, 243)

    def test_no_result_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="test-mitra-class32-publish-") as directory:
            path = Path(directory) / "row.json"
            worker.common.publish_new(path, {"old": True})
            with self.assertRaises(FileExistsError):
                worker.common.publish_new(path, {"old": False})

    def test_existing_helpers_unchanged(self):
        self.assertEqual(worker.digest_file(worker.historical.__file__), worker.HISTORICAL_WORKER_SHA256)
        path = HERE.parents[1] / "mitra_all_classification_regression_20260823_v3/hierarchical_mitra.py"
        self.assertEqual(worker.digest_file(path), worker.HIERARCHY_SHA256)


if __name__ == "__main__":
    unittest.main()
