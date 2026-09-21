"""CPU-only tests for strict-eight original-Mitra contribution accounting."""
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
import pfn_mitra8_one as worker


def cfg():
    return SimpleNamespace(seed=42, hyperparams={"max_epochs": 0, "max_samples_support": 8192,
        "max_samples_query": 1024, "precision": "bfloat16", "dim_output": 1,
        "n_ensembles": 8, "grad_scaler_enabled": False, "random_mirror_regression": True,
        "random_mirror_x": True, "shuffle_classes": False, "shuffle_features": False,
        "use_random_transforms": False})


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
    def any(self):
        return self.values.any()
    def detach(self):
        return self
    def float(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.values


class Model:
    def __init__(self):
        self.hooks = []
        self.dim_output, self.use_flash_attn = 1, False
        self.output_ndim, self.output_channels = 3, 1
    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))
    def run(self, xs, ys, query, value):
        args = (Tensor(xs[None]), Tensor(ys[None]), Tensor(query[None]),
                Tensor(np.zeros((1, xs.shape[1]), dtype=bool)),
                Tensor(np.zeros((1, len(xs)), dtype=bool)),
                Tensor(np.zeros((1, len(query)), dtype=bool)))
        shape = (1, len(query)) if self.output_ndim == 2 else (1, len(query), self.output_channels)
        result = Tensor(np.full(shape, value, dtype=np.float32))
        for hook in self.hooks:
            hook(self, args, result)


class Trainer:
    def __init__(self, member, rng, support_rows):
        self.member, self.rng, self.model, self.cfg = member, rng, Model(), cfg()
        self.pfn_rng_before_init = f"distinct-init-state-{member}"
        self.pfn_rng_after_fit = f"distinct-after-fit-{member}"
        self.pfn_preprocessing_rng_before_fit = f"distinct-preprocessor-state-{member}"
        self.pfn_native_preprocessor = {"feature_mirror": [[1, -1, 1]], "regression_mirror": bool(member % 2)}
        self.pfn_fit_calls, self.pfn_support_rows, self.pfn_optimizer_step_attempts = 1, support_rows, 0
        self.extra_forward, self.omit_last, self.impute_query = False, False, False
        self.coincident_native_context = False
        self.prediction_shape = "column"
    def predict(self, xs, ys, query):
        if self.impute_query:
            query[np.isnan(query)] = 0
        actual_query = query[:-1] if self.omit_last else query
        for start in range(0, len(actual_query), 1024):
            indices = self.rng.choice(len(xs), min(len(xs), 8192), replace=False)
            if self.coincident_native_context:
                # Independently executed native draws can realize the same
                # effective model context. Simulate that outcome, not reuse of
                # a cached member prediction; RNG still advances each time.
                indices = np.arange(min(len(xs), 8192))
            chunk = actual_query[start:start + 1024]
            self.model.run(xs[indices], ys[indices], chunk, self.member)
            if self.extra_forward:
                self.model.run(xs[indices], ys[indices], chunk, self.member)
        shape = ((len(actual_query),) if self.prediction_shape == "vector" else
                 (len(actual_query), 2 if self.prediction_shape == "multi" else 1))
        return np.full(shape, self.member, dtype=np.float32)


class Estimator:
    def __init__(self, support_rows=100):
        self.n_estimators, self.fine_tune, self.fine_tune_steps, self.seed = 8, False, 0, 42
        self.X = np.arange(support_rows * 3, dtype=np.float32).reshape(support_rows, 3)
        self.y = np.arange(support_rows, dtype=np.float64)
        rng = np.random.RandomState(42)
        self.trainers = [Trainer(member, rng, support_rows) for member in range(8)]
        self.skip_last, self.bad_average = False, False
    def predict(self, query):
        trainers = self.trainers[:-1] if self.skip_last else self.trainers
        outputs = [trainer.predict(self.X, self.y, query) for trainer in trainers]
        self.last_native_member_shapes = [tuple(output.shape) for output in outputs]
        return sum(outputs) / len(outputs) + int(self.bad_average)


class StrictEightTests(unittest.TestCase):
    def query(self):
        return np.arange(21, dtype=np.float32).reshape(7, 3)

    def test_exact_eight_actual_forwards_members_and_native_mean(self):
        estimator = Estimator()
        prediction, audit = worker.predict_actual8(estimator, self.query(), np)
        np.testing.assert_array_equal(prediction, np.full(7, 3.5, dtype=np.float32))
        self.assertTrue(audit["actual8_verified"])
        self.assertTrue(audit["all_test_rows_covered"])
        self.assertEqual(audit["actual_ensemble_count"], 8)
        self.assertEqual(audit["total_test_model_forward_calls"], 8)
        self.assertEqual(audit["minimum_contributions_per_test_row"], 8)
        self.assertEqual(audit["maximum_contributions_per_test_row"], 8)
        self.assertEqual(len(audit["member_audits"]), 8)
        self.assertTrue(all(not t.model.hooks for t in estimator.trainers))

    def test_native_query_nan_imputation_does_not_look_like_input_mismatch(self):
        estimator = Estimator()
        for trainer in estimator.trainers:
            trainer.impute_query = True
        query = self.query()
        query[0, 1] = np.nan
        _, audit = worker.predict_actual8(estimator, query, np)
        self.assertTrue(audit["actual8_verified"])
        self.assertNotEqual(audit["member_audits"][0]["query_entry_sha256"],
                            audit["member_audits"][1]["query_entry_sha256"])

    def test_scalar_head_2d_and_3d_preserve_vector_and_column_predict_shapes(self):
        for dimensions in (2, 3):
            for prediction_shape in ("vector", "column"):
                estimator = Estimator()
                for trainer in estimator.trainers:
                    trainer.model.output_ndim = dimensions
                    trainer.prediction_shape = prediction_shape
                with self.subTest(dimensions=dimensions, prediction_shape=prediction_shape):
                    prediction, audit = worker.predict_actual8(estimator, self.query(), np)
                    expected_prediction_shape = (7,) if prediction_shape == "vector" else (7, 1)
                    expected_forward_shape = [1, 7] if dimensions == 2 else [1, 7, 1]
                    self.assertEqual(estimator.last_native_member_shapes, [expected_prediction_shape] * 8)
                    for member in audit["member_audits"]:
                        self.assertEqual(member["native_prediction_shape"], list(expected_prediction_shape))
                        self.assertEqual(member["observed_model_output_shapes"], [expected_forward_shape])
                    np.testing.assert_array_equal(prediction, np.full(7, 3.5, dtype=np.float32))

    def test_multioutput_model_head_rejected(self):
        estimator = Estimator()
        estimator.trainers[3].model.output_channels = 2
        with self.assertRaisesRegex(RuntimeError, "scalar-head output shape"):
            worker.predict_actual8(estimator, self.query(), np)

    def test_multioutput_trainer_prediction_rejected(self):
        estimator = Estimator()
        estimator.trainers[3].prediction_shape = "multi"
        with self.assertRaisesRegex(RuntimeError, "Invalid native member prediction"):
            worker.predict_actual8(estimator, self.query(), np)

    def test_eight_executed_members_allow_naturally_coincident_contexts(self):
        estimator = Estimator()
        for trainer in estimator.trainers:
            trainer.coincident_native_context = True
        prediction, audit = worker.predict_actual8(estimator, self.query(), np)
        self.assertTrue(audit["actual8_verified"])
        self.assertEqual(audit["distinct_native_forward_contexts"], 1)
        self.assertEqual(audit["total_test_model_forward_calls"], 8)
        self.assertEqual(audit["minimum_contributions_per_test_row"], 8)
        self.assertEqual(audit["maximum_contributions_per_test_row"], 8)
        self.assertEqual(len({r["native_rng_before_predict_sha256"] for r in audit["member_audits"]}), 8)
        np.testing.assert_array_equal(prediction, np.full(7, 3.5, dtype=np.float32))

    def test_context_cap_9000_to_8192_preserved(self):
        _, audit = worker.predict_actual8(Estimator(support_rows=9000), self.query(), np)
        self.assertTrue(all(r["actual_support_rows"] == 8192 for r in audit["member_audits"]))

    def test_duplicate_trainer_objects_rejected(self):
        estimator = Estimator()
        estimator.trainers[-1] = estimator.trainers[0]
        with self.assertRaisesRegex(RuntimeError, "distinct native trainers"):
            worker.verify_fitted_ensemble(estimator, len(estimator.y))

    def test_independently_reseeded_rng_copies_rejected(self):
        estimator = Estimator()
        estimator.trainers[1].rng = np.random.RandomState(42)
        with self.assertRaisesRegex(RuntimeError, "share one advancing RNG"):
            worker.verify_fitted_ensemble(estimator, len(estimator.y))

    def test_identical_member_rng_entry_states_rejected(self):
        for attribute in ("pfn_rng_before_init", "pfn_preprocessing_rng_before_fit"):
            estimator = Estimator()
            for trainer in estimator.trainers:
                setattr(trainer, attribute, "reseeded-same")
            with self.subTest(attribute=attribute), self.assertRaisesRegex(RuntimeError, "advance distinctly"):
                worker.verify_fitted_ensemble(estimator, len(estimator.y))

    def test_seven_member_contributions_rejected(self):
        estimator = Estimator()
        estimator.skip_last = True
        with self.assertRaisesRegex(RuntimeError, "exactly one contribution"):
            worker.predict_actual8(estimator, self.query(), np)

    def test_extra_hidden_model_pass_rejected_and_hooks_removed(self):
        estimator = Estimator()
        estimator.trainers[2].extra_forward = True
        with self.assertRaisesRegex(RuntimeError, "context/query rows|Too many test chunk"):
            worker.predict_actual8(estimator, self.query(), np)
        self.assertTrue(all(not t.model.hooks for t in estimator.trainers))

    def test_omitted_query_row_rejected(self):
        estimator = Estimator()
        estimator.trainers[2].omit_last = True
        with self.assertRaisesRegex(RuntimeError, "context/query rows"):
            worker.predict_actual8(estimator, self.query(), np)

    def test_non_native_aggregation_rejected(self):
        estimator = Estimator()
        estimator.bad_average = True
        with self.assertRaisesRegex(RuntimeError, "arithmetic mean"):
            worker.predict_actual8(estimator, self.query(), np)

    def test_strict_config_seed_cap_precision_and_epoch_guards(self):
        worker.check_cfg8(cfg())
        for name, value in (("n_ensembles", 1), ("max_epochs", 1), ("max_samples_support", 4096),
                             ("max_samples_query", 512), ("precision", "float32"),
                             ("random_mirror_x", False)):
            changed = cfg()
            changed.hyperparams[name] = value
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, name):
                worker.check_cfg8(changed)
        changed = cfg()
        changed.seed = 0
        with self.assertRaisesRegex(RuntimeError, "seed"):
            worker.check_cfg8(changed)

    def test_optimizer_update_any_member_rejected(self):
        estimator = Estimator()
        estimator.trainers[7].pfn_optimizer_step_attempts = 1
        with self.assertRaisesRegex(RuntimeError, "no-update"):
            worker.verify_fitted_ensemble(estimator, len(estimator.y))

    def test_multiple_query_chunks_with_duplicate_rows_covered_exactly_eight(self):
        _, audit = worker.predict_actual8(Estimator(), np.zeros((2053, 3), dtype=np.float32), np)
        self.assertEqual(audit["total_test_model_forward_calls"], 24)
        self.assertEqual(audit["native_query_chunks_per_member"], 3)
        self.assertEqual(audit["minimum_contributions_per_test_row"], 8)
        self.assertEqual(audit["maximum_contributions_per_test_row"], 8)
        for member in audit["member_audits"]:
            self.assertEqual(member["query_chunk_ranges"], [[0, 1024], [1024, 2048], [2048, 2053]])

    def test_empty_test_split_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Empty official"):
            worker.predict_actual8(Estimator(), np.zeros((0, 3)), np)

    def test_reference_and_parity_for_every_original_suite(self):
        manifest_path = HERE.parent / "evaluation_manifest.json"
        root = HERE.parent / "rank_comparison_20260921T171633/results/step-22175"
        manifest = json.loads(manifest_path.read_text())
        for suite in ("talent", "BCCO", "CTR23", "TabArena", "PFN"):
            index = next(row["dataset_index"] for row in manifest["rows"] if row["suite"] == suite)
            man, row, checkpoint = worker.common.load_manifest(manifest_path, 22175, index)
            source = worker.load_source_result224(root / f"row-{index:03d}.json", man, row, checkpoint)
            worker.verify_data_parity224(source["data_audit"], source,
                source["target_transform_source"], source["target_transform"], suite)
            changed = deepcopy(source["data_audit"])
            changed["test_targets_sha256"] = "bad"
            with self.subTest(suite=suite), self.assertRaisesRegex(RuntimeError, "audit differs"):
                worker.verify_data_parity224(changed, source,
                    source["target_transform_source"], source["target_transform"], suite)

    def test_no_overwrite_publication(self):
        with tempfile.TemporaryDirectory(prefix="test-mitra8-") as directory:
            path = Path(directory) / "result.json"
            worker.common.publish_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                worker.common.publish_new(path, {"original": False})

    def test_pin_old_worker_and_frozen_original_runtime(self):
        self.assertEqual(worker.sha256_file(worker.historical.__file__), worker.HISTORICAL_WORKER_SHA256)
        audit = HERE.parent / "pfn28_foundation_comparison/runtime_audit"
        names = {"sklearn_interface": "mitra_sklearn_interface.py", "trainer_finetune": "mitra_trainer_finetune.py",
                 "dataset_finetune": "mitra_dataset_finetune.py", "preprocessor": "mitra_preprocessor.py"}
        for module, expected in worker.RUNTIME_PINS.items():
            self.assertEqual(worker.sha256_file(audit / names[module.rsplit(".", 1)[1]]), expected)


if __name__ == "__main__":
    unittest.main()
