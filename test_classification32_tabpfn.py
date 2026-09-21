"""CPU-only adversarial contribution tests; no weights, training, or GPU."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

import classification32_tabpfn as adapter


class OOM(RuntimeError):
    pass


@dataclass
class Config:
    member: int
    class_permutation: object


class FakeEstimator:
    def __init__(self, mode="normal"):
        self.mode = mode
        self.n_estimators, self.random_state, self.fit_mode = 32, 0, "fit_preprocessors"
        self.use_autocast_, self.forced_inference_dtype_ = False, "float32"
        self.softmax_temperature = 0.9
        self.average_before_softmax = False
        self.balance_probabilities = False
        self.fit_calls = 0

    def fit(self, xs, ys):
        self.fit_calls += 1
        self.classes_ = np.unique(ys)
        self.n_estimators_ = 31 if self.mode == "short_fit" else 32
        self.ensemble_configs_ = [Config(i, np.roll(np.arange(len(self.classes_)), i))
                                  for i in range(self.n_estimators_)]
        members = [SimpleNamespace(config=c) for c in self.ensemble_configs_]
        self.executor_ = SimpleNamespace(
            ensemble_members=members,
            ensemble_preprocessor=SimpleNamespace(pipeline_seeds=np.arange(len(members)) + 101),
            iter_outputs=self.iter_outputs)
        return self

    def iter_outputs(self, X, *, autocast, task_type):
        count = 31 if self.mode == "short_predict" else 33 if self.mode == "extra" else 32
        for i in range(count):
            if self.mode == "oom" and len(X) > 128 and i == 3:
                raise OOM("Test partial native query OOM")
            config = self.ensemble_configs_[0 if self.mode == "duplicate" else i % 32]
            rows = len(X) - 1 if self.mode == "missing_rows" else len(X)
            logits = (np.arange(rows * 10, dtype=np.float32).reshape(rows, 10) * 0.01 + i * 0.02)
            yield logits, config

    def _average_across_estimators(self, tensors):
        return tensors.mean(axis=0)

    def logits_to_probabilities(self, raw_logits):
        scaled = raw_logits / self.softmax_temperature
        exps = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
        probabilities = exps / exps.sum(axis=-1, keepdims=True)
        if self.mode == "skip_native_mean":
            return probabilities.mean(axis=0)
        return self._average_across_estimators(probabilities)

    def predict_proba(self, query):
        outputs = []
        for output, config in self.executor_.iter_outputs(query, autocast=False, task_type="multiclass"):
            aligned = output[:, config.class_permutation]
            outputs.append(aligned[:, None, :])
        stacked = np.stack(outputs)
        if self.mode == "replace_before_aggregate":
            stacked[-1] = stacked[0]
        elif self.mode == "omit_before_aggregate":
            stacked = stacked[:-1]
        probability = self.logits_to_probabilities(stacked).squeeze(1)
        return probability / probability.sum(axis=1, keepdims=True)


@unittest.skipIf(np is None, "NumPy needed for CPU contribution tests")
class TabPFN32Tests(unittest.TestCase):
    def setUp(self):
        self.arrays = {"X_train": np.arange(36, dtype=np.float32).reshape(12, 3),
                       "y_train": np.tile(np.arange(3, dtype=np.int64), 4),
                       "X_test": np.arange(15, dtype=np.float32).reshape(5, 3)}
        self.torch = SimpleNamespace(float32="float32", OutOfMemoryError=OOM,
                                     cuda=SimpleNamespace(empty_cache=lambda: None))
        self.model_config = {"checkpoint_path": "/unused/model.ckpt", "checkpoint_sha256": "f" * 64,
                             "hierarchy_helper_path": "/unused/hierarchy.py", "source_root": "/unused/source",
                             "source_commit": adapter.SOURCE_COMMIT,
                             "hierarchy_helper_sha256": adapter.HIERARCHY_SHA256}

    def fitted(self, mode="normal"):
        est = FakeEstimator(mode)
        audit = adapter.Native32Audit(est, self.torch)
        audit.fit(self.arrays["X_train"], self.arrays["y_train"])
        return est, audit

    def test_native32_output_and_mean_are_observed_without_numerical_change(self):
        est, audit = self.fitted()
        expected = est.predict_proba(self.arrays["X_test"])
        probability, _ = audit.predict_full(self.arrays["X_test"])
        np.testing.assert_array_equal(probability, expected)
        self.assertTrue(audit.nodes[0]["actual32_verified"])
        self.assertEqual(len(audit.nodes[0]["member_audits"]), 32)
        chunk = audit.nodes[0]["successful_prediction_chunks"][0]
        self.assertEqual(len(chunk["members"]), 32)
        self.assertTrue(chunk["native_aggregation_verified"])

    def test_nominal32_with_only31_fitted_fails(self):
        with self.assertRaisesRegex(RuntimeError, "generate32"):
            self.fitted("short_fit")

    def test_nominal32_with_missing_extra_duplicate_or_incomplete_outputs_fails(self):
        for mode in ("short_predict", "extra", "duplicate", "missing_rows"):
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                _, audit = self.fitted(mode)
                audit.predict_full(self.arrays["X_test"])

    def test_all_outputs_must_reach_actual_native_reduction(self):
        for mode in ("replace_before_aggregate", "omit_before_aggregate", "skip_native_mean"):
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                _, audit = self.fitted(mode)
                audit.predict_full(self.arrays["X_test"])

    def test_observers_restored_after_failure(self):
        est, audit = self.fitted("replace_before_aggregate")
        before = (est.executor_.iter_outputs, est.logits_to_probabilities, est._average_across_estimators)
        with self.assertRaises(RuntimeError):
            audit.predict_chunk(self.arrays["X_test"])
        self.assertEqual(before, (est.executor_.iter_outputs, est.logits_to_probabilities, est._average_across_estimators))

    def test_partial_oom_excluded_and_native_chunk_backoff_preserved(self):
        _, audit = self.fitted("oom")
        query = np.arange(600, dtype=np.float32).reshape(200, 3)
        probability, chunk = audit.predict_full(query)
        self.assertEqual(probability.shape, (200, 3))
        self.assertEqual(chunk, 128)
        node = audit.nodes[0]
        self.assertEqual(len(node["discarded_oom_attempts"]), 1)
        self.assertFalse(node["discarded_oom_attempts"][0]["included_in_final"])
        self.assertEqual([(c["start"], c["stop"]) for c in node["successful_prediction_chunks"]], [(0, 128), (128, 200)])
        self.assertEqual(node["minimum_contributions_per_test_row"], 32)

    def test_same_node_full_prediction_cannot_be_repeated(self):
        _, audit = self.fitted()
        audit.predict_full(self.arrays["X_test"])
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            audit.predict_full(self.arrays["X_test"])

    def test_hierarchy_retains32_contributors_per_original_node(self):
        source = Path(__file__).resolve().parents[2] / "tabpfn_v2_v25_v3_exact178_20260820/hierarchical_tabpfn.py"
        self.assertEqual(adapter.file_digest(source), adapter.HIERARCHY_SHA256)
        spec = importlib.util.spec_from_file_location("_test_original_tabpfn_hierarchy", source)
        hierarchy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hierarchy)
        arrays = {"X_train": np.arange(66, dtype=np.float32).reshape(22, 3),
                  "y_train": np.tile(np.arange(11, dtype=np.int64), 2), "X_test": self.arrays["X_test"]}
        with patch.object(adapter, "load_runtime", return_value=(FakeEstimator(), hierarchy, self.torch, {})):
            probability, audit = adapter.predict("tabpfn2", arrays, self.model_config)
        self.assertEqual(probability.shape, (5, 11))
        self.assertEqual(audit["node_count"], 3)
        self.assertEqual(audit["actual_members_per_test_row"], 32)
        self.assertTrue(all(n["actual32_verified"] for n in audit["ensemble_audits"]))
        np.testing.assert_allclose(probability.sum(axis=1), 1)

    def test_test_labels_are_never_requested(self):
        class ForbiddenLabels(dict):
            def __getitem__(self, key):
                if key == "y_test":
                    raise AssertionError("Test labels consumed")
                return super().__getitem__(key)
        values = ForbiddenLabels(self.arrays, y_test="must not read")
        adapter.validate_inputs(values)

    def test_input_contract_and_configuration_aliases(self):
        config = adapter.normalize_config(self.model_config)
        self.assertEqual(config["model_path"], self.model_config["checkpoint_path"])
        with self.assertRaisesRegex(RuntimeError, "Conflicting"):
            adapter.normalize_config({**self.model_config, "model_path": "/different"})
        with self.assertRaisesRegex(RuntimeError, "hierarchy"):
            adapter.normalize_config({**self.model_config, "hierarchy_helper_sha256": "0" * 64})
        for key in ("X_train", "X_test"):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "float32"):
                adapter.validate_inputs({**self.arrays, key: self.arrays[key].astype(np.float64)})


if __name__ == "__main__":
    unittest.main()
