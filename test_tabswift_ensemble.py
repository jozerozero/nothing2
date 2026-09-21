"""CPU tests execute pinned native generation/transform/aggregation via AST.

No Torch/sklearn/checkpoints are needed: model forwards are deterministic NumPy
fixtures, while native configuration deduction and final predict_proba code are
read directly from the official pinned source checkout rather than reimplemented.
"""
import ast
from collections import OrderedDict
from contextlib import redirect_stdout
from copy import deepcopy
import io
import itertools
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import List, Optional
import unittest

import numpy as np

import tabswift_ensemble as audit

OFFICIAL_ROOT = Path(os.environ.get("TABSWIFT_OFFICIAL_ROOT", str(
    Path(__file__).resolve().parents[2] / "tabswift_standard681_20260922_v1/official")))
SOURCE = OFFICIAL_ROOT / "TALENT/model/lib/tabswift"


def source_method(filename, classname, method):
    tree = ast.parse((SOURCE / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
    function = deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method))
    function.decorator_list = []
    namespace = dict(np=np, random=random, itertools=itertools, OrderedDict=OrderedDict, deepcopy=deepcopy,
                     check_is_fitted=lambda *a: None, OLD_SKLEARN=True, FeatureShuffler=FeatureShuffler)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(SOURCE / filename), "exec"), namespace)
    return namespace[method]


tree = ast.parse((SOURCE / "preprocessing.py").read_text())
shuffler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FeatureShuffler")
namespace = dict(random=random, itertools=itertools, deepcopy=deepcopy, Optional=Optional, List=List, np=np)
exec(compile(ast.fix_missing_locations(ast.Module(body=[shuffler], type_ignores=[])), "native_shuffler", "exec"), namespace)
FeatureShuffler = namespace["FeatureShuffler"]
NATIVE_GENERATE = source_method("preprocessing.py", "EnsembleGenerator", "_generate_ensemble")
NATIVE_REG_GENERATE = source_method("preprocessing.py", "EnsembleGenerator_Reg", "_generate_ensemble")
NATIVE_TRANSFORM = source_method("preprocessing.py", "EnsembleGenerator", "transform")
NATIVE_REG_TRANSFORM = source_method("preprocessing.py", "EnsembleGenerator_Reg", "transform")
NATIVE_CLF_PREDICT = source_method("classifier.py", "TabSwiftClassifier", "predict_proba")
NATIVE_REG_PREDICT = source_method("regressor.py", "TabSwiftRegressor", "predict_proba")
NATIVE_SOFTMAX = source_method("classifier.py", "TabSwiftClassifier", "softmax")


class Generator:
    def __init__(self, X, y, count, kind):
        self.n_estimators, self.n_features_in_ = count, X.shape[1]
        self.n_classes_ = len(np.unique(y))
        self.random_state, self.rng_ = 42, random.Random(42)
        self.feat_shuffle_method, self.norm_methods_ = "latin", ["none", "power"]
        self.class_shift = kind == "classification"
        self.X_, self.y_ = X, y
        self.unique_filter_ = SimpleNamespace(transform=lambda value: value)
        generate = NATIVE_GENERATE if self.class_shift else NATIVE_REG_GENERATE
        self.ensemble_configs_, self.feature_shuffle_patterns_, self.class_shift_offsets_ = generate(self)
        self.preprocessors_ = {}
        for method in self.ensemble_configs_:
            scale = 1. if method == "none" else .5
            self.preprocessors_[method] = SimpleNamespace(X_transformed_=X * scale, transform=lambda value, s=scale: value * s)
        self.kind = kind

    def transform(self, X):
        native = NATIVE_TRANSFORM if self.kind == "classification" else NATIVE_REG_TRANSFORM
        return native(self, X)


class ICL:
    register_tokens = 2
    def _icl_predictions(self, R, y_train, if_regression=False):
        return np.zeros((R.shape[0], R.shape[1] + self.register_tokens, 10), dtype=np.float32)
    def _icl_predictions_reg(self, R, y_train, if_regression=True):
        return np.zeros((R.shape[0], R.shape[1] + self.register_tokens, 1), dtype=np.float32)


class Model:
    max_classes = 10
    def __init__(self, kind, classes, truncate=False):
        self.kind, self.classes, self.truncate = kind, classes, truncate
        self.icl_predictor = ICL()
        self.hooks = []
    def register_forward_hook(self, hook, with_kwargs):
        assert with_kwargs
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))
    def __call__(self, X, y):
        if self.kind == "classification" and self.classes > self.max_classes:
            for xi, yi in zip(X, y):
                self.icl_predictor._icl_predictions(xi[None], yi[None])
                # Simulate two native hierarchy child nodes, each retaining ALL test rows.
                for half in np.array_split(np.arange(len(yi)), 2):
                    query = xi[len(yi):]
                    self.icl_predictor._icl_predictions(np.concatenate([xi[half], query])[None], yi[half][None])
        elif self.kind == "classification":
            self.icl_predictor._icl_predictions(X, y)
        else:
            self.icl_predictor._icl_predictions_reg(X, y)
        base = X[:, y.shape[1]:].sum(axis=-1).astype(np.float32) + y.mean(axis=1).astype(np.float32)[:, None]
        width = self.classes if self.kind == "classification" else 1
        result = np.stack([base * (k + 1) / 10 + k for k in range(width)], axis=-1).astype(np.float32)
        if self.truncate:
            result = result[:, :-1]
        for hook in self.hooks:
            hook(self, (X, y), {}, result)
        return result


class Estimator:
    def __init__(self, kind, count, classes=2, average_logits=True, truncate=False):
        self.kind, self.n_estimators, self.n_classes_ = kind, count, classes
        self.class_shift = kind == "classification"
        self.batch_size, self.random_state = 3, 42
        self.average_logits, self.softmax_temperature = average_logits, .9
        self.truncate = truncate
        self.X_encoder_ = SimpleNamespace(transform=lambda value: value)
    def _validate_data(self, X, **kwargs):
        return X
    def _apply_dimensionality_transform(self, X, is_fitting=False):
        return X
    def fit(self, X, y):
        transformed = self._apply_dimensionality_transform(X, is_fitting=True)
        self.ensemble_generator_ = Generator(transformed, y, self.n_estimators, self.kind)
        self.model_ = Model(self.kind, self.n_classes_, self.truncate)
        return self
    def _batch_forward(self, Xs, ys, shuffle_patterns=None):
        values = [self.model_(Xs[start:start+self.batch_size], ys[start:start+self.batch_size])
                  for start in range(0, len(Xs), self.batch_size)]
        return np.concatenate(values, axis=0)
    def predict_proba(self, X):
        native = NATIVE_CLF_PREDICT if self.kind == "classification" else NATIVE_REG_PREDICT
        return native(self, X)
    def predict(self, X):
        return self.predict_proba(X)
    softmax = staticmethod(NATIVE_SOFTMAX)


class EnsembleTests(unittest.TestCase):
    def arrays(self, features=1, classes=2):
        support = np.arange(12 * features, dtype=np.float32).reshape(12, features) / 30
        targets = np.arange(12) % classes
        test = np.arange(5 * features, dtype=np.float32).reshape(5, features) / 20
        return support, targets, test

    def execute(self, kind="classification", count=32, strict=True, features=1, classes=2, average_logits=True):
        support, targets, test = self.arrays(features, classes)
        estimator = Estimator(kind, count, classes, average_logits)
        handle = audit.configure(estimator, kind, count, strict)
        estimator.fit(support, targets)
        with redirect_stdout(io.StringIO()):
            result = estimator.predict(test)
        report = handle.finish(len(test))
        return estimator, handle, report, result

    def test_native_lowdim_classification_deduction_then_actual32(self):
        estimator, _, record, prediction = self.execute()
        self.assertEqual(record["extension_audit"]["native_count"], 4)
        self.assertEqual(record["actual_ensemble_count"], 32)
        self.assertEqual(record["unique_configuration_count"], 4)
        self.assertEqual(record["duplicate_configuration_member_count"], 28)
        self.assertEqual(record["actual_members_per_test_row"], 32)
        self.assertTrue(record["aggregation_verified_exact"])
        self.assertEqual(prediction.shape, (5, 2))
        self.assertEqual([len(c) for c in estimator.ensemble_generator_.ensemble_configs_.values()], [16, 16])
        self.assertTrue(all(member["aggregation_weight"] == 1/32 for member in record["members"]))

    def test_native_lowdim_regression_deduction_then_actual8(self):
        _, _, record, prediction = self.execute(kind="regression", count=8)
        self.assertEqual(record["extension_audit"]["native_count"], 2)
        self.assertEqual(record["actual_ensemble_count"], 8)
        self.assertEqual(record["unique_configuration_count"], 2)
        self.assertEqual(record["duplicate_configuration_member_count"], 6)
        self.assertEqual(prediction.shape, (5, 1))
        self.assertTrue(record["full_test_split"])

    def test_official16_configuration_and_predictions_bitwise_unchanged(self):
        for kind in ("classification", "regression"):
            support, targets, test = self.arrays()
            plain = Estimator(kind, 16).fit(support, targets)
            before = audit.configuration_records(plain.ensemble_generator_, kind)
            with redirect_stdout(io.StringIO()):
                reference = plain.predict(test)
            estimator, _, record, prediction = self.execute(kind=kind, count=16, strict=False)
            self.assertEqual(audit.configuration_records(estimator.ensemble_generator_, kind), before)
            self.assertTrue(np.array_equal(reference, prediction))
            self.assertFalse(record["extension_audit"]["extended"])
            self.assertEqual(record["actual_ensemble_count"], 4 if kind == "classification" else 2)

    def test_sufficient_native_counts_need_no_extension(self):
        _, _, record, _ = self.execute(features=10)
        self.assertEqual(record["actual_ensemble_count"], 32)
        self.assertFalse(record["extension_audit"]["extended"])
        _, _, reg, _ = self.execute(kind="regression", count=8, features=4)
        self.assertFalse(reg["extension_audit"]["extended"])

    def test_preserves_per_normalization_prefix_and_consumes_no_rng(self):
        support, targets, _ = self.arrays()
        generator = Generator(support, targets, 32, "classification")
        before = deepcopy(generator.ensemble_configs_)
        global_state, local_state = random.getstate(), generator.rng_.getstate()
        audit.extend_native_configs(generator, 32, "classification")
        self.assertEqual(random.getstate(), global_state)
        self.assertEqual(generator.rng_.getstate(), local_state)
        for norm, configs in before.items():
            self.assertEqual(generator.ensemble_configs_[norm][:len(configs)], configs)

    def test_deterministic_repeated_executions_and_probability_averaging(self):
        _, _, first, a = self.execute(average_logits=False)
        _, _, second, b = self.execute(average_logits=False)
        self.assertEqual(first["configurations"], second["configurations"])
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(first["aggregation_verified_exact"])

    def test_native_hierarchy_extra_compute_is_not_top_ensemble_count(self):
        _, _, record, result = self.execute(classes=12)
        self.assertEqual(result.shape, (5, 12))
        self.assertTrue(record["native_hierarchy_used"])
        self.assertEqual(record["actual_ensemble_count"], 32)
        self.assertEqual(record["icl_member_node_evaluations"], 96)
        self.assertTrue(all(call["test_rows"] == 5 for call in record["icl_execution_calls"]))

    def test_truncated_model_queries_fail_closed(self):
        support, targets, test = self.arrays()
        estimator = Estimator("classification", 32, truncate=True)
        handle = audit.configure(estimator, "classification", 32, True)
        estimator.fit(support, targets)
        with self.assertRaisesRegex(RuntimeError, "lost member/test rows"):
            estimator.predict(test)
        with self.assertRaises(RuntimeError):
            handle.finish(len(test))

    def test_PCA_hidden_row_drop_fails_closed(self):
        support, targets, _ = self.arrays()
        estimator = Estimator("classification", 32)
        estimator._apply_dimensionality_transform = lambda X, **kwargs: X[:-1]
        audit.configure(estimator, "classification", 32, True)
        with self.assertRaisesRegex(RuntimeError, "changed row count"):
            estimator.fit(support, targets)

    def test_native_aggregation_tampering_is_detected(self):
        support, targets, test = self.arrays()
        estimator = Estimator("classification", 32)
        original = estimator.predict_proba
        estimator.predict_proba = lambda X: original(X) + .01
        audit.configure(estimator, "classification", 32, True)
        estimator.fit(support, targets)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "final aggregation differs"):
            estimator.predict(test)

    def test_regression_label_shifts_and_wrong_budgets_rejected(self):
        estimator = Estimator("regression", 8)
        estimator.class_shift = True
        with self.assertRaisesRegex(RuntimeError, "class_shift=False"):
            audit.configure(estimator, "regression", 8, True)
        with self.assertRaises(RuntimeError):
            audit.configure(Estimator("classification", 8), "classification", 8, True)

    def test_close_restores_per_instance_methods_and_model_hooks(self):
        estimator, handle, _, _ = self.execute()
        handle.close()
        self.assertNotIn("fit", estimator.__dict__)
        self.assertNotIn("_batch_forward", estimator.__dict__)
        self.assertNotIn("_tabswift_ensemble_audit", estimator.__dict__)
        self.assertEqual(estimator.model_.hooks, [])
        self.assertNotIn("_icl_predictions", estimator.model_.icl_predictor.__dict__)


if __name__ == "__main__":
    unittest.main()
