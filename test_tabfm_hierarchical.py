"""CPU tests; native TabICLv2 hierarchy methods are executed via an ndarray shim."""
from __future__ import annotations

import ast
import hashlib
import inspect
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import tabfm_hierarchical as helper


REFERENCE_PATH = (Path(__file__).resolve().parents[2] /
    "tabicl_regression_dual8node_20260820/references/tabicl/src/tabicl/_model/learning.py")


class Tensor(np.ndarray):
    def __new__(cls, value, dtype=None):
        return np.asarray(value, dtype=dtype).view(cls)

    @property
    def device(self):
        return "cpu"

    def to(self, *args):
        return self

    def int(self):
        return Tensor(self, dtype=np.int64)

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self, dim))


class TorchShim:
    int = np.int64

    @staticmethod
    def zeros(shape, dtype=None, device=None):
        return Tensor(np.zeros(shape, dtype=dtype))

    @staticmethod
    def tensor(value, dtype=None):
        return Tensor(value, dtype)

    @staticmethod
    def unique(value, return_inverse=False):
        output = np.unique(value, return_inverse=return_inverse)
        return tuple(Tensor(v) for v in output) if return_inverse else Tensor(output)

    @staticmethod
    def searchsorted(a, b):
        return Tensor(np.searchsorted(a, b))

    @staticmethod
    def cat(values, dim=0):
        return Tensor(np.concatenate(values, axis=dim))


class ReferenceNode:
    def __init__(self, depth):
        self.depth = depth
        self.child_nodes = []


def native_reference_class():
    """Load only pinned, original hierarchy methods, not neural-network imports."""
    source = REFERENCE_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != helper.REFERENCE["sha256"]:
        raise AssertionError("TabICLv2 test reference changed")
    module = ast.parse(source)
    original = next(node for node in module.body if isinstance(node, ast.ClassDef)
                    and node.name == "ICLearning")
    names = {"_grouping", "_fit_node", "_fit_hierarchical", "_label_encoding", "_predict_hierarchical"}
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names]
    target = ast.ClassDef(name="NativeReference", bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {"torch": TorchShim, "math": math, "Tensor": Tensor,
                 "ClassNode": ReferenceNode, "Optional": __import__("typing").Optional}
    code = ast.fix_missing_locations(ast.Module(body=[target], type_ignores=[]))
    exec(compile(code, str(REFERENCE_PATH), "exec"), namespace)
    return namespace["NativeReference"]


class Parameter:
    _version = 0
    shape = (1,)
    dtype = "float32"

    def data_ptr(self):
        return id(self)


class Model:
    max_classes = 10
    training = False

    def __init__(self):
        self.parameter = Parameter()

    def parameters(self):
        return iter([self.parameter])


class Estimator:
    def __init__(self, model):
        self.model = model
        self.n_estimators = 32
        self.random_state = 42
        self.max_num_rows = None
        self.max_num_features = 500
        self.batch_size = 1
        self.cache_context = False
        self.enable_nnls = False
        self.class_shift = True
        self.average_logits = True
        self.softmax_temperature = 0.9


def conditional(y, X_test):
    counts = np.bincount(np.asarray(y, dtype=np.int64)).astype(float)
    # Depend on both the node's support class frequencies and each query row.
    scores = counts[None, :] + (np.asarray(X_test)[:, 0:1] + 1) * .05 * (
        1 + np.arange(len(counts))[None, :])
    return scores / scores.sum(axis=1, keepdims=True)


def predictor(est, X, y, X_test, node_info):
    assert len(y) == len(X) == node_info["support_rows"]
    est.classes_ = np.unique(y)
    return conditional(y, X_test), {
        "actual_ensemble_count": 32, "native_member_forward_verified": True,
        "full_test_split": True, "support_rows": len(y), "test_rows": len(X_test),
    }


class HierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = native_reference_class()

    def data(self, classes=25):
        y = np.repeat(np.arange(classes) * 3 + 7, 1 + np.arange(classes) % 4)
        X = np.column_stack([np.arange(len(y)), np.arange(len(y)) % 3]).astype(float)
        X_test = np.asarray([[1., 0.], [1., 0.], [3., 1.], [5., 2.]])
        return X, y, X_test

    def run_helper(self, classes=25, callback=predictor, factory=Estimator, model=None):
        X, y, X_test = self.data(classes)
        return helper.hierarchical_predict_proba(model or Model(), X, y, X_test,
            classifier_factory=factory, node_predictor=callback)

    def test_exact_pinned_native_grouping(self):
        ref = self.reference()
        for capacity in (2, 3, 10):
            ref.max_classes = capacity
            for count in range(1, 302):
                actual, groups = helper.balanced_groups(count, capacity)
                expected, expected_groups = ref._grouping(count)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(groups, expected_groups)

    def test_exact_pinned_native_probability_recursion(self):
        for count in (11, 25, 100, 101):
            X, y, X_test = self.data(count)
            reference = self.reference()
            reference.max_classes = 10
            reference._fit_hierarchical(Tensor(X), Tensor(np.unique(y, return_inverse=True)[1]))

            def node_probability(R, y_train, **kwargs):
                del kwargs
                targets = np.asarray(y_train)[0]
                query = np.asarray(R)[0, len(targets):]
                return Tensor(conditional(targets, query)[None])

            reference._predict_standard = node_probability
            expected = reference._predict_hierarchical(Tensor(X_test))
            actual, audit = self.run_helper(count)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
            self.assertEqual(audit["classes"], np.unique(y).tolist())

    def test_11_class_split_and_per_node_count(self):
        probs, audit = self.run_helper(11)
        self.assertEqual(audit["node_count"], 3)
        self.assertEqual(audit["tree"][0]["class_group_assignments"], [0] * 6 + [1] * 5)
        self.assertEqual(audit["tree"][0]["child_node_ids"], [1, 2])
        self.assertEqual(audit["actual_ensemble_count"], 32)
        self.assertEqual(audit["total_member_forward_count"], 96)
        np.testing.assert_allclose(probs.sum(axis=1), 1)
        np.testing.assert_array_equal(probs[0], probs[1])  # Duplicate query rows remain legitimate.

    def test_deep_tree_101_classes(self):
        _, audit = self.run_helper(101)
        self.assertEqual(audit["node_count"], 13)
        self.assertEqual(max(node["depth"] for node in audit["tree"]), 2)
        self.assertEqual(audit["total_member_forward_count"], 416)
        self.assertTrue(all(node["local_class_count"] <= 10 for node in audit["tree"]))

    def test_one_shared_model_and_every_test_row(self):
        model = Model()
        seen = []

        def observe(est, X, y, X_test, info):
            self.assertIs(est.model, model)
            self.assertEqual(len(X_test), 4)
            seen.append((info["node_id"], X[:, 0].astype(int).tolist(), y.copy()))
            self.assertEqual(X[:, 0].astype(int).tolist(), sorted(X[:, 0].astype(int).tolist()))
            return predictor(est, X, y, X_test, info)

        _, audit = self.run_helper(callback=observe, model=model)
        self.assertEqual(len(seen), audit["node_count"])
        self.assertTrue(audit["checkpoint_parameters_unchanged"])
        self.assertEqual(audit["optimizer_updates"], 0)

    def test_probability_columns_follow_estimator_labels(self):
        def reverse(est, X, y, X_test, info):
            p, audit = predictor(est, X, y, X_test, info)
            est.classes_ = est.classes_[::-1]
            return p[:, ::-1], audit

        expected, _ = self.run_helper()
        actual, _ = self.run_helper(callback=reverse)
        np.testing.assert_array_equal(actual, expected)

    def test_deterministic_and_test_feature_independent_tree(self):
        first, first_audit = self.run_helper()
        second, second_audit = self.run_helper()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_audit, second_audit)
        X, y, X_test = self.data()
        _, changed = helper.hierarchical_predict_proba(Model(), X, y, X_test + 100,
            classifier_factory=Estimator, node_predictor=predictor)
        self.assertEqual(first_audit["tree"], changed["tree"])
        self.assertNotIn("y_test", inspect.signature(helper.hierarchical_predict_proba).parameters)

    def test_inputs_unchanged(self):
        X, y, X_test = self.data()
        originals = [value.copy() for value in (X, y, X_test)]
        helper.hierarchical_predict_proba(Model(), X, y, X_test,
            classifier_factory=Estimator, node_predictor=predictor)
        for value, original in zip((X, y, X_test), originals):
            np.testing.assert_array_equal(value, original)

    def test_reject_native_size_task(self):
        with self.assertRaisesRegex(RuntimeError, "only for more classes"):
            self.run_helper(10)

    def test_reject_modified_default(self):
        def factory(model):
            est = Estimator(model)
            est.n_estimators = 8
            return est
        with self.assertRaisesRegex(RuntimeError, "n_estimators"):
            self.run_helper(factory=factory)

    def test_reject_different_model(self):
        with self.assertRaisesRegex(RuntimeError, "share one checkpoint"):
            self.run_helper(factory=lambda _: Estimator(Model()))

    def test_reject_parameter_mutation(self):
        def mutate(est, X, y, X_test, info):
            est.model.parameter._version += 1
            return predictor(est, X, y, X_test, info)
        with self.assertRaisesRegex(RuntimeError, "changed or replaced"):
            self.run_helper(callback=mutate)

    def test_reject_partial_or_unaudited_members(self):
        for key, wrong in (("actual_ensemble_count", 31), ("native_member_forward_verified", False),
                           ("test_rows", 3), ("support_rows", 1), ("full_test_split", False)):
            def partial(est, X, y, X_test, info):
                p, audit = predictor(est, X, y, X_test, info)
                audit[key] = wrong
                return p, audit
            with self.assertRaisesRegex(RuntimeError, "prove default32"):
                self.run_helper(callback=partial)

    def test_reject_invalid_probabilities(self):
        for mode in ("nan", "negative", "unnormalized", "shape"):
            def invalid(est, X, y, X_test, info):
                p, audit = predictor(est, X, y, X_test, info)
                if mode == "nan": p[0, 0] = np.nan
                elif mode == "negative": p[0, 0] = -1
                elif mode == "unnormalized": p *= 2
                else: p = p[:, :-1]
                return p, audit
            with self.assertRaises(RuntimeError):
                self.run_helper(callback=invalid)

    def test_string_labels_are_sorted_and_preserved(self):
        X, y, X_test = self.data(11)
        labels = np.asarray(["class-%03d" % value for value in y])
        _, audit = helper.hierarchical_predict_proba(Model(), X, labels, X_test,
            classifier_factory=Estimator, node_predictor=predictor)
        self.assertEqual(audit["classes"], sorted(set(labels)))

    def test_dataframe_dtype_and_original_row_order_preserved(self):
        import pandas as pd
        X, y, X_test = self.data()
        frame = pd.DataFrame({"row_id": X[:, 0], "category": pd.Categorical(X[:, 1].astype(str))})
        query = pd.DataFrame({"row_id": X_test[:, 0], "category": pd.Categorical(X_test[:, 1].astype(str))})

        def observe(est, support, targets, test, info):
            self.assertIsInstance(support, pd.DataFrame)
            self.assertEqual(str(support.dtypes.iloc[1]), "category")
            self.assertTrue(np.all(np.diff(support.row_id) > 0))
            est.classes_ = np.unique(targets)
            return conditional(targets, test[["row_id"]].to_numpy()), {
                "actual_ensemble_count": 32, "native_member_forward_verified": True,
                "full_test_split": True, "support_rows": len(targets), "test_rows": len(test),
            }
        p, _ = helper.hierarchical_predict_proba(Model(), frame, y, query,
            classifier_factory=Estimator, node_predictor=observe)
        np.testing.assert_allclose(p.sum(axis=1), 1)


if __name__ == "__main__":
    unittest.main()
