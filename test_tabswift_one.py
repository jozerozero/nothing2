"""CPU-only tests of the worker; no checkpoint, GPU, network, or source edits."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import typing
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
try:
    import sklearn.preprocessing
    from sklearn.impute import SimpleImputer
except ImportError:
    # The bundled CPU runtime lacks sklearn. These tiny test doubles exercise
    # the official function's control flow; they are NEVER production fallbacks.
    class LabelEncoder:
        def fit(self, y):
            self.classes_ = np.unique(y)
            return self
        def transform(self, y): return np.searchsorted(self.classes_, y)
        def inverse_transform(self, y): return self.classes_[np.asarray(y, dtype=int)]
    class OrdinalEncoder:
        def __init__(self, *, handle_unknown, unknown_value, dtype):
            self.unknown_value, self.dtype = unknown_value, dtype
        def fit(self, x):
            self.categories_ = [np.unique(column) for column in x.T]
            return self
        def transform(self, x):
            result = np.full(x.shape, self.unknown_value, dtype=self.dtype)
            for col, vocabulary in enumerate(self.categories_):
                for index, label in enumerate(vocabulary):
                    result[x[:, col] == label, col] = index
            return result
    class SimpleImputer:
        def __init__(self, **kwargs):
            raise AssertionError("mean/new recipe must not instantiate an imputer")
    sklearn = SimpleNamespace(preprocessing=SimpleNamespace(LabelEncoder=LabelEncoder, OrdinalEncoder=OrdinalEncoder))

import tabswift_one as worker


def recipe(variant="official16"):
    return {"protocol": {"variant": variant,
        "n_estimators": {"classification": 16, "regression": 16} if variant == "official16" else {
            "classification": 32, "regression": 8},
        "strict_actual_count": variant != "official16", "batch_size": 16, "random_state": 42}}


def official_processing():
    """Execute the three actual official functions without heavyweight imports.

    This is a CPU-test-only AST extraction, not a production preprocessing copy.
    The production worker imports the complete pinned TALENT module normally.
    """
    default = Path(__file__).resolve().parents[2] / "tabswift_standard681_20260922_v1/official"
    path = Path(os.environ.get("TABSWIFT_OFFICIAL_ROOT", default)) / "TALENT/model/lib/data.py"
    names = {"data_nan_process", "data_enc_process", "data_label_process", "raise_unknown"}
    source = ast.parse(path.read_text())
    module = ast.Module(body=[node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = dict(np=np, sklearn=sklearn, deepcopy=deepcopy, SimpleImputer=SimpleImputer, ty=typing)
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def identity(path, content):
    path.write_bytes(content)
    stat = path.stat()
    return dict(path=str(path.resolve()), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sha256=worker.data_helper.sha_file(path))


class Handle:
    def __init__(self, count=16, full=True):
        self.count, self.full, self.closed = count, full, False
    def finish(self, rows):
        return {"actual_ensemble_count": self.count, "full_test_split": self.full, "test_rows": rows}
    def close(self): self.closed = True


class Estimator:
    def __init__(self, n=16, regression=False):
        self.n_estimators, self.pca_dim, self.regression = n, 100, regression
        self.model_ = SimpleNamespace(training=False, max_classes=10)
        self.calls = []
    def fit(self, x, y):
        self.calls.append("fit")
        self.seen_x, self.seen_y = x.copy(), y.copy()
        if not self.regression:
            self.y_encoder_ = sklearn.preprocessing.LabelEncoder().fit(y)
        return self
    def predict_proba(self, x):
        self.calls.append("predict_proba")
        p = np.zeros((len(x), len(self.y_encoder_.classes_)))
        p[:, 0] = 1
        return p
    def predict(self, x):
        self.calls.append("predict")
        return np.arange(len(x), dtype=float).reshape(-1, 1)


class Tests(unittest.TestCase):
    def setUp(self):
        self.processing = official_processing()
        self.tx = pd.DataFrame({"category": ["z", "a", "a", "a"], "number": [1., 3., 5., 7.]})
        self.vx = pd.DataFrame({"category": ["unseen", "a"], "number": [np.nan, 1000.]})
        self.y = np.array([10., 20., 30., 40.])

    def test_protocols_only_change_requested_budget(self):
        for task, budget in (("classification", 32), ("regression", 8)):
            official = worker.protocol_settings(recipe(), task)
            strict = worker.protocol_settings(recipe("budget32x8"), task)
            self.assertEqual(official["n_estimators"], 16)
            self.assertEqual(strict.pop("n_estimators"), budget)
            official.pop("n_estimators")
            self.assertEqual(official, strict)
            self.assertEqual(official["class_shift"], task == "classification")
            self.assertEqual(official["batch_size"], 16)
            self.assertFalse(official["allow_auto_download"])

    def test_protocol_rejects_mixed_budgets_and_seed(self):
        for field, value in (("n_estimators", {"classification": 32, "regression": 16}),
                             ("strict_actual_count", True), ("batch_size", 8), ("random_state", 0)):
            man = recipe()
            man["protocol"][field] = value
            with self.assertRaises(RuntimeError): worker.protocol_settings(man, "regression")

    def test_preprocessing_support_only_exact_official_unknown_behavior(self):
        before, query = self.tx.copy(deep=True), self.vx.copy(deep=True)
        train, y, test, info, encoder, audit = worker.official_preprocess(
            self.tx, self.y, self.vx, "regression", self.processing)
        # Numeric first; test-only NaN uses support mean4, not query1000.
        np.testing.assert_array_equal(train[:, 0], [1., 3., 5., 7.])
        np.testing.assert_array_equal(test[:, 0], [4., 1000.])
        # Official branch uses first support category'z' (code1), NOT mode'a'.
        np.testing.assert_array_equal(test[:, 1], [1., 0.])
        self.assertEqual(audit["official_unknown_replacement_values"], [1])
        np.testing.assert_allclose(y * info["std"] + info["mean"], self.y)
        self.assertEqual(info["mean"], 25.)
        self.assertIsNone(encoder)
        self.assertFalse(audit["test_labels_used"])
        self.assertEqual(audit["additional_support_rows"], 0)
        self.assertEqual(train.shape[0], len(self.tx))
        pd.testing.assert_frame_equal(self.tx, before)
        pd.testing.assert_frame_equal(self.vx, query)

    def test_preprocessing_never_receives_test_labels(self):
        import inspect
        signature = inspect.signature(worker.official_preprocess)
        self.assertEqual(list(signature.parameters), ["tx", "ys", "vx", "task", "processing"])

    def test_label_encoding_is_invertible_and_support_only(self):
        labels = np.array([20, 10, 10, 20])
        train, y, test, info, encoder, audit = worker.official_preprocess(
            self.tx, labels, self.vx, "classification", self.processing)
        np.testing.assert_array_equal(y, [1, 0, 0, 1])
        np.testing.assert_array_equal(encoder.inverse_transform(y), labels)
        self.assertEqual(info, {"policy": "none"})
        self.assertIsNone(audit["regression_inverse"])

    def test_all_numeric_and_all_categorical_native_blocks(self):
        for names in (["number"], ["category"]):
            train, _, test, *_ = worker.official_preprocess(
                self.tx[names], self.y, self.vx[names], "regression", self.processing)
            self.assertEqual(train.shape, (4, 1))
            self.assertEqual(test.shape, (2, 1))
            self.assertTrue(np.isfinite(train).all() and np.isfinite(test).all())

    def test_constant_regression_target_fails_not_silently_rescaled(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                worker.official_preprocess(self.tx, np.ones(4), self.vx, "regression", self.processing)

    def test_native_predict_classification_and_regression_once(self):
        for regression in (False, True):
            est, handle = Estimator(regression=regression), Handle()
            prediction, audit = worker.predict_native(est, np.zeros((4, 2)),
                np.array([10, 20, 10, 20]) if not regression else self.y,
                np.zeros((2, 2)), "regression" if regression else "classification", False,
                lambda *a, **kw: handle)
            self.assertEqual(est.calls, ["fit", "predict" if regression else "predict_proba"])
            np.testing.assert_array_equal(prediction, [0., 1.] if regression else [10, 10])
            self.assertEqual(audit["actual_ensemble_count"], 16)
            self.assertTrue(handle.closed)

    def test_official_native_deduction_accepted_but_strict_short_count_fails(self):
        args = (np.zeros((4, 2)), np.array([0, 1, 0, 1]), np.zeros((2, 2)), "classification")
        _, audit = worker.predict_native(Estimator(), *args, False, lambda *a, **kw: Handle(count=4))
        self.assertEqual(audit["actual_ensemble_count"], 4)
        handle = Handle(count=31)
        with self.assertRaisesRegex(RuntimeError, "Strict actual"):
            worker.predict_native(Estimator(n=32), *args, True, lambda *a, **kw: handle)
        self.assertTrue(handle.closed)

    def test_native_hierarchy_over100_fails_without_substitution(self):
        est, handle = Estimator(), Handle()
        with self.assertRaisesRegex(RuntimeError, "Native TabSwift hierarchy"):
            worker.predict_native(est, np.zeros((101, 2)), np.arange(101), np.zeros((2, 2)),
                "classification", False, lambda *a, **kw: handle)
        self.assertEqual(est.calls, ["fit"])
        self.assertTrue(handle.closed)

    def test_missing_full_query_evidence_fails(self):
        with self.assertRaisesRegex(RuntimeError, "full-query"):
            worker.predict_native(Estimator(), np.zeros((4, 2)), np.array([0, 1, 0, 1]),
                np.zeros((2, 2)), "classification", False, lambda *a, **kw: Handle(full=False))

    def test_checkpoint_load_safe_once_and_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            rec = identity(Path(directory) / "swift.ckpt", b"fake fixture")
            native = lambda *a, **kw: {"config": {"max_classes": 10}, "state_dict": {"weight": object()}}
            torch = SimpleNamespace(load=native)
            with worker.checkpoint_load_guard(torch, rec) as audit:
                value = torch.load(rec["path"], map_location="cpu", weights_only=True)
                self.assertIn("state_dict", value)
                with self.assertRaisesRegex(RuntimeError, "more than once"):
                    torch.load(rec["path"], map_location="cpu", weights_only=True)
            self.assertIs(torch.load, native)
            self.assertEqual(audit["loads"], 1)
            self.assertEqual(audit["state_dict_key_count"], 1)

    def test_checkpoint_guard_rejects_unsafe_or_other_file(self):
        with tempfile.TemporaryDirectory() as directory:
            rec = identity(Path(directory) / "swift.ckpt", b"fake fixture")
            torch = SimpleNamespace(load=lambda *a, **kw: self.fail("unsafe native load reached"))
            with worker.checkpoint_load_guard(torch, rec):
                for filename, kw in ((rec["path"], {"map_location": "cpu", "weights_only": False}),
                                     (str(Path(directory) / "other.ckpt"), {"map_location": "cpu", "weights_only": True})):
                    with self.assertRaisesRegex(RuntimeError, "unsafe"):
                        torch.load(filename, **kw)

    def test_output_guard_and_exclusive_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            man = {"output_root": str(root / "campaign")}
            output = root / "campaign/results/row-000.json"
            self.assertEqual(worker.checked_output(man, output), output.resolve())
            with self.assertRaises(RuntimeError): worker.checked_output(man, root / "elsewhere.json")
            worker.common.publish_new(output, {"complete": True})
            with self.assertRaises(RuntimeError): worker.checked_output(man, output)
            with self.assertRaises(FileExistsError): worker.common.publish_new(output, {"complete": False})
            self.assertEqual(output.read_text().count('true'), 1)


if __name__ == "__main__":
    unittest.main()
