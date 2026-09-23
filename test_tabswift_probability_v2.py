"""CPU tests; no checkpoint, GPU, network, or frozen-source edits."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

import tabswift_one as frozen
import tabswift_one_v2 as worker


def recipe(variant="official16"):
    return {"protocol": {"variant": variant,
        "n_estimators": {"classification": 16, "regression": 16} if variant == "official16" else {
            "classification": 32, "regression": 8},
        "strict_actual_count": variant == "budget32x8", "batch_size": 16,
        "random_state": 42, "precision": "fp32", "use_amp": False}}


class Tensor:
    def __init__(self, dtype="torch.float32"):
        self.dtype = dtype

    def is_floating_point(self):
        return self.dtype.startswith("torch.float") or self.dtype == "torch.bfloat16"


class FakeTorch:
    Tensor = Tensor
    float32 = "torch.float32"

    def __init__(self):
        self.enabled = {"cuda": False, "cpu": False}
        self.contexts = []
        self.precision = "high"
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                        cudnn=SimpleNamespace(allow_tf32=True))
        self.autocast = self._autocast

    @contextmanager
    def _autocast(self, device_type, dtype=None, enabled=True, cache_enabled=None):
        self.contexts.append((device_type, dtype, enabled, cache_enabled))
        previous = self.enabled[device_type]
        self.enabled[device_type] = enabled
        try:
            yield
        finally:
            self.enabled[device_type] = previous

    def is_autocast_enabled(self, device_type):
        return self.enabled[device_type]

    def get_float32_matmul_precision(self):
        return self.precision

    def set_float32_matmul_precision(self, value):
        self.precision = value


class Module:
    def __init__(self, torch, child=None, emulate_native_autocast=False):
        self.torch = torch
        self.child = child
        self.emulate_native_autocast = emulate_native_autocast
        self.training = False
        self.max_classes = 10
        self.parameter = Tensor()
        self.buffer = Tensor("torch.int64")
        self.output = Tensor()
        self.before, self.after = {}, {}

    def parameters(self):
        return iter([self.parameter])

    def buffers(self):
        return iter([self.buffer])

    def modules(self):
        return iter([self] + ([self.child] if self.child else []))

    @staticmethod
    def register(hooks, hook):
        key = len(hooks)
        hooks[key] = hook
        return SimpleNamespace(remove=lambda: hooks.pop(key, None))

    def register_forward_pre_hook(self, hook, *, with_kwargs):
        assert with_kwargs
        return self.register(self.before, hook)

    def register_forward_hook(self, hook, *, with_kwargs):
        assert with_kwargs
        return self.register(self.after, hook)

    def __call__(self, *args, **kwargs):
        for hook in tuple(self.before.values()):
            hook(self, args, kwargs)
        if self.emulate_native_autocast:
            # This reproduces pinned learning.py's unconditional context.
            with self.torch.autocast(device_type="cuda"):
                result = self.child(args[0])
        else:
            result = self.output
        for hook in tuple(self.after.values()):
            hook(self, args, kwargs, result)
        return result


class Handle:
    def __init__(self, count=16):
        self.closed = False
        self.count = count

    def finish(self, rows):
        return {"actual_ensemble_count": self.count, "full_test_split": True}

    def close(self):
        self.closed = True


class Estimator:
    def __init__(self, torch):
        self.use_amp = False
        self.n_estimators = 16
        self.pca_dim = 100
        self.model_ = Module(torch, child=Module(torch), emulate_native_autocast=True)
        self.proba = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float32)
        self.regression = np.array([1.25, -3.5], dtype=np.float32)
        self.y_encoder_ = SimpleNamespace(inverse_transform=lambda indices: np.array([10, 20])[indices])

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        self.model_(Tensor(), use_amp=self.use_amp)
        return self.proba

    def predict(self, x):
        self.model_(Tensor(), use_amp=self.use_amp)
        return self.regression


class ProtocolTests(unittest.TestCase):
    def test_only_explicit_precision_changes_constructor_recipe(self):
        for variant in ("official16", "budget32x8"):
            for task in ("classification", "regression"):
                old = frozen.protocol_settings(recipe(variant), task)
                new = worker.protocol_settings(recipe(variant), task)
                self.assertTrue(old.pop("use_amp"))
                self.assertFalse(new.pop("use_amp"))
                self.assertEqual(old, new)

    def test_missing_wrong_or_implicit_precision_rejected(self):
        for field, values in (("use_amp", [None, True, 0, "false"]),
                              ("precision", [None, "float32", "fp16", "bf16"])):
            for value in values:
                manifest = recipe()
                if value is None:
                    manifest["protocol"].pop(field)
                else:
                    manifest["protocol"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                    worker.protocol_settings(manifest, "classification")


class ProbabilityTests(unittest.TestCase):
    def test_fp32_values_and_argmax_unchanged_including_allowed_roundoff(self):
        probabilities = np.array([[0.1, 0.2, 0.7], [0.2, 0.4, 0.40001]], dtype=np.float32)
        before = probabilities.copy()
        expected = probabilities.argmax(axis=1).copy()
        audit = worker.validate_native_probabilities(probabilities, 2, 3)
        np.testing.assert_array_equal(probabilities, before)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), expected)
        self.assertEqual(audit["native_dtype"], "float32")
        self.assertEqual(audit["absolute_row_sum_tolerance"], 2e-5)
        self.assertLessEqual(audit["maximum_absolute_row_sum_error"], 2e-5)
        self.assertFalse(audit["probabilities_renormalized"])

    def test_half_float64_integer_or_wrong_shape_rejected(self):
        for dtype in (np.float16, np.float64, np.int32):
            with self.subTest(dtype=dtype), self.assertRaises(RuntimeError):
                worker.validate_native_probabilities(np.array([[1, 0]], dtype=dtype), 1, 2)
        for values, rows, classes in ((np.array([1, 0], dtype=np.float32), 1, 2),
                                      (np.ones((1, 2), dtype=np.float32), 2, 2),
                                      (np.empty((0, 2), dtype=np.float32), 0, 2)):
            with self.assertRaises(RuntimeError):
                worker.validate_native_probabilities(values, rows, classes)

    def test_nonfinite_negative_excess_and_bad_normalization_rejected(self):
        for values in ([np.nan, 1], [np.inf, 0], [-0.001, 1.001], [0, 1.00001],
                       [0.2, 0.2], [0.5, 0.5001], [0.5, 0.4999]):
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                worker.validate_native_probabilities(np.array([values], dtype=np.float32), 1, 2)


class RuntimePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.torch = FakeTorch()
        self.estimator = Estimator(self.torch)
        self.original = self.torch.autocast

    def assert_restored(self):
        self.assertIs(self.torch.autocast, self.original)
        self.assertEqual(self.torch.precision, "high")
        self.assertTrue(self.torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(self.torch.backends.cudnn.allow_tf32)
        for module in self.estimator.model_.modules():
            self.assertFalse(module.before)
            self.assertFalse(module.after)

    def test_unconditional_nested_native_autocast_is_disabled_and_audited(self):
        self.torch.enabled["cpu"] = True
        with worker.fp32_native_guard(self.torch, self.estimator) as audit:
            self.estimator.model_(Tensor(), use_amp=False)
            self.assertEqual(self.torch.precision, "highest")
            self.assertFalse(self.torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(self.torch.backends.cudnn.allow_tf32)
        self.assertEqual(audit["autocast_requests_disabled"], 1)
        self.assertEqual(audit["forward_hook_calls"], 2)
        self.assertGreater(audit["checked_float_tensor_inputs"], 0)
        self.assertGreater(audit["checked_float_tensor_outputs"], 0)
        self.assertEqual(audit["autocast_enabled_observations"], 0)
        self.assertTrue(all(context[2] is False for context in self.torch.contexts))
        self.assertTrue(self.torch.enabled["cpu"])
        self.assert_restored()

    def test_positional_autocast_arguments_preserved_except_enabled(self):
        with worker.fp32_native_guard(self.torch, self.estimator):
            with self.torch.autocast("cuda", "sentinel_dtype", True, False):
                self.estimator.model_(Tensor(), use_amp=False)
        self.assertIn(("cuda", "sentinel_dtype", False, False), self.torch.contexts)
        self.assert_restored()

    def test_half_parameters_and_half_buffers_rejected_without_conversion(self):
        for field in ("parameter", "buffer"):
            tensor = getattr(self.estimator.model_, field)
            previous = tensor.dtype
            for bad in ("torch.float16", "torch.bfloat16"):
                tensor.dtype = bad
                with self.subTest(field=field, dtype=bad), self.assertRaises(RuntimeError):
                    with worker.fp32_native_guard(self.torch, self.estimator):
                        self.fail("must fail before prediction")
                self.assertEqual(tensor.dtype, bad)
                self.assert_restored()
            tensor.dtype = previous

    def test_half_input_output_and_wrong_use_amp_rejected(self):
        for kind in ("input", "output", "use_amp"):
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                with worker.fp32_native_guard(self.torch, self.estimator):
                    if kind == "output":
                        self.estimator.model_.child.output.dtype = "torch.float16"
                    self.estimator.model_(Tensor("torch.float16" if kind == "input" else "torch.float32"),
                                          use_amp=kind == "use_amp")
            self.estimator.model_.child.output.dtype = "torch.float32"
            self.assert_restored()

    def test_no_actual_forward_fails_and_restores_context(self):
        with self.assertRaises(RuntimeError):
            with worker.fp32_native_guard(self.torch, self.estimator):
                pass
        self.assert_restored()

    def test_classifier_prediction_argmax_and_raw_probabilities_unchanged(self):
        handle = Handle()
        before = self.estimator.proba.copy()
        expected = self.estimator.y_encoder_.inverse_transform(before.argmax(axis=1))
        prediction, audit = worker.predict_native(self.estimator, np.zeros((4, 2)),
            np.array([10, 20, 10, 20]), np.zeros((2, 2)), "classification", False,
            lambda *args, **kwargs: handle, self.torch)
        np.testing.assert_array_equal(prediction, expected)
        np.testing.assert_array_equal(self.estimator.proba, before)
        self.assertEqual(audit["probability_sha256"], worker.data_helper.array_digest(before))
        self.assertTrue(handle.closed)
        self.assertFalse(audit["precision_audit"]["native_use_amp"])
        self.assert_restored()

    def test_regression_values_unchanged_and_half_predictions_rejected(self):
        for dtype in (np.float32, np.float16):
            self.estimator.regression = self.estimator.regression.astype(dtype)
            before = self.estimator.regression.copy()
            handle = Handle()
            def predict():
                return worker.predict_native(self.estimator, np.zeros((4, 2)), np.arange(4),
                    np.zeros((2, 2)), "regression", False,
                    lambda *args, **kwargs: handle, self.torch)
            if dtype == np.float32:
                result, audit = predict()
                np.testing.assert_array_equal(result, before)
                self.assertIsNone(audit["probability_validation"])
            else:
                with self.assertRaises(RuntimeError):
                    predict()
            np.testing.assert_array_equal(self.estimator.regression, before)
            self.assertTrue(handle.closed)
            self.assert_restored()


if __name__ == "__main__":
    unittest.main()
