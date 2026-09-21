"""CPU-only tests; no model downloads, checkpoint loads, GPUs, or source edits."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import ast
import itertools
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import classification32_tabicl as adapter


class Handle:
    def __init__(self, hooks, value):
        self.hooks, self.value = hooks, value
    def remove(self):
        self.hooks.remove(self.value)


class Module:
    def __init__(self):
        self.pre, self.post = [], []
    def register_forward_pre_hook(self, fn, with_kwargs=False):
        item = (fn, with_kwargs)
        self.pre.append(item)
        return Handle(self.pre, item)
    def register_forward_hook(self, fn, with_kwargs=False):
        item = (fn, with_kwargs)
        self.post.append(item)
        return Handle(self.post, item)
    def __call__(self, *args, **kwargs):
        for fn, with_kwargs in self.pre:
            fn(self, args, kwargs) if with_kwargs else fn(self, args)
        result = self.forward(*args, **kwargs)
        for fn, with_kwargs in self.post:
            fn(self, args, kwargs, result) if with_kwargs else fn(self, args, result)
        return result


class Model(Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        self.seen_support = []
        self.seen_queries = []
    def forward(self, X, y_train, **_):
        n = y_train.shape[1]
        self.seen_support.extend((x[:n].copy(), y.copy()) for x, y in zip(X, y_train))
        self.seen_queries.extend(x[n:].copy() for x in X)
        # Deterministic and sensitive to support row order; actual distinct calls.
        weights = np.arange(1, n + 1, dtype=np.float32)
        context = (X[:, :n, 0] * weights).sum(axis=1) / n
        result = np.stack([X[:, n:, 0] * (k + 1) + context[:, None] / (k + 1)
                           for k in range(self.classes)], axis=-1)
        return result.astype(np.float32)


class Generator:
    classification = True
    def __init__(self, X, y, features=1, classes=2):
        self.y_, self.X_ = np.asarray(y), np.asarray(X)
        self.preprocessors_ = {
            "none": SimpleNamespace(X_transformed_=np.asarray(X, dtype=np.float32)),
            "power": SimpleNamespace(X_transformed_=np.asarray(X, dtype=np.float32) * 0.7)}
        pairs = list(itertools.product(
            [list(range(i, features)) + list(range(i)) for i in range(features)],
            [list(range(i, classes)) + list(range(i)) for i in range(classes)]))
        random.Random(42).shuffle(pairs)
        combinations = list(itertools.product(pairs, ["none", "power"]))[:32]
        self.ensemble_configs_ = OrderedDict(
            (norm, [pair for pair, selected in combinations if selected == norm])
            for norm in ("none", "power"))
        self.feature_shuffles_ = OrderedDict((k, [p[0] for p in v]) for k, v in self.ensemble_configs_.items())
        self.class_shuffles_ = OrderedDict((k, [p[1] for p in v]) for k, v in self.ensemble_configs_.items())
    def transform(self, X, mode="both", **_):
        assert mode == "both"
        data = OrderedDict()
        for norm, configs in self.ensemble_configs_.items():
            test = np.asarray(X, dtype=np.float32) * (0.7 if norm == "power" else 1)
            joined = np.concatenate([self.preprocessors_[norm].X_transformed_, test])
            data[norm] = (np.stack([joined[:, f] for f, c in configs]),
                          np.stack([np.asarray(c, dtype=np.float32)[self.y_] for f, c in configs]))
        return data


class Estimator:
    n_estimators, kv_cache, use_pseudo_ssmax_thinking = 32, False, False
    average_logits, batch_size, random_state = True, 8, 42
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = Model(len(self.classes_))
        self.ensemble_generator_ = Generator(X, y, X.shape[1], len(self.classes_))
        self.model_kv_cache_ = None
        return self
    def _batch_forward(self, Xs, ys, feature_shuffles=None):
        n = int(np.ceil(len(Xs) / self.batch_size))
        return np.concatenate([self.model_(X=X, y_train=y)
                               for X, y in zip(np.array_split(Xs, n), np.array_split(ys, n))])
    def _aggregate_ensemble_outputs(self, outputs, class_shuffles):
        avg = np.zeros_like(outputs[0])
        for out, shuffle in zip(outputs, class_shuffles):
            avg += out[..., shuffle]
        avg /= len(class_shuffles)
        avg = avg / 0.9
        exp = np.exp(avg - avg.max(axis=1, keepdims=True))
        result = exp / exp.sum(axis=1, keepdims=True)
        return result / result.sum(axis=1, keepdims=True)
    def predict_proba(self, X):
        data = self.ensemble_generator_.transform(X, mode="both")
        outputs = [self._batch_forward(Xs, ys, self.ensemble_generator_.feature_shuffles_[norm])
                   for norm, (Xs, ys) in data.items()]
        shuffles = [c for group in self.ensemble_generator_.class_shuffles_.values() for c in group]
        return self._aggregate_ensemble_outputs(np.concatenate(outputs), shuffles)


def arrays(features=1, classes=2):
    return {"X_train": np.arange(12 * features, dtype=np.float32).reshape(12, features) / 20,
            "y_train": np.arange(12) % classes,
            "X_test": np.arange(5 * features, dtype=np.float32).reshape(5, features) / 10}


class Tests(unittest.TestCase):
    def test_source_identity_shapes_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "preprocessing.py"
            source.write_bytes(b"frozen source\n")
            stat = source.stat()
            sha = adapter.file_sha256(source)
            adapter._verify_source_pins(root, {"source_files": {source.name: sha}})
            adapter._verify_source_pins(root, {"source_files": [{"path": str(source),
                "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha}],
                "runtime_sources": {str(source): sha}})
            with self.assertRaisesRegex(RuntimeError, "Source pin mismatch"):
                adapter._verify_source_pins(root, {"runtime_sources": {str(source): "0" * 64}})
            with self.assertRaisesRegex(RuntimeError, "Source size mismatch"):
                adapter._verify_source_pins(root, {"source_files": [{"path": str(source),
                    "sha256": sha, "size_bytes": 999}]})

    def test_factory_keeps_original_constructor_and_checks_loop_selection(self):
        class Classifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        class Gen:
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            package = root / "src/tabicl"
            package.mkdir(parents=True)
            checkpoint = root / "step-19650.ckpt"
            checkpoint.write_bytes(b"test checkpoint")
            classifier_path, prep_path = package / "classifier.py", package / "preprocessing.py"
            classifier_path.write_bytes(b"audited classifier placeholder")
            prep_path.write_bytes(b"audited generator placeholder")
            config = {"source_root": str(root), "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": adapter.file_sha256(checkpoint), "loop": 3, "checkpoint_step": 19650}
            real_sha = adapter.file_sha256
            def sha(path):
                return ({classifier_path: adapter.CLASSIFIER_SHA256, prep_path: adapter.PREPROCESSING_SHA256}
                        .get(Path(path)) or real_sha(path))
            fake = SimpleNamespace(__file__=str(package / "__init__.py"), TabICLClassifier=Classifier,
                                   EnsembleGenerator=Gen)
            old_path = list(sys.path)
            try:
                with patch.dict(adapter.os.environ, {"PYTHONHASHSEED": "0"}), \
                     patch.object(adapter.importlib, "import_module", return_value=fake), \
                     patch.object(adapter.inspect, "getfile", side_effect=lambda x: classifier_path if x is Classifier else prep_path), \
                     patch.object(adapter, "file_sha256", side_effect=sha):
                    estimator = adapter.build_estimator("loop3", config)
                    self.assertEqual(estimator.kwargs["n_estimators"], 32)
                    self.assertEqual(estimator.kwargs["batch_size"], 8)
                    self.assertEqual(estimator.kwargs["random_state"], 42)
                    self.assertFalse(estimator.kwargs["allow_auto_download"])
                    self.assertFalse(estimator.kwargs["use_amp"])
                    self.assertEqual(estimator.kwargs["norm_methods"], ["none", "power"])
                    with self.assertRaisesRegex(RuntimeError, "Only native Loop3"):
                        adapter.build_estimator("loop4", config)
            finally:
                sys.path[:] = old_path

    def test_native32_bitwise_prediction_and_generator_unchanged(self):
        data = arrays(features=8)
        reference = Estimator().fit(data["X_train"], data["y_train"])
        expected = reference.predict_proba(data["X_test"].copy())
        model = Estimator()
        got, audit = adapter.predict_estimator(model, data, model_key="tabiclv2")
        np.testing.assert_array_equal(got, expected)
        self.assertTrue(audit["native_generator_unchanged"])
        self.assertEqual(audit["supplemental_ensemble_count"], 0)
        self.assertEqual(model.ensemble_generator_.ensemble_configs_, reference.ensemble_generator_.ensemble_configs_)
        self.assertEqual(len(audit["forward_batches"]), 4)
        self.assertEqual(sum(b["batch_members"] for b in audit["forward_batches"]), 32)
        self.assertNotIn("transform", model.ensemble_generator_.__dict__)
        self.assertFalse(model.model_.pre or model.model_.post)

    def test_low_feature_real32_and_query_unchanged(self):
        data = arrays()
        before = {k: v.copy() for k, v in data.items()}
        model = Estimator()
        got, audit = adapter.predict_estimator(model, data, model_key="tabiclv1")
        self.assertEqual(got.shape, (5, 2))
        self.assertEqual(audit["native_ensemble_count"], 4)
        self.assertEqual(audit["supplemental_ensemble_count"], 28)
        self.assertEqual(audit["unique_configuration_count"], 32)
        self.assertTrue(audit["actual32_verified"])
        self.assertEqual(audit["actual_members_per_test_row"], 32)
        self.assertEqual(len(model.model_.seen_support), 32)
        for member, (X, y), test in zip(audit["member_audits"], model.model_.seen_support, model.model_.seen_queries):
            self.assertEqual(member["forward_contributions"], 1)
            self.assertEqual(member["aggregation_contributions"], 1)
            scale = .7 if member["configuration"]["normalization"] == "power" else 1
            np.testing.assert_array_equal(test, data["X_test"] * scale)
            original = np.asarray(data["X_train"] * scale, dtype=np.float32)
            self.assertEqual(sorted(X[:, 0]), sorted(original[:, 0]))
            order = [int(np.flatnonzero(original[:, 0] == v)[0]) for v in X[:, 0]]
            cls = member["configuration"]["class_order"]
            np.testing.assert_array_equal(y, np.asarray(cls)[data["y_train"]][order])
        for key in data:
            np.testing.assert_array_equal(data[key], before[key])
        self.assertEqual(sum(map(len, model.ensemble_generator_.ensemble_configs_.values())), 4)
        self.assertFalse(audit["statistically_independent_predictions_claimed"])

    def test_deterministic_support_only_planning(self):
        data = arrays(features=2)
        g = Generator(data["X_train"], data["y_train"], 2, 2)
        first, second = adapter.prepare_members(g), adapter.prepare_members(g)
        self.assertEqual(first[2], second[2])
        self.assertEqual(sum(len(x) for x in first[2].values()), 32)
        self.assertEqual(sum(map(len, g.ensemble_configs_.values())), 8)

    def test_insufficient_distinct_support_fails_closed(self):
        g = Generator(np.ones((4, 1), dtype=np.float32), np.zeros(4, dtype=int), 1, 1)
        with self.assertRaisesRegex(RuntimeError, "Cannot realize 32"):
            adapter.prepare_members(g)

    def test_test_labels_rejected(self):
        data = arrays()
        data["y_test"] = np.zeros(5)
        with self.assertRaisesRegex(RuntimeError, "Only support labels"):
            adapter.predict_estimator(Estimator(), data, model_key="tabiclv2")

    def test_silent_forward_bypass_rejected(self):
        class Bypass(Estimator):
            def _batch_forward(self, Xs, ys, feature_shuffles=None):
                return np.ones((len(Xs), 5, 2), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "Not all members actually forwarded"):
            adapter.predict_estimator(Bypass(), arrays(), model_key="tabiclv2")

    def test_wrong_query_coverage_rejected(self):
        class Dropped(Estimator):
            def _batch_forward(self, Xs, ys, feature_shuffles=None):
                return super()._batch_forward(Xs[:, :-1], ys, feature_shuffles)
        with self.assertRaisesRegex(RuntimeError, "omitted members/support/query"):
            adapter.predict_estimator(Dropped(), arrays(), model_key="tabiclv2")

    def test_output_padding_rejected(self):
        class Padding(Estimator):
            def predict_proba(self, X):
                data = self.ensemble_generator_.transform(X, mode="both")
                outputs = [self._batch_forward(a, b) for a, b in data.values()]
                value = np.concatenate(outputs)
                value[-1] = value[0]
                shuffles = [c for group in self.ensemble_generator_.class_shuffles_.values() for c in group]
                return self._aggregate_ensemble_outputs(value, shuffles)
        with self.assertRaisesRegex(RuntimeError, "did not come from"):
            adapter.predict_estimator(Padding(), arrays(), model_key="tabiclv2")

    def test_nan_outputs_rejected_and_hooks_cleaned(self):
        class NaN(Estimator):
            def _batch_forward(self, Xs, ys, feature_shuffles=None):
                result = super()._batch_forward(Xs, ys, feature_shuffles)
                result[:] = np.nan
                return result
        model = NaN()
        with self.assertRaisesRegex(RuntimeError, "Invalid native member outputs"):
            adapter.predict_estimator(model, arrays(), model_key="tabiclv2")
        self.assertFalse(model.model_.pre or model.model_.post)
        self.assertNotIn("_batch_forward", model.__dict__)
        self.assertNotIn("transform", model.ensemble_generator_.__dict__)

    def test_taffy_checkpoint_validation(self):
        config = {"shared_depth_icl_enabled": True, "shared_depth_icl_dataset_conditioned": True,
                  "shared_depth_icl_num_passes": 3, "shared_depth_icl_rho": 1.,
                  "icl_num_blocks": 12, "max_classes": 10}
        adapter._check_taffy_config(config, 3)
        with self.assertRaisesRegex(RuntimeError, "checkpoint config mismatch"):
            adapter._check_taffy_config(config, 4)
        config["max_classes"] = 0
        with self.assertRaisesRegex(RuntimeError, "Regression checkpoint"):
            adapter._check_taffy_config(config, 3)

    def test_loop_hooks_count_all_12_blocks_and_cleanup(self):
        class Block(Module):
            def forward(self):
                return None
        class Encoder(Module):
            shared_depth_enabled = shared_depth_dataset_conditioned = True
            shared_depth_num_passes, shared_depth_rho = 3, 1.
            def __init__(self):
                super().__init__()
                self.blocks = [Block() for _ in range(12)]
            def forward(self):
                for _ in range(self.shared_depth_num_passes):
                    for block in self.blocks:
                        block()
                return SimpleNamespace(dtype="torch.float32")
        encoder = Encoder()
        estimator = SimpleNamespace(model_config_={"shared_depth_icl_enabled": True,
            "shared_depth_icl_dataset_conditioned": True, "shared_depth_icl_num_passes": 3,
            "shared_depth_icl_rho": 1., "icl_num_blocks": 12, "max_classes": 10},
            model_=SimpleNamespace(icl_predictor=SimpleNamespace(tf_icl=encoder)))
        handles, calls = [], []
        adapter._loop_hooks(estimator, 3, handles, calls)
        encoder()
        self.assertEqual(calls, [[3] * 12])
        encoder.shared_depth_num_passes = 2
        with self.assertRaisesRegex(RuntimeError, "Wrong realized Loop3"):
            encoder()
        for handle in handles:
            handle.remove()
        self.assertFalse(encoder.pre or encoder.post or any(b.post for b in encoder.blocks))

    def test_source_runtime_generate_method_is_compatible(self):
        path = Path(__file__).resolve().parents[2] / "classification_ensemble_audit_20260921_v1/tabicl_runtime/preprocessing.py"
        if not path.is_file():
            self.skipTest("Read-only runtime audit snapshot not present")
        parsed = ast.parse(path.read_text())
        shuffler = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "Shuffler")
        recursion = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "RecursionLimitManager")
        gen = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "EnsembleGenerator")
        method = next(n for n in gen.body if isinstance(n, ast.FunctionDef) and n.name == "_generate_ensemble")
        namespace = dict(random=random, itertools=itertools, np=np, OrderedDict=OrderedDict, sys=sys, deepcopy=deepcopy,
                         Optional=__import__("typing").Optional, List=__import__("typing").List)
        exec(compile(ast.Module(body=[recursion, shuffler, method], type_ignores=[]), str(path), "exec"), namespace)
        for features, classes, expected in [(1, 2, 4), (2, 2, 8), (3, 3, 18), (8, 2, 32)]:
            data = arrays(features, classes)
            g = Generator(data["X_train"], data["y_train"], features, classes)
            g.n_features_in_, g.n_classes_, g.n_estimators = features, classes, 32
            g.feat_shuffle_method, g.class_shuffle_method = "latin", "shift"
            g.random_state, g.rng_, g.norm_methods_ = 42, random.Random(42), ["none", "power"]
            g.ensemble_configs_, g.feature_shuffles_, g.class_shuffles_ = namespace["_generate_ensemble"](g)
            native, _, members, _ = adapter.prepare_members(g)
            self.assertEqual(native, expected)
            self.assertEqual(sum(map(len, members.values())), 32)


if __name__ == "__main__":
    unittest.main()
