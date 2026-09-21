"""Local CPU-only contract tests; no official checkpoints or GPU required."""
from collections import OrderedDict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import tabfm_default_one as worker


class Tensor:
    def __init__(self, array):
        self.value = np.asarray(array)
        self.shape, self.dtype = self.value.shape, self.value.dtype
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


class Model:
    def __init__(self):
        self.hooks = []
        self.training, self.max_classes = False, 10
        self.parameter = SimpleNamespace(data_ptr=lambda: 1234, _version=0,
                                         shape=(1,), dtype='torch.bfloat16')
    def parameters(self): return iter([self.parameter])
    def register_forward_hook(self, hook, with_kwargs=False):
        assert with_kwargs
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))
    def forward(self, rows, support, batch=1, features=2):
        args = (Tensor(np.zeros((batch, rows, features))),
                Tensor(np.zeros((batch, rows))), Tensor(np.repeat(support, batch)))
        output = Tensor(np.zeros((batch, rows, 2)))
        for hook in self.hooks:
            hook(self, args, {}, output)


class Estimator:
    def __init__(self, *, fail=None):
        self.model, self.fail = Model(), fail
        self.n_estimators, self.max_num_features, self.max_num_rows = 32, 500, None
        self.classes_ = np.array([0, 1])
        self.y_encoder_ = SimpleNamespace(inverse_transform=lambda x: x)
        self.y_scaler_ = SimpleNamespace(mean_=np.array([1000.]), scale_=np.array([20.]))
        self.ensemble_generator_ = SimpleNamespace(norm_methods_=['none', 'power'],
            ensemble_configs_=OrderedDict((key, [(np.array([0, 1]), 0, None, None)] * 16)
                                          for key in ('none', 'power')))
    def fit(self, x, y):
        self.seen_x, self.seen_y = x.copy(), y.copy()
        if self.fail == 'fit_forward':
            self.model.forward(len(y) + 2, len(y))
        return self
    def _batch_forward(self, x):
        support = len(self.seen_y)
        count = 31 if self.fail == 'missing_forward' else 32
        for _ in range(count):
            rows = support + len(x) - (1 if self.fail == 'short_query' else 0)
            self.model.forward(rows, support)
        result = np.zeros((32, len(x), 2))
        if self.fail == 'nan': result[0, 0, 0] = np.nan
        return result
    def predict_proba(self, x):
        self._batch_forward(x)
        p = np.tile([0.8, 0.2], (len(x), 1))
        if self.fail == 'bad_probabilities': p[0] = [1., 1.]
        return p
    def predict(self, x):
        self._batch_forward(x)
        return np.repeat(1000., len(x))


class HierarchyEstimator(Estimator):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.random_state, self.batch_size = 42, 1
        self.cache_context = self.enable_nnls = False
        self.class_shift = self.average_logits = True
        self.softmax_temperature = 0.9
    def fit(self, x, y):
        super().fit(x, y)
        self.classes_ = np.unique(y)
        return self
    def predict_proba(self, x):
        self._batch_forward(x)
        return np.ones((len(x), len(self.classes_))) / len(self.classes_)


def identity(path, content=None):
    path = path.resolve()
    if content is not None:
        path.write_bytes(content)
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, sha256=worker.sha_file(path))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tx = pd.DataFrame({'N0': [1., 2., 3., 4.], 'C0': ['a', 'b', 'a', 'b']})
        self.ys = np.array([0, 1, 0, 1])
        self.vx = self.tx.iloc[:2].copy()

    def call(self, estimator=None, **kwargs):
        estimator = estimator or Estimator()
        prediction, audit = worker.native_predict(estimator, 'classification', self.tx, self.ys, self.vx, **kwargs)
        return estimator, prediction, audit

    def test_actual_members_not_unique_configuration_claim(self):
        estimator, pred, audit = self.call()
        np.testing.assert_array_equal(pred, [0, 0])
        self.assertEqual(audit['actual_ensemble_count'], 32)
        self.assertEqual(audit['unique_configuration_count'], 2)
        self.assertTrue(audit['native_member_forward_verified'])
        self.assertFalse(audit['strict32_protocol_claimed'])
        self.assertEqual(len(audit['forward_calls']), 32)
        self.assertEqual(audit['batch_predictions'][0]['members'], 32)
        self.assertEqual(estimator.model.hooks, [])
        self.assertNotIn('_batch_forward', estimator.__dict__)

    def test_return_probabilities_avoids_second_predict(self):
        _, pred, audit = self.call(return_probabilities=True)
        self.assertEqual(pred.shape, (2, 2))
        self.assertEqual(len(audit['forward_calls']), 32)
        self.assertEqual(audit['support_rows'], 4)
        self.assertTrue(audit['full_test_split'])

    def test_hierarchy_callback_integrates_native_full_forward_audit(self):
        from tabfm_hierarchical import hierarchical_predict_proba
        tx = pd.DataFrame({'N0': np.arange(22, dtype=float), 'C0': ['a', 'b'] * 11})
        ys = np.repeat(np.arange(11), 2)
        model = Model()
        probabilities, audit = hierarchical_predict_proba(model, tx, ys, tx.iloc[:3],
            classifier_factory=HierarchyEstimator,
            node_predictor=lambda est, sx, sy, qx, info: worker.native_predict(
                est, 'classification', sx, sy, qx, return_probabilities=True))
        self.assertEqual(probabilities.shape, (3, 11))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
        self.assertEqual(audit['node_count'], 3)
        self.assertEqual(audit['actual_ensemble_count'], 32)
        self.assertEqual(audit['actual_ensemble_count_scope'], 'per hierarchy node')
        self.assertEqual(audit['total_member_forward_count'], 96)
        self.assertTrue(audit['native_member_forward_verified'])
        self.assertEqual(model.hooks, [])

    def test_loader_only_device_task_path_no_estimator_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            model, calls = Model(), []
            class NativeConstructor:
                def __init__(self, model, n_estimators=32, max_num_rows=None):
                    self.model, self.n_estimators, self.max_num_rows = model, n_estimators, max_num_rows
            modules = {
                'tabfm': SimpleNamespace(__file__=str(root / 'tabfm/__init__.py'), __version__='1.0.1',
                    TabFMClassifier=NativeConstructor, TabFMRegressor=NativeConstructor),
                'tabfm.src.pytorch.tabfm_v1_0_0': SimpleNamespace(__file__=str(root / 'tabfm/src/pytorch/tabfm_v1_0_0.py'),
                    load=lambda **kw: (calls.append(kw), model)[1]),
                'tabfm.src.classifier_and_regressor': SimpleNamespace(__file__=str(root / 'tabfm/src/classifier_and_regressor.py'))}
            with patch.object(worker.importlib, 'import_module', side_effect=lambda n: modules[n]):
                estimator, audit = worker.load_native({'official_source': {'path': str(root)}}, 'regression', root)
            self.assertEqual(calls, [{'model_type': 'regression', 'checkpoint_path': str(root), 'device': 'cuda:0'}])
            self.assertEqual(estimator.n_estimators, 32)
            self.assertEqual(audit['overrides'], {})
            self.assertIsNone(audit['loader_dtype_override'])
            self.assertEqual(audit['model_parameter_dtypes'], ['torch.bfloat16'])

    def test_raw_mixed_dataframe_preserved(self):
        estimator, _, _ = self.call()
        pd.testing.assert_frame_equal(estimator.seen_x, self.tx)
        np.testing.assert_array_equal(estimator.seen_y, self.ys)
        self.assertEqual(str(estimator.seen_x.C0.dtype), 'object')

    def test_classification_cache_only_audits_never_becomes_model_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            raw = identity(root / 'original.raw', b'frozen original split')
            encoded_train = np.arange(8, dtype=np.float32).reshape(4, 2) + 100
            encoded_test = np.arange(4, dtype=np.float32).reshape(2, 2) + 200
            yt = np.array([0, 1])
            np.savez(root / 'cache.npz', X_train=encoded_train, y_train=self.ys,
                     X_test=encoded_test, y_test=yt)
            cache = identity(root / 'cache.npz')
            rows = []
            for suite, count in worker.CLASS_COUNTS.items():
                for _ in range(count):
                    rows.append(dict(dataset_index=len(rows), suite=suite, dataset=f'fixture_{len(rows)}',
                        source_path=str(root), cache=cache, input_fingerprint=worker.common.object_digest(cache),
                        train_rows=4, test_rows=2, features=2, classes=2, class_labels=['a', 'b']))
            manifest = {'rows': rows, 'membership_count': 457}
            official = SimpleNamespace(
                talent_split=lambda _: (self.tx.copy(), self.ys.copy(), self.vx.copy(), yt.copy(), {'frozen': True}),
                drop_missing_targets=lambda tx, ys, vx, yt: (tx, ys, vx, yt, {'dropped': 0}),
                preprocess_labels=lambda ys, yt: (ys, yt, ['a', 'b']),
                preprocess_features=lambda tx, vx: (encoded_train, encoded_test, {'audit_only': True}))
            canonical = SimpleNamespace(canonicalize_features=lambda tx, vx: (tx, vx, {'fit_split': 'support_only'}))
            campaign = {'classification_manifest': 'class', 'regression_manifest': 'reg',
                        'classification_raw_inputs': {'0': [raw]}}
            with patch.object(worker, 'verify_manifest', side_effect=lambda name: manifest if name == 'class' else {}), \
                 patch.object(worker, 'import_data_helpers', return_value=(official, canonical)):
                _, row, tx, ys, vx, actual_yt, audit, hashes = worker.load_classification(campaign, 0, np, pd)
            pd.testing.assert_frame_equal(tx, self.tx)
            pd.testing.assert_frame_equal(vx, self.vx)
            np.testing.assert_array_equal(ys, self.ys)
            np.testing.assert_array_equal(actual_yt, yt)
            self.assertTrue(audit['cache_exact_match'])
            self.assertFalse(audit['cached_encoded_features_fed_to_model'])
            self.assertEqual(hashes, [raw])

    def test_regression_loader_never_applies_outer_taffy_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = identity(Path(directory) / 'data.raw', b'original')
            row = {'input_files': [raw]}
            ys, yt = np.array([100., 200., 300., 400.]), np.array([150., 250.])
            def forbidden(*_):
                raise AssertionError('Outer Taffy target transform must not be called')
            loaded = (self.tx, ys, self.vx, yt, {}, forbidden, {'source': 'frozen_but_unused'})
            with patch.object(worker.common, 'verify_file', return_value=Path(directory) / 'manifest'), \
                 patch.object(worker.common, 'load_manifest', return_value=({}, row, {})), \
                 patch.object(worker.common, 'load_raw_data', return_value=loaded):
                _, _, _, actual_ys, _, actual_yt, audit, _ = worker.load_regression(
                    {'regression_manifest': {}}, 0, np, pd)
            np.testing.assert_array_equal(actual_ys, ys)
            np.testing.assert_array_equal(actual_yt, yt)
            self.assertFalse(audit['source_taffy_target_transform_applied'])

    def test_original_regression_units_enter_and_exit_native(self):
        estimator = Estimator()
        y = np.array([900., 1100., 950., 1050.])
        prediction, audit = worker.native_predict(estimator, 'regression', self.tx, y, self.vx)
        np.testing.assert_array_equal(estimator.seen_y, y)
        np.testing.assert_array_equal(prediction, [1000., 1000.])
        self.assertEqual(audit['native_target_scaler_mean'], [1000.])

    def test_bad_native_evidence_rejected_and_hooks_removed(self):
        for failure in ('fit_forward', 'missing_forward', 'short_query', 'nan', 'bad_probabilities'):
            with self.subTest(failure=failure):
                estimator = Estimator(fail=failure)
                with self.assertRaises(RuntimeError): self.call(estimator)
                self.assertEqual(estimator.model.hooks, [])
                self.assertNotIn('_batch_forward', estimator.__dict__)

    def test_instance_batch_method_restored(self):
        estimator = Estimator()
        original = estimator._batch_forward
        estimator._batch_forward = original
        self.call(estimator)
        self.assertIs(estimator._batch_forward, original)

    def test_raw_snapshot_metadata_sha_and_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw'
            record = identity(path, b'abc')
            metadata_only = {k: v for k, v in record.items() if k != 'sha256'}
            self.assertEqual(worker.raw_snapshot([metadata_only]), [record])
            with self.assertRaises(RuntimeError): worker.raw_snapshot([record, record])
            bad = dict(record, sha256='0' * 64)
            with self.assertRaises(RuntimeError): worker.raw_snapshot([bad])
            bad = dict(record, size_bytes=4)
            with self.assertRaises(RuntimeError): worker.raw_snapshot([bad])

    def test_cache_requires_exact_values_rows_and_dtypes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.npz'
            arr = np.arange(6, dtype=np.float32).reshape(3, 2)
            np.savez(path, X_train=arr)
            row = {'dataset': 'fixture', 'cache': identity(path)}
            self.assertIn('X_train', worker.exact_cache_check(row, {'X_train': arr}, np))
            for bad in (arr[::-1], arr.astype(np.float64), arr[:2]):
                with self.assertRaises(RuntimeError): worker.exact_cache_check(row, {'X_train': bad}, np)

    def test_manifest_digest_no_silent_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            value = {'rows': [1], 'manifest_id': worker.common.object_digest({'rows': [1]})}
            record = identity(path, json.dumps(value).encode())
            self.assertEqual(worker.verify_manifest(record), value)
            value['rows'] = [2]
            record = identity(path, json.dumps(value).encode())
            with self.assertRaises(RuntimeError): worker.verify_manifest(record)

    def test_weight_directory_must_be_exact_pinned_native_task_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = identity(root / 'config.json', b'{"max_classes":10}')
            w = identity(root / 'model.safetensors', b'not loaded in CPU test')
            campaign = {'weights': {'classification': {'directory': str(root), 'config': c, 'weights': w}}}
            found, config = worker.weight_directory(campaign, 'classification')
            self.assertEqual(found, root.resolve())
            self.assertEqual(config['max_classes'], 10)

    def test_atomic_publication_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            worker.common.publish_new(path, {'complete': True})
            with self.assertRaises(FileExistsError): worker.common.publish_new(path, {'complete': False})
            self.assertTrue(json.loads(path.read_text())['complete'])

    def test_error_receipt_is_explicit_and_rethrow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            with patch.object(worker, 'evaluate', side_effect=RuntimeError('fixture')):
                with self.assertRaises(RuntimeError):
                    worker.main(['--campaign', 'unused', '--task-kind', 'regression',
                                 '--dataset-index', '7', '--output', str(path)])
            receipt = json.loads(path.read_text())
            self.assertFalse(receipt['complete'])
            self.assertEqual(receipt['status'], 'error')
            self.assertEqual(receipt['dataset_index'], 7)


if __name__ == '__main__':
    unittest.main()
