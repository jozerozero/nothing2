"""CPU mocks/config tests only; no actual estimator or GPU smoke is claimed."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import table6_missing190_fit as adapter

WORKSPACE = Path(__file__).resolve().parents[3]
R19 = WORKSPACE / 'outputs/table6_remaining19_standard_hpo_bg8_20260912_v1'
TALENT = WORKSPACE / 'tmp/limix-paper-repro/TALENT'


def request(method='TabM'):
    section = 'fit' if method in adapter.CPU_METHODS else 'training'
    return {'mode': 'trial', 'method': method, 'seed': 0, 'cpus': 4,
            'max_epoch': 200, 'batch_size': 1024, 'work_dir': '/unused/logical/receipt',
            'row': {'dataset': 'toy', 'task_kind': 'classification', 'input_fingerprint': 'frozen'},
            'config': {'model': {}, section: {'n_bins': 17}}}


class Trial:
    def suggest_int(self, name, low, high, **kwargs):
        return low

    def suggest_float(self, name, low, high, **kwargs):
        return low

    def suggest_categorical(self, name, values):
        return values[0]


class ConfigTests(unittest.TestCase):
    def test_registry_exact_nine_and_tree_cpu_lane(self):
        self.assertEqual(len(adapter.METHODS), 9)
        self.assertEqual(adapter.CPU_METHODS, {'CatBoost', 'LightGBM', 'XGBoost'})

    def test_tabm_fixed_fields_added_without_sampled_defaults(self):
        sampled = {'model': {'num_embeddings': {'d_embedding': 17}, 'backbone': {'d_block': 128}},
                   'training': {'n_bins': 13}}
        before = copy.deepcopy(sampled)
        out = adapter.postprocess_sample('TabM', sampled)
        self.assertEqual(sampled, before)
        self.assertEqual(out['model']['num_embeddings'], {'d_embedding': 17, 'type': 'PLREmbeddings', 'lite': True})
        self.assertEqual(out['model']['backbone'], {'d_block': 128, 'type': 'MLP'})
        self.assertEqual((out['model']['k'], out['model']['arch_type']), (32, 'tabm'))
        self.assertNotIn('lr', out['training'])

    def test_tabr_fixed_fields_preserve_sampled_values(self):
        value = {'model': {'num_embeddings': {}, 'dropout0': .4, 'dropout1': .3}}
        out = adapter.postprocess_sample('TabR', value)['model']
        self.assertEqual(out['dropout0'], .4)
        self.assertEqual(out['dropout1'], .3)
        self.assertEqual(out['activation'], 'ReLU')
        self.assertEqual(out['normalization'], 'LayerNorm')
        self.assertEqual(out['mixer_normalization'], 'auto')

    def test_all_nine_real_packaged_search_spaces_sample_on_cpu(self):
        if not (R19/'fit.py').is_file() or not TALENT.is_dir():
            self.skipTest('local pinned fit / TALENT source snapshot not present')
        spec = importlib.util.spec_from_file_location('_test_r19_sampler', R19/'fit.py')
        fit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fit)
        common = types.ModuleType('common')
        common.METHODS = adapter.METHODS
        with mock.patch.dict(sys.modules, {'common': common}), mock.patch.object(adapter, 'initialize', return_value=fit):
            for method in adapter.METHODS:
                with self.subTest(method=method):
                    config = adapter.suggest_config(Trial(), method, 'classification', TALENT)
                    section = 'fit' if method in adapter.CPU_METHODS else 'training'
                    self.assertEqual(config[section]['n_bins'], 2)
                    adapter.runtime_request({**request(method), 'config': config})
                    default = adapter.default_config(method, TALENT)
                    self.assertEqual(default[section]['n_bins'], 2)
                    if method in adapter.CPU_METHODS:
                        self.assertNotIn('training', config)
                        self.assertNotIn('training', default)

    def test_tree_nbins_alias_is_runtime_only_and_thread_limited(self):
        for method in adapter.CPU_METHODS:
            with self.subTest(method=method):
                req = request(method)
                before = copy.deepcopy(req)
                out = adapter.runtime_request(req)
                self.assertEqual(req, before)
                self.assertEqual(out['config']['fit']['n_bins'], 17)
                self.assertEqual(out['config']['training']['n_bins'], 17)
                key = 'thread_count' if method == 'CatBoost' else 'n_jobs'
                self.assertEqual(out['config']['model'][key], 4)
                self.assertNotIn('task_type', out['config']['model'])

    def test_catboost_no_duplicate_native_task_type(self):
        req = request('CatBoost')
        req['config']['model']['task_type'] = 'CPU'
        with self.assertRaisesRegex(ValueError, 'constructor supplies'):
            adapter.runtime_request(req)

    def test_invalid_classical_config_gpu_request_and_bins_rejected(self):
        cases = []
        for bins in (None, True, 1, 257):
            req = request('XGBoost')
            req['config']['fit']['n_bins'] = bins
            cases.append(req)
        req = request('XGBoost')
        req['config']['model']['tree_method'] = 'gpu_hist'
        cases.append(req)
        req = request('CatBoost')
        req['config']['training'] = {'n_bins': 19}
        cases.append(req)
        for req in cases:
            with self.assertRaises(ValueError):
                adapter.runtime_request(req)


class ApiAdapterTests(unittest.TestCase):
    def fake_api(self):
        package = types.ModuleType('TALENT')
        package.__path__ = []
        api = types.ModuleType('TALENT.api')
        api.build_args = mock.Mock(side_effect=lambda name, **kwargs: types.SimpleNamespace(
            config=kwargs['config'], tune_metric=kwargs.get('tune_metric')))
        package.api = api
        return package, api

    def test_native_accuracy_methods_disable_only_unsupported_override(self):
        package, api = self.fake_api()
        old = api.build_args
        seen = []

        def original(req, bundle):
            args = api.build_args(adapter.METHODS[req['method']], config=req['config'], tune_metric='Accuracy')
            seen.append(args.tune_metric)
            return {'config': req['config'], 'effective_config': args.config, 'score': .5}

        with mock.patch.dict(sys.modules, {'TALENT': package, 'TALENT.api': api}):
            for method in ('TabNet', 'TabCaps', 'TabM', 'SwitchTab', 'TabTransformer'):
                out = adapter.run_talent_adapter(original, request(method), object())
                self.assertEqual(out['config'], request(method)['config'])
                self.assertIs(api.build_args, old)
        self.assertEqual(seen, [None, None, 'Accuracy', 'Accuracy', 'Accuracy'])

    def test_classical_alias_keeps_selected_config_receipt_unchanged(self):
        package, api = self.fake_api()
        req = request('XGBoost')

        def original(adapted, bundle):
            args = api.build_args('xgboost', config=adapted['config'], tune_metric='Accuracy')
            self.assertEqual(args.config['fit']['n_bins'], 17)
            self.assertEqual(args.config['training']['n_bins'], 17)
            return {'config': adapted['config'], 'effective_config': args.config}

        with mock.patch.dict(sys.modules, {'TALENT': package, 'TALENT.api': api}):
            out = adapter.run_talent_adapter(original, req, object())
        self.assertEqual(out['config'], req['config'])
        self.assertNotIn('training', out['config'])
        self.assertEqual(out['effective_config']['model']['n_jobs'], 4)

    def test_build_args_patch_restored_after_fit_exception(self):
        package, api = self.fake_api()
        old = api.build_args
        with mock.patch.dict(sys.modules, {'TALENT': package, 'TALENT.api': api}):
            with self.assertRaisesRegex(RuntimeError, 'native fit failed'):
                adapter.run_talent_adapter(mock.Mock(side_effect=RuntimeError('native fit failed')),
                                           request('TabCaps'), object())
            self.assertIs(api.build_args, old)


class FaissTests(unittest.TestCase):
    def test_rocm_uses_exact_cpu_l2_and_returns_original_query_device(self):
        class Tensor:
            device = 'cuda:0'
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return [[1., 2.]]

        index = types.SimpleNamespace(reset=mock.Mock(), add=mock.Mock(),
                                      search=mock.Mock(return_value=([[.5]], [[4]])))
        faiss = types.SimpleNamespace(IndexFlatL2=mock.Mock(return_value=index),
                                      GpuIndexFlatL2=mock.Mock(), omp_set_num_threads=mock.Mock())
        np = types.SimpleNamespace(float32='float32', ascontiguousarray=mock.Mock(side_effect=lambda a, dtype: a))
        torch = types.SimpleNamespace(version=types.SimpleNamespace(hip='6.2'),
                                      as_tensor=lambda values, device: (values, device))
        mode = adapter.install_tabr_cpu_faiss(4, faiss_module=faiss, torch_module=torch, numpy_module=np)
        self.assertTrue(mode.startswith('cpu_exact_l2'))
        wrapped = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 2)
        wrapped.reset()
        wrapped.add(Tensor())
        result = wrapped.search(Tensor(), 1)
        self.assertEqual(result, (([[.5]], 'cuda:0'), ([[4]], 'cuda:0')))
        faiss.omp_set_num_threads.assert_called_once_with(4)
        index.add.assert_called_once_with([[1., 2.]])
        index.search.assert_called_once_with([[1., 2.]], 1)

    def test_cuda_native_faiss_not_replaced(self):
        gpu = mock.Mock()
        faiss = types.SimpleNamespace(GpuIndexFlatL2=gpu)
        torch = types.SimpleNamespace(version=types.SimpleNamespace(hip=None))
        mode = adapter.install_tabr_cpu_faiss(4, faiss_module=faiss, torch_module=torch, numpy_module=object())
        self.assertEqual(mode, 'native_faiss_gpu')
        self.assertIs(faiss.GpuIndexFlatL2, gpu)


class ScopeTests(unittest.TestCase):
    def plan_for_request(self, req):
        return {'pairs': [{'method': req['method'], 'dataset': req['row']['dataset']}],
                'rows': [copy.deepcopy(req['row'])]}

    def test_scope_and_tab_caps_classification_only(self):
        req = request('TabCaps')
        plan = self.plan_for_request(req)
        adapter.validate_request(req, plan)
        for changed in ({'task_kind': 'regression'}, {'dataset': 'another'}, {'input_fingerprint': 'changed'}):
            bad = copy.deepcopy(req)
            bad['row'].update(changed)
            with self.assertRaises(ValueError):
                adapter.validate_request(bad, plan)

    def test_seed_and_formal_epoch_contract(self):
        req = request()
        plan = self.plan_for_request(req)
        for change in ({'seed': 1}, {'seed': True}, {'seed': 15}, {'max_epoch': 2},
                       {'batch_size': 512}, {'tune_threshold': True}):
            with self.assertRaises(ValueError):
                adapter.validate_request({**req, **change}, plan)
        adapter.validate_request({**req, 'mode': 'final', 'seed': 14}, plan)

    def test_smoke_is_explicit_short_validation_fit(self):
        req = request()
        plan = self.plan_for_request(req)
        adapter.validate_request({**req, 'max_epoch': 2}, plan, diagnostic_smoke=True)
        for changed in ({}, {'mode': 'final', 'max_epoch': 2}, {'seed': 1, 'max_epoch': 2}):
            with self.assertRaises(ValueError):
                adapter.validate_request({**req, **changed}, plan, diagnostic_smoke=True)

    def test_cpu_visibility_is_cleared_before_legacy_execute(self):
        req = request('CatBoost')
        plan = self.plan_for_request(req)
        plan['plan_id'] = 'test-plan'
        mask = {'CPU_ONLY': '1', 'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': '-1',
                'ROCR_VISIBLE_DEVICES': '-1', 'GPU_DEVICE_ORDINAL': '-1'}

        def legacy_execute(request):
            self.assertEqual({key: os.environ[key] for key in mask}, mask)
            return {'complete': True}

        with mock.patch.object(adapter, 'initialize', return_value=types.SimpleNamespace(execute=legacy_execute, os=os)), \
                mock.patch.object(adapter, '_PLAN', plan), mock.patch.dict(os.environ, dict.fromkeys(mask, '0')):
            out = adapter.execute(req)
        self.assertEqual(out['missing190_plan_id'], 'test-plan')
        self.assertFalse(out['diagnostic_smoke'])

    def test_pinned_execute_cannot_clobber_rocm_cpu_mask_before_data_load(self):
        if not (R19/'fit.py').is_file():
            self.skipTest('local pinned R19 fixture absent')
        spec = importlib.util.spec_from_file_location('_test_cpu_mask_r19', R19/'fit.py')
        fit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fit)
        req = request('CatBoost')
        common = types.ModuleType('common')
        common.CPU_METHODS, common.METHODS = adapter.CPU_METHODS, adapter.METHODS
        common.object_digest = adapter.object_digest
        data = types.ModuleType('data')
        def load(row):
            self.assertEqual({key: os.environ[key] for key in adapter.CPU_MASK}, adapter.CPU_MASK)
            self.assertIsNot(fit.os, os)
            self.assertIsNot(os.environ, fit.os.environ)
            raise RuntimeError('mask checked before data load')
        data.load = load
        with mock.patch.dict(sys.modules, {'common': common, 'data': data}), \
                mock.patch.object(adapter, 'initialize', return_value=fit), \
                mock.patch.object(adapter, '_PLAN', self.plan_for_request(req)), \
                mock.patch.dict(os.environ, dict.fromkeys(adapter.CPU_MASK, '0')):
            with self.assertRaisesRegex(RuntimeError, 'mask checked before data load'):
                adapter.execute(req)
        self.assertIs(fit.os, os)


class InitializationTests(unittest.TestCase):
    def setUp(self):
        if not (R19/'fit.py').is_file():
            self.skipTest('local pinned R19 fixture absent')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        out = Path(self.tmp.name) / adapter.NAME
        out.mkdir()
        methods = list(adapter.METHODS)
        rows = [{'dataset': f'row{i}', 'task_kind': 'classification', 'input_fingerprint': f'fingerprint{i}'}
                for i in range(22)]
        pairs = [{'method': methods[i % 9], 'dataset': rows[i // 9]['dataset'],
                  'task_kind': 'classification', 'input_fingerprint': rows[i // 9]['input_fingerprint'],
                  'hpo_trials': 100, 'seeds': list(range(15))} for i in range(190)]
        self.plan = {'name': adapter.NAME, 'applicable_pair_target': 190, 'methods': adapter.METHODS,
                     'cpu_methods': sorted(adapter.CPU_METHODS), 'hpo_trials': 100, 'hpo_seed': 0,
                     'seeds': list(range(15)), 'max_epoch': 200, 'batch_size': 1024,
                     'source_contract': {'remaining19_plan_id': adapter.R19_PLAN_ID},
                     'rows': rows, 'pairs': pairs, 'output_root': str(out)}
        self.plan['plan_id'] = adapter.object_digest(self.plan)
        self.path = out/'plan.json'
        self.path.write_text(json.dumps(self.plan))

    def test_registry_and_output_injected_before_data_import(self):
        before_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), before_path))
        with mock.patch.dict(sys.modules), mock.patch.object(adapter, '_FIT', None), \
                mock.patch.object(adapter, '_PLAN', None), mock.patch.object(adapter, '_PLAN_PATH', None):
            sys.modules.pop('common', None)
            sys.modules.pop('data', None)
            fit = adapter.initialize(self.path, R19)
            common = sys.modules['common']
            self.assertEqual(common.OUT, self.path.parent.resolve())
            self.assertEqual(common.METHODS, adapter.METHODS)
            self.assertEqual(common.CPU_METHODS, adapter.CPU_METHODS)
            self.assertNotIn('data', sys.modules)
            self.assertIs(adapter.initialize(), fit)

    def test_existing_data_import_rejected_instead_of_reusing_old_output(self):
        with mock.patch.dict(sys.modules, {'data': types.ModuleType('data')}), \
                mock.patch.object(adapter, '_FIT', None), mock.patch.object(adapter, '_PLAN_PATH', None):
            with self.assertRaisesRegex(ValueError, 'data module already imported'):
                adapter.initialize(self.path, R19)

    def test_bad_plan_digest_and_old_output_namespace_rejected(self):
        changed = copy.deepcopy(self.plan)
        changed['max_epoch'] = 2
        with self.assertRaisesRegex(ValueError, 'digest'):
            adapter.validate_plan(changed)
        changed = copy.deepcopy(self.plan)
        changed['output_root'] = str(R19)
        changed.pop('plan_id')
        changed['plan_id'] = adapter.object_digest(changed)
        with self.assertRaisesRegex(ValueError, 'namespace'):
            adapter.validate_plan(changed)


if __name__ == '__main__':
    unittest.main()
