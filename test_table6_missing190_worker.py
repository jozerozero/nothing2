"""CPU-only coordinator tests; these do NOT claim any model/GPU smoke passed."""
from copy import deepcopy
from pathlib import Path
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import table6_missing190_plan as plan_builder
import table6_missing190_worker as worker
from test_table6_missing190_plan import fixtures

WORKSPACE = Path(__file__).resolve().parents[3]
R19 = Path(os.environ.get('T6_BASE_STAGE', str(WORKSPACE/'outputs/table6_remaining19_standard_hpo_bg8_20260912_v1')))
SHORT = Path(os.environ.get('T6_SHORT_SOURCE', str(WORKSPACE/'outputs/table6_gpu_short2h_20260914_v1/short_worker.py')))


class Study:
    def __init__(self):
        self.trials, self.user_attrs = [], {}
        self.asks = 0

    def ask(self):
        trial = types.SimpleNamespace(number=len(self.trials), state='RUNNING', user_attrs={}, value=None)
        trial.set_user_attr = lambda key, value: trial.user_attrs.__setitem__(key, deepcopy(value))
        self.trials.append(trial); self.asks += 1
        return trial

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def get_trials(self, deepcopy=False):
        return list(self.trials)

    def tell(self, number, value=None, state=None):
        self.trials[number].state = state or 'COMPLETE'
        self.trials[number].value = value

    @property
    def best_trial(self):
        return max((t for t in self.trials if t.state == 'COMPLETE'), key=lambda t: t.value)


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='missing190-worker-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.out = self.root/plan_builder.NAME
        fixed, remaining = fixtures()
        for value in (fixed, remaining):
            value['plan_id'] = plan_builder.object_digest(value)
        sources = {key: {'path': str(self.root/key/'plan.json'), 'sha256': 'a'*64, 'size_bytes': 1}
                   for key in ('fixedbest_plan', 'remaining19_plan')}
        with patch.object(plan_builder, 'FIXED_PLAN_ID', fixed['plan_id']), \
                patch.object(plan_builder, 'R19_PLAN_ID', remaining['plan_id']):
            plan = plan_builder.build_documents(fixed, remaining, self.out, sources)
        plan['source_contract']['remaining19_plan_id'] = plan_builder.R19_PLAN_ID
        plan.pop('plan_id'); plan['plan_id'] = plan_builder.object_digest(plan)
        self.out.mkdir()
        self.plan_path = self.out/'plan.json'
        self.plan_path.write_text(json.dumps(plan))
        self.old_modules = dict(sys.modules)
        self.old_path = list(sys.path)
        self.old_bytecode = sys.dont_write_bytecode
        self.environ = patch.dict(os.environ, {'SLURM_JOB_ID': '1234', 'SLURM_PROCID': '0', 'SLURM_STEP_ID': '1'})
        self.environ.start()
        self.addCleanup(self.restore_imports)
        self.runtime = worker.Runtime(self.plan_path, R19, SHORT)
        self.runtime.short.BUDGET = self.runtime.short.Budget(10000, clock=lambda: 0)

    def restore_imports(self):
        self.environ.stop()
        sys.path[:] = self.old_path
        sys.dont_write_bytecode = self.old_bytecode
        for name in list(sys.modules):
            if name not in self.old_modules and (name in {'common', 'fit', 'data', 'optuna'} or name.startswith('_missing190')):
                sys.modules.pop(name, None)
        for name in ('common', 'fit', 'data', 'optuna'):
            if name in self.old_modules:
                sys.modules[name] = self.old_modules[name]

    def study(self):
        study = Study()
        sys.modules['optuna'] = types.SimpleNamespace(
            create_study=lambda **kwargs: study,
            samplers=types.SimpleNamespace(TPESampler=lambda **kwargs: kwargs),
            trial=types.SimpleNamespace(TrialState=types.SimpleNamespace(RUNNING='RUNNING', COMPLETE='COMPLETE', FAIL='FAIL')))
        self.runtime.fit.suggest_config = lambda trial, method, kind: {'model': {'trial': trial.number}, 'training': {'n_bins': 2}}
        return study

    def pair(self):
        pair = next(p for p in self.runtime.plan['pairs'] if p['method'] == 'TabM')
        row = next(r for r in self.runtime.plan['rows'] if r['dataset'] == pair['dataset'])
        return pair, row

    def test_isolated_new_output_and_registry(self):
        self.assertEqual(self.runtime.base.OUT, self.out)
        self.assertEqual(sys.modules['common'].OUT, self.out)
        self.assertEqual(set(self.runtime.short.GPU_METHODS), set(self.runtime.fit.METHODS))
        self.assertEqual(self.runtime.base.CPU_METHODS, {'CatBoost', 'LightGBM', 'XGBoost'})
        self.assertIs(sys.modules['fit'], self.runtime.fit)
        self.assertEqual(self.runtime.load_plan(), self.runtime.plan)
        self.assertEqual(worker.digest(R19/'worker.py'), worker.PINNED['worker.py'])
        self.assertNotIn('data', sys.modules)

    def test_exact_child_rewrite_and_no_legacy_fit_launch(self):
        argv = [sys.executable, '-B', str(R19/'fit.py'), '--request', '/r.json', '--response', '/s.json']
        for smoke in (False, True):
            value = worker.rewrite_child(argv, legacy_fit=R19/'fit.py', new_fit=self.runtime.fit_path,
                                         plan_path=self.plan_path, smoke=smoke)
            self.assertEqual(value[2], str(self.runtime.fit_path))
            self.assertIn(str(self.plan_path), value)
            self.assertEqual('--smoke' in value, smoke)
        argv[2] = '/another/fit.py'
        with self.assertRaisesRegex(ValueError, 'unexpected immutable'):
            worker.rewrite_child(argv, legacy_fit=R19/'fit.py', new_fit=self.runtime.fit_path,
                                 plan_path=self.plan_path, smoke=False)

    def test_write_guard_and_exclusive_receipts(self):
        original = self.root/'original.json'; original.write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'escaped'):
            self.runtime.base.atomic(original, {})
        target = self.out/'pairs/one/response.json'
        self.runtime.base.atomic(target, {'value': 1})
        with self.assertRaises(FileExistsError):
            self.runtime.base.atomic(target, {'value': 2})
        self.assertEqual(json.loads(target.read_text()), {'value': 1})
        self.assertEqual(original.read_text(), 'preserve')
        link = self.out/'escape'; link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escaped'):
            self.runtime.base.atomic(link/'original.json', {})

    def test_100_trials_15_seeds_native_aggregation_new_protocol(self):
        study = self.study(); requests = []
        def execute(request, folder, claim=None):
            requests.append(deepcopy(request))
            return {'score': .75, 'complete': True, 'metrics': {'ACC': .5, 'AUC': .6, 'F1': .4}}
        self.runtime.short.execute = execute
        pair, row = self.pair()
        self.runtime.short.run_pair(pair, row, self.runtime.plan, 'gpu', None)
        self.assertEqual(study.asks, 100)
        self.assertEqual(sum(r['mode'] == 'trial' for r in requests), 100)
        self.assertEqual([r['seed'] for r in requests if r['mode'] == 'final'], list(range(15)))
        self.assertTrue(all(r['batch_size'] == 1024 and r['max_epoch'] == 200 and r['cpus'] == 8 for r in requests))
        output = self.out/'results'/pair['method']/(pair['key']+'.json')
        result = json.loads(output.read_text())
        self.assertEqual(result['protocol'], worker.PROTOCOL)
        self.assertEqual(result['runtime_sources'], self.runtime.sources)
        self.runtime.base.validate_complete(result, pair, self.runtime.plan)
        result['protocol'] = worker.LEGACY_PROTOCOL
        with self.assertRaisesRegex(ValueError, 'wrong protocol'):
            self.runtime.base.validate_complete(result, pair, self.runtime.plan)

    def test_interrupted_trial_resumes_same_number_without_spending_attempt(self):
        study = self.study(); pair, row = self.pair()
        self.runtime.short.execute = lambda *args: (_ for _ in ()).throw(self.runtime.short.BudgetStop('pause'))
        with self.assertRaises(self.runtime.short.BudgetStop):
            self.runtime.short.run_pair(pair, row, self.runtime.plan, 'gpu', None)
        self.assertEqual(study.asks, 1)
        self.assertEqual(study.trials[0].state, 'RUNNING')
        saved = deepcopy(study.trials[0].user_attrs['config']); executed = []
        def finish(request, folder, claim=None):
            executed.append(deepcopy(request))
            return {'score': .75, 'complete': True, 'metrics': {'ACC': .5, 'AUC': .6, 'F1': .4}}
        self.runtime.short.execute = finish
        self.runtime.short.run_pair(pair, row, self.runtime.plan, 'gpu', None)
        self.assertEqual(executed[0]['config'], saved)
        self.assertEqual(study.asks, 100)
        self.assertEqual(len(study.trials), 100)
        self.assertFalse((self.out/'errors'/(pair['key']+'.json')).exists())

    def test_cpu_tree_is_admitted_with_eight_cpu_formal_requests(self):
        self.study(); requests = []
        pair = next(p for p in self.runtime.plan['pairs'] if p['method'] == 'CatBoost')
        row = next(r for r in self.runtime.plan['rows'] if r['dataset'] == pair['dataset'])
        def execute(request, folder, claim=None):
            requests.append(deepcopy(request))
            return {'score': .7, 'complete': True, 'metrics': {'ACC': .5, 'AUC': .6, 'F1': .4}}
        self.runtime.short.execute = execute
        self.runtime.short.run_pair(pair, row, self.runtime.plan, 'gpu', None)
        self.assertEqual(len(requests), 115)
        self.assertTrue(all(r['method'] == 'CatBoost' and r['cpus'] == 8 for r in requests))
        result = json.loads((self.out/'results/CatBoost'/(pair['key']+'.json')).read_text())
        self.assertEqual(result['execution_mode'], 'cpu')
        self.assertIn('trees execute CPU-only', result['lane'])

    def test_expired_budget_does_not_allocate_a_trial(self):
        study = self.study(); pair, row = self.pair()
        self.runtime.short.BUDGET.request_stop('expired')
        with self.assertRaises(self.runtime.short.BudgetStop):
            self.runtime.short.run_pair(pair, row, self.runtime.plan, 'gpu', None)
        self.assertEqual(study.asks, 0)

    def test_all_failed_trials_produce_no_selection_or_default_fallback(self):
        study = self.study(); pair, row = self.pair()
        self.runtime.short.execute = lambda *args: (_ for _ in ()).throw(RuntimeError('actual fit failed'))
        self.runtime.fit.default_config = lambda *args: self.fail('defaults must never be used')
        with self.assertRaisesRegex(AssertionError, 'all 100 search attempts failed'):
            self.runtime.short.run_pair(pair, row, self.runtime.plan, 'gpu', None)
        self.assertEqual(study.asks, 100)
        self.assertTrue(all(t.state == 'FAIL' for t in study.trials))
        self.assertFalse((self.out/'pairs'/pair['key']/'selected.json').exists())

    def test_cpu_only_allocation_formal_rejected(self):
        with self.assertRaisesRegex(ValueError, 'GPU-reserved lane'):
            self.runtime.worker('cpu')

    def test_cpu_smoke_mask_uses_explicit_rocm_disabled_sentinels(self):
        worker.mask_cpu_visibility()
        self.assertEqual(os.environ['CPU_ONLY'], '1')
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
        for key in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
            self.assertEqual(os.environ[key], '-1')

    def test_formal_requires_all_nine_actual_smokes(self):
        with self.assertRaises(FileNotFoundError):
            self.runtime.worker('gpu')
        for method in self.runtime.fit.METHODS:
            folder = self.out/'startup_smoke'/method; folder.mkdir(parents=True)
            (folder/'pass.json').write_text(json.dumps({'passed': True, 'method': method,
                'plan_id': self.runtime.plan['plan_id'], 'runtime_sources': self.runtime.sources}))
        with self.assertRaises(FileNotFoundError):
            self.runtime.check_smoke()  # Bare passed=True cannot substitute for an actual response.

    def test_response_adapter_and_smoke_identity_required(self):
        with self.assertRaisesRegex(ValueError, 'wrong plan/adapter'):
            self.runtime.base.validate_response({'complete': True}, {})
        with self.assertRaisesRegex(ValueError, 'smoke/formal'):
            self.runtime.base.validate_response({'missing190_plan_id': self.runtime.plan['plan_id'],
                                                  'diagnostic_smoke': True}, {})

    def smoke_proofs(self):
        for method in self.runtime.fit.METHODS:
            pair = next(p for p in self.runtime.plan['pairs'] if p['method'] == method)
            row = next(r for r in self.runtime.plan['rows'] if r['dataset'] == pair['dataset'])
            config = {'model': {}, 'fit' if method in self.runtime.fit.CPU_METHODS else 'training': {'n_bins': 2}}
            request = {'mode': 'trial', 'method': method, 'row': row, 'config': config, 'seed': 0,
                       'max_epoch': 2, 'batch_size': 64, 'cpus': 16 if method in self.runtime.fit.CPU_METHODS else 8}
            folder = self.out/'startup_smoke'/method
            request_path = folder/'trial/request.json'
            self.runtime.base.atomic(request_path, request)
            response = {'complete': True, 'mode': 'trial', 'method': method, 'seed': 0,
                        'dataset': row['dataset'], 'task_kind': 'classification', 'row_id': row['row_id'],
                        'data_audit': {'input_fingerprint': row['input_fingerprint']}, 'config': config,
                        'request_config_digest': plan_builder.object_digest(config), 'test_evaluated': False,
                        'objective_name': 'Accuracy', 'score': .75, 'fit_seconds': 2.0,
                        'device': 'cpu' if method in self.runtime.fit.CPU_METHODS else 'cuda:0',
                        'missing190_adapter': {'method': method}, 'diagnostic_smoke': True,
                        'missing190_plan_id': self.runtime.plan['plan_id'], 'request_sha256': worker.digest(request_path)}
            response_path = folder/'trial/response.json'
            self.runtime.base.atomic(response_path, response)
            self.runtime.base.atomic(folder/'pass.json', {'passed': True, 'method': method,
                'plan_id': self.runtime.plan['plan_id'], 'runtime_sources': self.runtime.sources,
                'request_sha256': worker.digest(request_path), 'response_sha256': worker.digest(response_path)})

    def test_all_nine_smoke_proofs_and_hash_tamper(self):
        self.smoke_proofs()
        self.assertEqual(len(self.runtime.check_smoke()), 9)
        target = self.out/'startup_smoke/TabM/trial/response.json'
        record = json.loads(target.read_text()); record['score'] = .8
        target.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'proof changed'):
            self.runtime.check_smoke()

    def test_gate_records_all_nine_true_execution_routes_and_rejects_overlap(self):
        self.smoke_proofs()
        def preflight(rank, pci, cpus):
            self.runtime.base.atomic(self.out/'preflight/1234'/f'gpu-{rank}.json', {
                'passed': True, 'plan_id': self.runtime.plan['plan_id'], 'runtime_sources': self.runtime.sources,
                'job_id': '1234', 'rank': rank, 'mode': 'gpu', 'devices': 1, 'physical_gpu': pci,
                'cpu_affinity': cpus, 'host': 'node1'})
        preflight(0, '0000:01:00.0', list(range(8)))
        gate = self.runtime.gate(1)
        self.assertEqual(len(gate['methods']), 9)
        self.assertEqual(gate['actual_execution_modes']['CatBoost'], 'cpu')
        self.assertEqual(gate['actual_execution_modes']['TabM'], 'gpu')
        preflight(1, '0000:01:00.0', list(range(8,16)))
        with self.assertRaisesRegex(ValueError, 'GPU overlap'):
            self.runtime.gate(2)
        preflight(1, '0000:02:00.0', list(range(4,12)))
        with self.assertRaisesRegex(ValueError, 'CPU affinity overlaps'):
            self.runtime.gate(2)

    def test_wrong_or_missing_monotonic_clock_rejected(self):
        with patch.dict(os.environ, {'JOB_BUDGET_END_EPOCH': '10000'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'monotonic deadline'):
                worker.monotonic_budget(self.runtime.short)

    def test_source_has_no_default_calls_or_old_startup_entry(self):
        import ast
        tree = ast.parse(Path(worker.__file__).read_text())
        attrs = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn('default_config', attrs)
        self.assertNotIn('main', attrs)


if __name__ == '__main__':
    unittest.main()
