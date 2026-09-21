"""Isolated fit adapter for the NEW standard-split missing-190 CLS campaign.

No historical score/default replaces HPO: the worker must perform 100 trials
and 15 final seeds.  This module runs exactly one trial/final fit per process.
It reuses pinned R19 fit/data bytes, but injects a nine-method registry and a
NEW cache/output root before importing data.  It never edits old campaign code.
Runtime GPU/CPU smoke is still required; CPU mocks are not model validation.
"""
from __future__ import annotations

import argparse
from collections.abc import MutableMapping
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
R19 = ROOT / 'stage/table6_remaining19_standard_hpo_bg8_20260912_v1'
METHODS = {'CatBoost': 'catboost', 'LightGBM': 'lightgbm', 'XGBoost': 'xgboost',
           'TabM': 'tabm', 'TabR': 'tabr', 'TabTransformer': 'tabtransformer',
           'TabNet': 'tabnet', 'SwitchTab': 'switchtab', 'TabCaps': 'tabcaps'}
CPU_METHODS = {'CatBoost', 'LightGBM', 'XGBoost'}
CPU_MASK = {'CPU_ONLY': '1', 'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': '-1',
            'ROCR_VISIBLE_DEVICES': '-1', 'GPU_DEVICE_ORDINAL': '-1'}
NATIVE_ACCURACY = {'TabNet', 'TabCaps'}
PINNED_R19 = {
    'fit.py': '101ef103b72e1af0a624a76c2311557458dc8c5cec94276f9192b8481333adf6',
    'common.py': '8d03dc74f623ec5e00694da5fc6f9e54897c56036ee473177fdfa3d0fabecad0',
    'data.py': '8681149b3c7b683d659dd79038474a66af382cb025c69b12728c180bcac36c80',
}
NAME = 'table6_missing190_standard_hpo_20260922_v1'
R19_PLAN_ID = '47c3d448249235ad4f7ae6248424a1f8f38094ae204d2d4ece472b4effb61aff'
_FIT = None
_PLAN = None
_PLAN_PATH = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_plan(plan):
    body = dict(plan)
    plan_id = body.pop('plan_id', None)
    require(plan_id == object_digest(body), 'missing190 plan digest mismatch')
    require(plan.get('name') == NAME and plan.get('applicable_pair_target') == 190,
            'not the missing190 campaign')
    require(plan.get('methods') == METHODS and set(plan.get('cpu_methods', [])) == CPU_METHODS,
            'missing190 method/CPU registry mismatch')
    require(plan.get('hpo_trials') == 100 and plan.get('hpo_seed') == 0 and
            plan.get('seeds') == list(range(15)) and plan.get('max_epoch') == 200 and
            plan.get('batch_size') == 1024, 'missing190 HPO/seed/training contract mismatch')
    require(plan.get('source_contract', {}).get('remaining19_plan_id') == R19_PLAN_ID,
            'wrong parent data contract')
    rows = plan.get('rows', [])
    by_name = {row['dataset']: row for row in rows}
    require(len(by_name) == len(rows) and all(row['task_kind'] == 'classification' for row in rows),
            'duplicate or non-classification rows')
    pairs = plan.get('pairs', [])
    require(len(pairs) == len({(p['method'], p['dataset']) for p in pairs}) == 190,
            'missing190 pair membership mismatch')
    require({p['method'] for p in pairs} == set(METHODS), 'missing190 pair roster incomplete')
    for pair in pairs:
        require(pair['method'] in METHODS and pair['dataset'] in by_name and
                pair['task_kind'] == 'classification' and pair['hpo_trials'] == 100 and
                pair['seeds'] == list(range(15)), 'invalid missing190 pair')
        require(pair['input_fingerprint'] == by_name[pair['dataset']]['input_fingerprint'],
                'pair/data fingerprint mismatch')
    output = Path(plan['output_root'])
    require(output.is_absolute() and output.name == NAME and not output.is_symlink(),
            'requires the independent missing190 output namespace')
    resolved = output.resolve()
    for old_name in ('table6_remaining19_standard_hpo_bg8_20260912_v1',
                     'table6_standard457_fixedbest_gt1_20260909_v1',
                     'table6_standard457_fixedbest_bg1_gpu_recovery_20260909_v2'):
        old = (ROOT / 'evaluation' / old_name).resolve()
        require(not resolved.is_relative_to(old) and not old.is_relative_to(resolved),
                'missing190 output overlaps an old result namespace')
    return plan


def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def initialize(plan_path=None, stage=None):
    """Initialize once in a fresh process; refuse accidental module pollution."""
    global _FIT, _PLAN, _PLAN_PATH
    path = Path(plan_path or os.environ.get('T6_MISSING190_PLAN') or _PLAN_PATH or '')
    require(path.is_absolute() and path.is_file(), 'set --plan or T6_MISSING190_PLAN to a frozen plan')
    path = path.resolve()
    if _FIT is not None:
        require(path == _PLAN_PATH, 'cannot switch missing190 plan in one process')
        require(json.loads(path.read_text()) == _PLAN, 'missing190 plan changed in process')
        return _FIT
    plan = validate_plan(json.loads(path.read_text()))
    stage = Path(stage or os.environ.get('T6_BASE_STAGE', str(R19))).resolve()
    for filename, expected in PINNED_R19.items():
        require(hashlib.sha256((stage / filename).read_bytes()).hexdigest() == expected,
                'pinned R19 source changed: ' + filename)
    require('data' not in sys.modules, 'fresh fit process required: data module already imported')
    if 'common' in sys.modules:
        require(Path(sys.modules['common'].__file__).resolve() == stage / 'common.py',
                'unrelated common module already imported')
    else:
        _import_file('common', stage / 'common.py')
    common = sys.modules['common']
    # Assign, do not mutate the old dictionaries referenced by another module.
    common.METHODS = dict(METHODS)
    common.CPU_METHODS = set(CPU_METHODS)
    common.OUT = Path(plan['output_root']).resolve()
    sys.path.insert(0, str(stage))
    fit = _import_file('_table6_missing190_r19_fit', stage / 'fit.py')
    original = fit._run_talent

    def adapted_run(request, bundle):
        return run_talent_adapter(original, request, bundle)

    fit._run_talent = adapted_run
    _FIT, _PLAN, _PLAN_PATH = fit, plan, path
    return fit


def postprocess_sample(method, config):
    """Only fixed TALENT HPO additions, never unsampled default parameters."""
    require(method in METHODS, 'unknown missing190 method')
    config = copy.deepcopy(config)
    model = config.setdefault('model', {})
    if method in {'TabR', 'TabM'}:
        require(isinstance(model.get('num_embeddings'), dict), 'sampled PLR embeddings missing')
        model['num_embeddings'].setdefault('type', 'PLREmbeddings')
        model['num_embeddings'].setdefault('lite', True)
    if method == 'TabM':
        require(isinstance(model.get('backbone'), dict), 'sampled TabM backbone missing')
        model['backbone'].setdefault('type', 'MLP')
        model.setdefault('arch_type', 'tabm')
        model.setdefault('k', 32)
    if method == 'TabR':
        for key, value in {'d_multiplier': 2.0, 'mixer_normalization': 'auto', 'dropout1': 0.0,
                           'normalization': 'LayerNorm', 'activation': 'ReLU'}.items():
            model.setdefault(key, value)
    return config


def default_config(method, talent_root=None):
    require(method in METHODS, 'unknown missing190 method')
    fit = initialize()
    # The immutable R19 default helper only knows RF uses the classical `fit`
    # section. Load the same packaged JSON explicitly for these three trees.
    path = fit._config_dir(talent_root) / 'default' / (METHODS[method] + '.json')
    config = json.loads(path.read_text())[METHODS[method]]
    section = 'fit' if method in CPU_METHODS else 'training'
    config.setdefault(section, {}).setdefault('n_bins', 2)
    return postprocess_sample(method, config)


def suggest_config(trial, method, task_kind, talent_root=None):
    require(task_kind == 'classification' and method in METHODS,
            'only the missing190 classification methods may enter HPO')
    fit = initialize()
    # Uses the unchanged packaged task-specific space and Optuna distributions.
    return postprocess_sample(method, fit.suggest_config(trial, method, task_kind, talent_root))


def runtime_request(request):
    """Adapt computational arguments; leave the requested/selected config intact."""
    result = copy.deepcopy(request)
    method = result['method']
    require(method in METHODS and result['row']['task_kind'] == 'classification',
            'all nine missing190 methods are classification-only in this campaign')
    config = result['config']
    section = 'fit' if method in CPU_METHODS else 'training'
    bins = config.get(section, {}).get('n_bins')
    require(type(bins) is int and 2 <= bins <= 256, 'selected n_bins missing or invalid')
    cpus = result.get('cpus', 4)
    require(type(cpus) is int and cpus >= 1, 'invalid per-fit CPU allocation')
    if method in CPU_METHODS:
        require('training' not in config, 'tree configs must use fit, not a conflicting training section')
        # R19 reads training.n_bins for every non-RF method. A runtime-only alias
        # lets its unchanged checks pass; TALENT still consumes fit.n_bins.
        config['training'] = {'n_bins': bins}
        if method == 'CatBoost':
            require('task_type' not in config['model'],
                    'CatBoost TALENT constructor supplies task_type itself')
            config['model']['thread_count'] = cpus
        else:
            config['model']['n_jobs'] = cpus
        if method == 'XGBoost':
            require(config['model'].get('tree_method') != 'gpu_hist' and
                    not str(config['model'].get('device', '')).startswith('cuda'),
                    'CUDA XGBoost configuration forbidden in the CPU tree lane')
    return result


def install_tabr_cpu_faiss(cpus, *, faiss_module=None, torch_module=None, numpy_module=None):
    """Exact float32 L2 retrieval; transfer indices/distances back to query device."""
    if faiss_module is None:
        import faiss as faiss_module
    if torch_module is None:
        import torch as torch_module
    if numpy_module is None:
        import numpy as numpy_module
    faiss, torch, np = faiss_module, torch_module, numpy_module
    if hasattr(faiss, 'omp_set_num_threads'):
        faiss.omp_set_num_threads(cpus)
    if hasattr(faiss, 'GpuIndexFlatL2') and not getattr(torch.version, 'hip', None):
        return 'native_faiss_gpu'
    require(hasattr(faiss, 'IndexFlatL2'), 'FAISS CPU IndexFlatL2 unavailable')

    class CpuIndexFlatL2Adapter:
        def __init__(self, resources, dimension):
            self.index = faiss.IndexFlatL2(dimension)

        def reset(self):
            self.index.reset()

        def add(self, values):
            array = np.ascontiguousarray(values.detach().float().cpu().numpy(), dtype=np.float32)
            self.index.add(array)

        def search(self, values, count):
            array = np.ascontiguousarray(values.detach().float().cpu().numpy(), dtype=np.float32)
            distances, indices = self.index.search(array, count)
            return (torch.as_tensor(distances, device=values.device),
                    torch.as_tensor(indices, device=values.device))

    faiss.StandardGpuResources = lambda: None
    faiss.GpuIndexFlatL2 = CpuIndexFlatL2Adapter
    return 'cpu_exact_l2_float32; indices/distances returned to query device'


def run_talent_adapter(original, request, bundle):
    """Narrow fresh-process API shim around the pinned R19 implementation."""
    import TALENT.api as api
    method = request['method']
    adapted = runtime_request(request)
    retrieval = None
    if method == 'TabR':
        retrieval = install_tabr_cpu_faiss(adapted.get('cpus', 4))
    original_build = api.build_args

    def build_args(name, **kwargs):
        require(name == METHODS[method], 'TALENT method changed during isolated fit')
        if method in NATIVE_ACCURACY:
            # These specialized fit loops already use eval_metric=['accuracy'].
            # TALENT explicitly rejects its generic tune_metric argument here.
            kwargs['tune_metric'] = None
        args = original_build(name, **kwargs)
        section = 'fit' if method in CPU_METHODS else 'training'
        expected = request['config'][section]['n_bins']
        require(args.config[section]['n_bins'] == expected, 'TALENT changed selected n_bins')
        # Preserve the original R19 alias check even if build_args strips extra
        # classical sections. This alias is not passed to the tree estimator.
        if method in CPU_METHODS:
            args.config['training'] = {'n_bins': expected}
        return args

    api.build_args = build_args
    try:
        result = original(adapted, bundle)
    finally:
        api.build_args = original_build
    result['config'] = copy.deepcopy(request['config'])
    if method in NATIVE_ACCURACY:
        # These specialized estimators manage their own dtype; the generic
        # args.use_float flag does not establish their actual tensor precision.
        result['precision'] = 'unchanged native specialized estimator precision'
    result['missing190_adapter'] = {
        'method': method, 'talent_name': METHODS[method],
        'native_validation_accuracy': method in NATIVE_ACCURACY,
        'selected_n_bins_section': 'fit' if method in CPU_METHODS else 'training',
        'tree_training_alias_runtime_only': method in CPU_METHODS,
        'retrieval_backend': retrieval, 'threshold_tuning': False,
        'true_model_smoke_claimed_by_adapter': False,
    }
    return result


def validate_request(request, plan, diagnostic_smoke=False):
    require(request.get('method') in METHODS and request.get('mode') in {'trial', 'final'},
            'invalid missing190 method or fit mode')
    row = request.get('row', {})
    require(row.get('task_kind') == 'classification', 'missing190/TabCaps scope is classification only')
    require(any(pair['method'] == request['method'] and pair['dataset'] == row.get('dataset')
                for pair in plan['pairs']), 'request is outside exact missing190 pair membership')
    frozen = next((r for r in plan['rows'] if r['dataset'] == row.get('dataset')), None)
    require(row == frozen, 'request row differs from pinned standard split')
    seed = request.get('seed')
    require(type(seed) is int and 0 <= seed < 15 and (request['mode'] != 'trial' or seed == 0),
            'trial seed must be zero; final seeds must be 0..14')
    epochs = request.get('max_epoch', 200)
    if diagnostic_smoke:
        require(request['mode'] == 'trial' and epochs in (1, 2), 'smoke requires a 1/2-epoch validation-only trial')
    else:
        require(epochs == 200 and request.get('batch_size', 1024) == 1024,
                'formal fit requires the frozen 200 epochs / batch size 1024')
    require(not request.get('tune_threshold', False), 'threshold tuning is outside standard protocol')
    runtime_request(request)


class _CpuEnvironment(MutableMapping):
    """Keep the CPU mask when the pinned fit writes its older empty masks."""
    def __getitem__(self, key):
        return os.environ[key]

    def __setitem__(self, key, value):
        os.environ[key] = CPU_MASK.get(key, value)

    def __delitem__(self, key):
        require(key not in CPU_MASK, 'cannot remove the isolated CPU visibility mask')
        del os.environ[key]

    def __iter__(self):
        return iter(os.environ)

    def __len__(self):
        return len(os.environ)


class _CpuOS:
    # Local to the imported legacy fit module; never replace the shared os
    # module's environ. Its data import observes the real, protected masks.
    environ = _CpuEnvironment()

    def __getattr__(self, name):
        return getattr(os, name)


def execute(request, *, diagnostic_smoke=False):
    fit = initialize()
    validate_request(request, _PLAN, diagnostic_smoke)
    if request['method'] in CPU_METHODS:
        # Before any TALENT/numpy/torch import in the isolated fit process.
        # Empty HIP/ROCR visibility has differed between ROCm runtimes. Use the
        # known-working CPU-only sentinel, while CUDA follows its empty mask.
        os.environ.update(CPU_MASK)
        original_os = fit.os
        fit.os = _CpuOS()
        try:
            result = fit.execute(request)
        finally:
            fit.os = original_os
    else:
        result = fit.execute(request)
    result.update(missing190_plan_id=_PLAN['plan_id'], diagnostic_smoke=bool(diagnostic_smoke),
                  selection_provenance='fresh validation HPO; not a historical fixed-best result')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    if args.plan is not None:
        os.environ['T6_MISSING190_PLAN'] = str(args.plan.resolve())
    fit = initialize()
    root = Path(_PLAN['output_root']).resolve()
    for path in (args.request, args.response):
        require(path.is_absolute() and path.resolve().is_relative_to(root), 'fit receipts must stay inside new missing190 output')
        require(not path.is_symlink(), 'refuse symlink fit receipt')
    smoke_root = root / 'startup_smoke'
    require(args.response.resolve().is_relative_to(smoke_root) == args.smoke,
            'smoke and formal receipt namespaces must be separate')
    require(not args.response.exists(), 'refuse to overwrite existing fit response')
    raw = args.request.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        request = json.loads(raw)
        result = execute(request, diagnostic_smoke=args.smoke)
    except Exception as exc:
        fit.publish_response(args.response, {'schema_version': 1, 'complete': False,
                             'request_sha256': digest, 'diagnostic_smoke': args.smoke,
                             'error_type': type(exc).__name__, 'error': str(exc),
                             'traceback': traceback.format_exc()})
        raise
    fit.publish_response(args.response, {**result, 'request_sha256': digest})
    print(json.dumps({'complete': True, 'diagnostic_smoke': args.smoke, 'response': str(args.response)}))


if __name__ == '__main__':
    main()
