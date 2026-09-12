"""Assemble native, byte-identical G5SC model/optimizer with the T25 task."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
DEFAULT_T25 = WORK / 'outputs/tabicl_regression_supportonly_20260820_v2/source'
DEFAULT_OVERLAY = WORK / 'outputs/tabicl_regression_adapted_e4_20260822_v1/source_overlay'
EXPECTED = {
    'train/_run.py': 'ffecf29231cbab486272aab91b68db1412207a8f6233c615001a122ca35b1340',
    'train/_train_config.py': 'd9b50b9a5b1ce39b81f97d2f0b96d75af7983dcd968882997662988c7f2a2fcf',
    'train/_optim.py': '6f76dc339f2ad8d02112ff828e88c04f0b8126df3ee8329b8840a45c13ec581a',
}
EXPECTED_G5_MODEL_HASHES = {
    '_model/__init__.py': 'c24eccee285dd1c5dea721708cd3e52501fe39f29a865fb6cd3d8b448a33ad4f',
    '_model/attention.py': 'dd7494e40ab7d9f7acc24e93d9c6d9cc5f39aad027b74e67fd0d49fee460dd03',
    '_model/attention_gate.py': 'd607e01925373a65a60a0ed19ea8450187981bca81a51644381ea75ef281ad54',
    '_model/embedding.py': '9e3c618bdccf2e374f9a2dfc079d9673e8ef27fc665bc56c4bca7b7b03b2b968',
    '_model/encoders.py': 'bdfea8d4d96513e9211030fa9674fbfd40176f3f2ccdcc524c526c31c850b6c1',
    '_model/function_tokens.py': '660d6366ed3ed51cba8c7de6efb0a99922579110b6c99118fe27bfb35913cec3',
    '_model/inference.py': '9dadc3318090559015a465d65449a04771a2a5a0c947b99617c45871af61470a',
    '_model/inference_config.py': 'f611b37a67ee195ff0639317f044bb0576d724b1342ba5fda0372b1776052446',
    '_model/interaction.py': 'a49d8af76045f7040715e0f4f369eeac87115b68a021ea8342c3e0c1d07a1015',
    '_model/kv_cache.py': '09203f48c7181dcad3873e6c25cf6bc9e6251f85301354c6f7e1a129a0a51b9e',
    '_model/layers.py': 'ee6f8846a37f56245b3d005fc05e32417fef24fbcf77e49e171d4fb680297ad6',
    '_model/learning.py': '159840cda23c90708c4c0d34c6a2fd9a567390a6f7033fdb632b627f3ba00e28',
    '_model/quantile_dist.py': '7e09ac0c3ce0260dc07cef79b0a98ee63b0785efafbf415fd243fe2981499b99',
    '_model/rope.py': 'f59874106743b58330f35699df27ad8d45d4ae692b38c64626f58cdf2da12eab',
    '_model/schema_expert.py': '3c945a3201cfe2a38020cdfd83cbd2033083a5330a3ffa9332d94611e16c047a',
    '_model/ssmax.py': '1739bf7e8e4a791fbaf49b18b44dfa4fbc6366cc20a3ea69ee4d22b3260056b6',
    '_model/tabicl.py': '25582e2d387b2d55621104ab0d393597b63c479b11a38d5da2c6b1f149cd4f91',
}
OVERLAY_FILES = ('_dataset.py', '_graph_scm.py', '_regression_target_prior.py',
                 '_support_only_preprocessing.py')
EXPECTED_T25_TASK_HASHES = {
    '_dataset.py': 'ad0acffc5753d5d242b9ddedd7d15247e31b7c8139bbadea1b873a2061b54dd4',
    '_graph_scm.py': 'f9a9a6d3283b8bdacf573d171f6a0b9fbc452e7765ffe8b75b38ed0943a11744',
    '_regression_target_prior.py': '723e4e50edd36f3122e4502a7fd650e895042685af39a995fb26cfb32a0276fe',
    '_support_only_preprocessing.py': '5ebe75ac156b7ae124e46569b65f8455aac0e81a08bc8f9c7f8c177515478f7b',
    'graph_lib/_config.py': '7e2b6a9422a75bf8454e80dc9a3f0d96d37a9f0f75acd209dddbef2234479024',
}
EXPECTED_SAFE_TAIL_SHA256 = 'b955520f3f363d68d6df5cb1ffaabf16b1bf3b1c1f286eb216d57d4afb49e1c6'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise AssertionError((old, text.count(old), count))
    return text.replace(old, new)


def method(text, cls, name, transform):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines(keepends=True)
    return ''.join(lines[:fn.lineno - 1]) + transform(''.join(lines[fn.lineno - 1:fn.end_lineno])) + ''.join(lines[fn.end_lineno:])


def replace_prior_constructor(text):
    text = textwrap.dedent(text)
    node = next(n for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'PriorDataset')
    lines = text.splitlines(keepends=True)
    before = ''.join(lines[:node.lineno - 1]) + lines[node.lineno - 1][:node.col_offset]
    after = lines[node.end_lineno - 1][node.end_col_offset:] + ''.join(lines[node.end_lineno:])
    return textwrap.indent(before + 'build_t25_prior(self.config)' + after, '    ')


def prepare(source, destination, *, t25_source=None, t25_prior_overlay=None):
    """source=full G5SC; T25 inputs=source + adapted overlay; destination must be new.

    The remote adapted T25 source already contains the four overlay files and
    can be supplied as both t25_source and t25_prior_overlay.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    t25_source = Path(t25_source or DEFAULT_T25).resolve()
    t25_prior_overlay = Path(t25_prior_overlay or DEFAULT_OVERLAY).resolve()
    base, prior_base = source / 'src/tabicl', t25_source / 'src/tabicl/prior'
    overlay = t25_prior_overlay / 'src/tabicl/prior'
    for name, expected in EXPECTED.items():
        if digest(base / name) != expected:
            raise AssertionError(f'Unrecognized G5SC baseline: {name}')
    # AppleDouble transfer sidecars are metadata, not Python model modules;
    # the immutable source copy below already excludes them.
    model_hashes = {str(p.relative_to(base)): digest(p) for p in sorted((base / '_model').rglob('*.py'))
                    if not p.name.startswith('._') and '__MACOSX' not in p.parts}
    if not EXPECTED_G5_MODEL_HASHES or model_hashes != EXPECTED_G5_MODEL_HASHES:
        raise AssertionError('Source is not the canonical audited G5SC model snapshot')
    if not prior_base.is_dir():
        raise FileNotFoundError(prior_base)
    for name in OVERLAY_FILES:
        if not (overlay / name).is_file():
            raise FileNotFoundError(overlay / name)
    for name, expected in EXPECTED_T25_TASK_HASHES.items():
        task_path = (overlay if name in OVERLAY_FILES else prior_base) / name
        if digest(task_path) != expected:
            raise AssertionError(f'Unrecognized frozen T25 task source: {task_path}')
    if digest(HERE / 'regression_safe_tail_length_runtime_patch.py') != EXPECTED_SAFE_TAIL_SHA256:
        raise AssertionError('The frozen T25 Safe-Tail implementation changed')
    mapper = (overlay / '_regression_target_prior.py').read_text()
    if 'map_to_template_from_support(y, train_size, levels, template)' not in mapper or 'del levels' in mapper:
        raise AssertionError('T25 overlay must preserve nonuniform target quantile levels')
    if 'from ._support_only_preprocessing import NonFinitePriorError' not in mapper:
        raise AssertionError('T25 overlay must include finite-prior resampling support')
    if destination.exists():
        raise FileExistsError(f'refuse overwrite: {destination}')
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '._*', '__MACOSX')
    shutil.copytree(source, destination, ignore=ignore)
    out = destination / 'src/tabicl'
    shutil.copytree(prior_base, out / 'prior', dirs_exist_ok=True, ignore=ignore)
    for name in OVERLAY_FILES:
        shutil.copyfile(overlay / name, out / 'prior' / name)
    # Package installation covers spawned workers and nested generator pools.
    shutil.copyfile(HERE / 'regression_safe_tail_length_runtime_patch.py', out / 'prior/_t25_safe_tail.py')
    init = (out / 'prior/__init__.py').read_text()
    init += '\n# Frozen T25 task, installed identically in every generator process.\nfrom ._t25_safe_tail import install as _install_t25_safe_tail\n_install_t25_safe_tail()\n'
    (out / 'prior/__init__.py').write_text(init)
    shutil.copyfile(HERE / 'trainer_adapter.py', out / 'train/_t25_regression_adapter.py')
    shutil.copyfile(HERE / 'runtime_probe.py', out / 'train/_g5sc_runtime_probe.py')

    text = (out / 'train/_train_config.py').read_text()
    text = replace(text, '    return parser',
        '    parser.add_argument("--regression_method", choices=("quantile",), default="quantile")\n'
        '    parser.add_argument("--num_quantiles", type=int, default=999)\n'
        '    return parser')
    (out / 'train/_train_config.py').write_text(text)
    text = (out / 'train/_run.py').read_text()
    text = replace(text, 'from tabicl.train._train_config import build_parser',
        'from tabicl.train._train_config import build_parser\n'
        'from tabicl.train._t25_regression_adapter import (\n'
        '    build_t25_prior, run_regression_micro_batch, validate_regression_task,\n'
        ')')
    text = replace(text, '        self.config = config',
        '        self.config = config\n        validate_regression_task(config)')
    text = method(text, 'Trainer', 'build_model', lambda s: replace(replace(s,
        '            "max_classes": self.config.max_classes,',
        '            "max_classes": 0,\n            "num_quantiles": self.config.num_quantiles,'),
        '        model.to(device=self.config.device)',
        '        model.to(device=self.config.device)\n'
        '        from tabicl.train._g5sc_runtime_probe import probe\n'
        '        probe(model, self.config.checkpoint_dir)'))
    text = method(text, 'Trainer', 'configure_prior', replace_prior_constructor)
    text = method(text, 'Trainer', 'run_micro_batch', lambda s:
        '    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches, timings=None):\n'
        '        """T25 continuous regression supervision; native G5SC outer update loop."""\n'
        '        return run_regression_micro_batch(\n'
        '            self, micro_batch, micro_batch_idx, num_micro_batches, timings\n'
        '        )')
    text = method(text, 'Trainer', 'run_batch', lambda s: replace(s,
        '            results.update({"ce": 0.0, "accuracy": 0.0})',
        '            results.update({"pinball": 0.0, "mse": 0.0})'))
    text = method(text, 'Trainer', 'paired_training_contract', lambda s: replace(s,
        '        prior_config_payload = json.dumps(prior_config, sort_keys=True, default=str).encode("utf-8")',
        '        prior_config["t25_graph_config"] = vars(PriorConfig.from_args(self.config))\n'
        '        prior_config["regression_method"] = self.config.regression_method\n'
        '        prior_config["num_quantiles"] = self.config.num_quantiles\n'
        '        prior_config["safe_tail"] = True\n'
        '        prior_config["effective_generator_n_jobs"] = 1\n'
        '        with open(self.config.regression_target_profile, "rb") as profile_handle:\n'
        '            prior_config["target_profile_sha256"] = hashlib.sha256(profile_handle.read()).hexdigest()\n'
        '        prior_config_payload = json.dumps(prior_config, sort_keys=True, default=str).encode("utf-8")'))
    (out / 'train/_run.py').write_text(text)
    for path in out.rglob('*.py'):
        compile(path.read_bytes(), str(path), 'exec')
    actual = {str(p.relative_to(out)): digest(p) for p in sorted((out / '_model').rglob('*.py'))}
    if actual != model_hashes or digest(out / 'train/_optim.py') != EXPECTED['train/_optim.py']:
        raise AssertionError('Original G5SC model or Muon/scheduler changed during assembly')
    report = {
        'schema_version': 2, 'architecture': 'native_G5SC_backbone_T25_regression_task',
        'source': str(source), 'derived_source': str(destination),
        't25_source': str(t25_source), 't25_prior_overlay': str(t25_prior_overlay),
        'g5sc_model_hashes': model_hashes, 'g5sc_model_bytes_identical': True,
        'g5sc_optimizer_scheduler_sha256': EXPECTED['train/_optim.py'], 'baseline_hashes': EXPECTED,
        't25_task_hashes': EXPECTED_T25_TASK_HASHES, 'safe_tail_sha256': EXPECTED_SAFE_TAIL_SHA256,
        'changed_files': ['train/_run.py', 'train/_train_config.py', 'prior/*'],
        'new_files': ['train/_t25_regression_adapter.py', 'train/_g5sc_runtime_probe.py', 'prior/_t25_safe_tail.py'],
        'assembled_python_hashes': {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*.py'))},
        'native_model_task_config': {'max_classes': 0, 'num_quantiles': 999, 'bias_free_ln': False,
            'shared_depth_icl_enabled': True, 'shared_depth_icl_dataset_conditioned': True,
            'shared_depth_icl_rho': 1.0, 'shared_depth_icl_num_passes': [3, 4]},
        'task': {'loss': 'mean pinball at alpha=0.001,...,0.999',
            'continuous_label_encoders': ['col_embedder.y_encoder', 'icl_predictor.y_encoder'],
            'regression_target_prior': 'rw_sample50', 'regression_target_mix_probability': 0.25,
            'safe_tail': True, 'loss_clip': [-8.0, 8.0], 'cross_table_e4': False,
            'effective_generator_n_jobs': 1, 'dataloader_num_workers': 4,
            'length_curriculum': False, 'regression_label_statistics': 'four zeros, native G5SC branch'},
        'added_trainable_parameters': 0, 'native_condition_gate_parameters': 52, 'base_blocks': 12,
        'gate': 'tanh(a + 0.1*(2*sigmoid(w@support_stats51)-1))', 'handwritten_gated_encoder_used': False,
    }
    (destination.parent / 'source_contract.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path, help='Original complete G5SC source')
    p.add_argument('--destination', required=True, type=Path)
    p.add_argument('--t25-source', type=Path)
    p.add_argument('--t25-prior-overlay', type=Path)
    args = p.parse_args()
    print(json.dumps(prepare(args.source, args.destination, t25_source=args.t25_source,
                             t25_prior_overlay=args.t25_prior_overlay), sort_keys=True))
