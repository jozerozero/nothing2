"""Derive an isolated regression source; abort on any unrecognized base code."""
import argparse
import ast
import hashlib
import json
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
EXPECTED = {
    "_model/tabicl.py": "583786e5ffd70b9eb425bca8e52aed415dadcdf4254c908b3b6fa52026a56491",
    "_model/learning.py": "93e1a0c9e6f9a0c8f3d6114dec414ec859d34d6cf4a97010241d2d704473a81f",
    "_model/encoders.py": "a3e0e9bde47e7f587c4a196d124ea631e25755d4f5ac5d6a8568618a4e299a4e",
    "train/_run.py": "856985faffcebd951519f5ca611f0e6709062c38a35e2251564a325fce3d83a1",
    "train/_train_config.py": "576a6fb5d6d30a9d6b5fedc7ae57299031aba9dd8722977577bde5592435ef18",
}


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)


def method(text, cls, name, transform):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines(keepends=True)
    old = ''.join(lines[fn.lineno-1:fn.end_lineno])
    return ''.join(lines[:fn.lineno-1]) + transform(old) + ''.join(lines[fn.end_lineno:])


def context_arg(text):
    signature_end = text.index('        """')
    signature, rest = text[:signature_end], text[signature_end:]
    if '\n    )' in signature:
        signature = replace(signature, '\n    )', '\n        dataset_context: Optional[Tensor] = None,\n    )')
    else:
        signature = replace(signature, ') -> Tensor:', ', dataset_context: Optional[Tensor] = None) -> Tensor:')
    return signature + rest


def prepare(source, destination):
    base = source / 'src/tabicl'
    for name, expected in EXPECTED.items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest() == expected, name
    assert not destination.exists(), f'refuse overwrite: {destination}'
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '._*', '__MACOSX'))
    out = destination / 'src/tabicl'
    changes = []
    def save(name, text):
        ast.parse(text)
        (out/name).write_text(text)
        changes.append(name)

    text = (out/'_model/learning.py').read_text()
    text = replace(text, 'from .encoders import Encoder', 'from .g5sc_regression_loop import GatedLoopEncoder as Encoder')
    text = method(text, 'ICLearning', '__init__', lambda s: replace(replace(s,
        '        recompute: bool = False,',
        '        recompute: bool = False,\n        shared_depth_num_passes: int = 1,'),
        '            recompute=recompute,',
        '            recompute=recompute,\n            shared_depth_num_passes=shared_depth_num_passes,'))
    # Propagate the context through every inference batching/cache path.
    names = ['_icl_predictions', '_predict_standard', '_inference_forward', 'forward',
             '_icl_predictions_repr_cache', 'forward_with_repr_cache',
             '_icl_predictions_with_cache', 'forward_with_cache']
    for name in names:
        text = method(text, 'ICLearning', name, context_arg)
    text = replace(text, 'self.tf_icl(R, train_size=train_size)',
                   'self.tf_icl(R, train_size=train_size, dataset_context=dataset_context)', 2)
    text = replace(text, 'inputs=OrderedDict([("R", R), ("y_train", y_train)])',
                   'inputs=OrderedDict([("R", R), ("y_train", y_train), ("dataset_context", dataset_context)])')
    text = replace(text, 'out = self._predict_standard(R, y_train)',
                   'out = self._predict_standard(R, y_train, dataset_context=dataset_context)')
    text = replace(text, 'out = self._icl_predictions(R, y_train)',
                   'out = self._icl_predictions(R, y_train, dataset_context=dataset_context)')
    text = replace(text, 'self._inference_forward(R, y_train, return_logits, softmax_temperature, mgr_config)',
                   'self._inference_forward(R, y_train, return_logits, softmax_temperature, mgr_config, dataset_context)')
    text = replace(text, 'inputs=OrderedDict([("R", R), ("train_size", train_size)])',
                   'inputs=OrderedDict([("R", R), ("train_size", train_size), ("dataset_context", dataset_context)])')
    text = method(text, 'ICLearning', '_icl_predictions_with_cache', lambda s: replace(s,
        '            store_cache=store_cache,', '            store_cache=store_cache,\n            dataset_context=dataset_context,'))
    text = method(text, 'ICLearning', 'forward_with_cache', lambda s: replace(s,
        '                    ("store_cache", store_cache),',
        '                    ("store_cache", store_cache),\n                    ("dataset_context", dataset_context),'))
    save('_model/learning.py', text)

    text = (out/'_model/tabicl.py').read_text()
    text = replace(text, 'from .learning import ICLearning',
                   'from .learning import ICLearning\nfrom .g5sc_regression_loop import SupportSchemaStatistics')
    text = method(text, 'TabICL', '__init__', lambda s: replace(replace(replace(s,
        '        recompute: bool = False,',
        '        recompute: bool = False,\n        shared_depth_icl_num_passes: int = 1,'),
        '        self.max_classes = max_classes',
        '        self.max_classes = max_classes\n'
        '        self.shared_depth_icl_num_passes = int(shared_depth_icl_num_passes)\n'
        '        if shared_depth_icl_num_passes > 1 and max_classes != 0:\n'
        '            raise ValueError("This isolated loop implementation is regression-only")\n'
        '        self.shared_depth_condition_stats = SupportSchemaStatistics(max_classes=0) if shared_depth_icl_num_passes > 1 else None'),
        '        self.icl_predictor = ICLearning(',
        '        self.icl_predictor = ICLearning(\n            shared_depth_num_passes=shared_depth_icl_num_passes,'))
    text = method(text, 'TabICL', '_train_forward', lambda s: replace(replace(s,
        '        B, T, H = X.shape',
        '        B, T, H = X.shape\n'
        '        dataset_context = self.shared_depth_condition_stats(X, y_train, d=d, total_seq_len=T) if self.shared_depth_condition_stats is not None else None'),
        'self.icl_predictor(representations, y_train=y_train)',
        'self.icl_predictor(representations, y_train=y_train, dataset_context=dataset_context)'))
    text = method(text, 'TabICL', '_inference_forward', lambda s: replace(replace(s,
        '        train_size = y_train.shape[1]',
        '        dataset_context = self.shared_depth_condition_stats(X, y_train, total_seq_len=X.shape[1]) if self.shared_depth_condition_stats is not None else None\n'
        '        train_size = y_train.shape[1]'),
        '            mgr_config=inference_config.ICL_CONFIG,',
        '            mgr_config=inference_config.ICL_CONFIG,\n            dataset_context=dataset_context,'))
    cache_context = '''        dataset_context = None
        if self.shared_depth_condition_stats is not None:
            if store_cache:
                dataset_context = self.shared_depth_condition_stats(X, y_train, total_seq_len=X.shape[1])
                self._cache.g5sc_dataset_context = dataset_context.detach()
            else:
                dataset_context = getattr(self._cache, "g5sc_dataset_context", None)
                if dataset_context is None:
                    raise ValueError("Loop checkpoint requires cached support context")

'''
    text = method(text, 'TabICL', 'forward_with_cache', lambda s: replace(replace(s,
        '        # Column-wise embedding with cache support -> Row-wise interaction',
        cache_context + '        # Column-wise embedding with cache support -> Row-wise interaction'),
        '                mgr_config=inference_config.ICL_CONFIG,',
        '                mgr_config=inference_config.ICL_CONFIG,\n                dataset_context=dataset_context,', 2))
    save('_model/tabicl.py', text)

    text = (out/'_model/kv_cache.py').read_text()
    text = replace(text, '    num_classes: Optional[int] = None',
                   '    num_classes: Optional[int] = None\n    g5sc_dataset_context: Optional[Tensor] = None')
    text = method(text, 'TabICLCache', 'slice_batch', lambda s: replace(s,
        '            num_classes=self.num_classes,',
        '            num_classes=self.num_classes,\n'
        '            g5sc_dataset_context=self.g5sc_dataset_context[indices] if self.g5sc_dataset_context is not None else None,'))
    text = method(text, 'TabICLCache', 'to', lambda s: replace(s,
        '            num_classes=self.num_classes,',
        '            num_classes=self.num_classes,\n'
        '            g5sc_dataset_context=self.g5sc_dataset_context.to(device=device, dtype=dtype) if self.g5sc_dataset_context is not None else None,'))
    text = method(text, 'TabICLCache', 'concat', lambda s: replace(replace(s,
        '        total_batch = sum(c.train_shape[0] for c in caches)',
        '        contexts = [c.g5sc_dataset_context for c in caches if c.g5sc_dataset_context is not None]\n'
        '        if contexts and len(contexts) != len(caches):\n'
        '            raise ValueError("Cannot mix gated and non-gated caches")\n'
        '        total_batch = sum(c.train_shape[0] for c in caches)'),
        '            num_classes=caches[0].num_classes,',
        '            num_classes=caches[0].num_classes,\n'
        '            g5sc_dataset_context=torch.cat(contexts, dim=dim) if contexts else None,'))
    save('_model/kv_cache.py', text)

    text = (out/'train/_train_config.py').read_text()
    text = replace(text, '    parser.add_argument("--checkpoint_dir",',
        '    parser.add_argument("--shared_depth_icl_num_passes", type=int, choices=(1, 3, 4), default=1)\n'
        '    parser.add_argument("--checkpoint_dir",')
    save('train/_train_config.py', text)
    text = (out/'train/_run.py').read_text()
    text = replace(text, '            "recompute": self.config.recompute,',
        '            "recompute": self.config.recompute,\n'
        '            "shared_depth_icl_num_passes": self.config.shared_depth_icl_num_passes,')
    text = replace(text, '        model.to(device=self.config.device)',
        '        model.to(device=self.config.device)\n'
        '        if self.config.shared_depth_icl_num_passes > 1:\n'
        '            from tabicl._model.g5sc_runtime_probe import probe\n'
        '            probe(model, self.config.checkpoint_dir)')
    save('train/_run.py', text)
    text = (out/'train/_muon.py').read_text()
    text = replace(text, 'ns_input.reshape(len(g), -1)',
                   'ns_input.reshape(g.shape[0] if g.ndim else 1, -1)')
    save('train/_muon.py', text)
    for src, dst in [('gated_encoder.py', '_model/g5sc_regression_loop.py'),
                     ('support_stats.py', '_model/g5sc_support_stats.py'),
                     ('runtime_probe.py', '_model/g5sc_runtime_probe.py')]:
        shutil.copyfile(HERE/src, out/dst)
        changes.append(dst)
    for path in out.rglob('*.py'):
        # Bytes respect Python source encoding declarations; Finder sidecars
        # are excluded above and are not executable Python source.
        compile(path.read_bytes(), str(path), 'exec')
    report = {'baseline_hashes': EXPECTED, 'changed_files': changes,
              'source': str(source), 'derived_source': str(destination),
              'added_trainable_parameters': 52, 'base_blocks': 12,
              'added_pass_activation_recomputation': True,
              'gate': 'tanh(a + 0.1*(2*sigmoid(w@support_stats51)-1))'}
    (destination.parent/'source_contract.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, type=pathlib.Path)
    p.add_argument('--destination', required=True, type=pathlib.Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.source.resolve(), a.destination.resolve())))
