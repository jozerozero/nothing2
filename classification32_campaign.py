"""Independent classification actual32 campaign; historical artifacts are read-only.

Prepare freezes the original457 caches and selected model artifacts. Each worker
claims one (model,membership) atomically, runs a fresh process and publishes only
actual32-audited, finite, full-test results. Failed attempts are retained; no
automatic retry, result overwrite, stale-claim deletion, or parent job mutation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

BASE = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
NAME = 'classification_actual32_20260921_v1'
STAGE = BASE / 'stage' / NAME
ROOT = BASE / 'evaluation' / NAME
FT = BASE / 'stage/reg_loop3_step22175_finetune50_20260921_v1'
PYTHON = '/vast/users/guangyi.chen/causal_group/zijian.li/tabicl_causal/tabicl-main-paper2602-dataset/.conda_env/bin/python3'
COUNTS = {'talent': 200, 'BCCO': 106, 'OpenML-CC18': 62, 'PFN': 29, 'TabArena': 33, 'TabZilla': 27}
MODELS = ('tabpfn2', 'tabpfn25', 'tabpfn3', 'limix2m', 'limix16m',
          'tabiclv1', 'tabiclv2', 'taffy_loop3', 'taffy_loop4', 'mitra2')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    value = {'path': str(path), 'size_bytes': before.st_size,
             'mtime_ns': before.st_mtime_ns, 'sha256': sha(path)}
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'File changed during freeze')
    return value


def verify(rec, full=False):
    p = Path(rec['path']).resolve(strict=True)
    s = p.stat()
    require(str(p) == rec['path'] and s.st_size == rec['size_bytes']
            and s.st_mtime_ns == rec['mtime_ns'], 'Frozen file changed: ' + str(p))
    if full:
        require(sha(p) == rec['sha256'], 'Frozen content changed: ' + str(p))
    return p


def atomic(path, obj, immutable=True):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT.resolve()), 'Writes must stay inside new campaign')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    try:
        if immutable:
            os.link(temp, path)
        else:
            os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def source_records(root):
    root = Path(root)
    return [identity(p) for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts]


def prepare():
    require(not (ROOT / 'manifest.json').exists(), 'Campaign already frozen; do not replace manifest')
    original = FT / 'mitra_class32_457_20260921_v1/manifest.json'
    # The existing Mitra campaign owns the verified frozen cache inventory only.
    data = read(original)
    require(data['manifest_id'] == digest_obj({k: v for k, v in data.items() if k != 'manifest_id'}),
            'Historical frozen classification inventory failed integrity check')
    rows = data['rows']
    require(len(rows) == len({r['dataset'] for r in rows}) == 457
            and dict(Counter(r['suite'] for r in rows)) == COUNTS, 'Expected exact classification457')
    require([r['dataset_index'] for r in rows] == list(range(457)), 'Dataset ordering changed')
    for row in rows:
        verify(row['cache'], full=True)
    bench = BASE / 'stage/benchmark8_seven_suites_20260820_v1'
    tabstage = BASE / 'stage/tabpfn_v2_v25_v3_exact178_20260820_v1'
    limstage = BASE / 'stage/five_method_exact178_rw4096_step500_20260818_v1'
    loopstage = BASE / 'stage/loop34_top10_standard457_bg40_20260914_v1'
    official = BASE / 'hard96_observed_label_expanding_power2_final25k_v2_2'
    mitstage = BASE / 'stage/mitra_dual_standard681_bg1_20260915_v1'
    tabweights = {r['name']: r for r in read(tabstage / 'model_manifest.json')['models']}
    selections = {r['name']: r for r in read(loopstage / 'selection_manifest.json')['selected']}
    configs = {}
    for key in MODELS:
        cfg = {'model_key': key, 'n_estimators': 32, 'new_results_only': True,
               'python': PYTHON, 'env': {}, 'runtime_sources': {}, 'seed': 0}
        if key.startswith('tabpfn'):
            cfg.update(adapter='classification32_tabpfn', source_root=str(tabstage / 'source'),
                checkpoint_path=tabweights[key]['path'], source_commit='38f574987deb5f313a832e2b42108ee3ef190e85',
                hierarchy_helper_path=str(tabstage / 'hierarchical_tabpfn.py'), test_chunk=4096)
            cfg['env'] = {'PYTHONPATH': f'{tabstage}/deps:{tabstage}/source/src:{tabstage}:{bench}',
                          'HF_HOME': str(tabstage / 'hf_cache')}
            cfg['source_files'] = source_records(tabstage / 'source/src/tabpfn')
        elif key.startswith('limix'):
            repo = limstage / 'limix16m_source'
            checkpoint = (BASE / 'stage/limix2m_adapted_exact178_20260810_v7/weights/LimiX-2M.ckpt'
                          if key == 'limix2m' else limstage / 'weights/LimiX-16M.ckpt')
            cfg.update(adapter='classification32_limix', source_root=str(repo), checkpoint_path=str(checkpoint),
                       config_path=str(repo / ('config/cls_default_2M_retrieval.json' if key == 'limix2m'
                                              else 'config/cls_default_16M_retrieval.json')),
                       hierarchy_helper_path=str(bench / 'run_limix_lane.py'))
            cfg['env'] = {'PYTHONPATH': f'{limstage}/limix16m_python_deps:{repo}:{bench}'}
            cfg['source_files'] = [identity(repo / name) for name in
                ('inference/predictor.py', 'inference/inference_method.py', 'inference/preprocess.py',
                 'utils/data_utils.py', 'utils/retrieval_utils.py', 'model/layer.py',
                 'model/transformer.py', 'utils/loading.py')]
        elif key.startswith('tabicl') or key.startswith('taffy'):
            if key.startswith('taffy'):
                loop, step = (3, 19650) if key == 'taffy_loop3' else (4, 22400)
                picked = selections[f'loop{loop}-step{step}']
                checkpoint = picked['checkpoint']['path']
                source = loopstage / 'source'
                cfg.update(loop=loop, checkpoint_step=step)
            else:
                source = official
                filename = 'tabicl-classifier-v1-20250208.ckpt' if key == 'tabiclv1' else 'tabicl-classifier-v2-20260212.ckpt'
                checkpoint = str(Path('/vast/users/guangyi.chen/.cache/huggingface/hub/models--jingang--TabICL/snapshots/4dcd344ece2c00be9e831fdd35bed57b5ad83e19') / filename)
            cfg.update(adapter='classification32_tabicl', source_root=str(source),
                       checkpoint_path=checkpoint, seed=42)
            cfg['env'] = {'PYTHONPATH': f'{source}/src', 'CROSS_TABLE_ARM': 'E4',
                          'TABICL_SOURCE_ROOT': str(source), 'TABICL_EVAL_DISABLE_LOCAL_SRC': '1'}
            cfg['source_files'] = source_records(source / 'src/tabicl')
        else:
            cfg.update(adapter='classification32_mitra2', python=str(mitstage / 'venv/bin/python'),
                       source_root=str(mitstage), checkpoint_path=str(mitstage / 'weights/mitra2-classification/model.safetensors'),
                       config_path=str(mitstage / 'weights/mitra2-classification/config.json'),
                       legacy_helper_path=str(mitstage / 'models.py'), patches_path=str(mitstage / 'official_patches.py'),
                       hierarchy_helper_path=str(BASE / 'stage/mitra_all_classification_regression_20260823_v3/hierarchical_mitra.py'))
            cfg['env'] = {'PYTHONPATH': str(mitstage)}
            cfg['source_files'] = [identity(mitstage / x) for x in ('models.py', 'official_patches.py')]
            ag = mitstage / 'venv/lib/python3.11/site-packages/autogluon/tabular/models/mitra'
            cfg['source_files'] += [identity(ag / x) for x in ('sklearn_interface.py',
                '_internal/core/trainer_finetune.py', '_internal/data/dataset_finetune.py',
                '_internal/data/preprocessor.py', '_internal/models/tab2d.py')]
        cfg['checkpoint_identity'] = identity(cfg['checkpoint_path'])
        cfg['checkpoint_sha256'] = cfg['checkpoint_identity']['sha256']
        cfg['model_path'] = cfg['checkpoint_path']
        cfg['model_sha256'] = cfg['checkpoint_sha256']
        for label in ('config', 'hierarchy_helper', 'legacy_helper', 'patches'):
            if label + '_path' in cfg:
                cfg[label + '_identity'] = identity(cfg[label + '_path'])
                cfg[label + '_sha256'] = cfg[label + '_identity']['sha256']
        cfg['runtime_sources'] = {r['path']: r['sha256'] for r in cfg['source_files']}
        require(Path(cfg['python']).is_file(), 'Runtime missing: ' + cfg['python'])
        configs[key] = cfg
    own = [identity(p) for p in sorted(Path(__file__).parent.glob('classification32_*.py'))]
    own += [identity(Path(__file__).parent / p) for p in
            ('classification32_slurm.sh', 'mitra_class32_one.py', 'pfn_mitra_one.py', 'eval_one.py')]
    def smallest(predicate):
        eligible = [r for r in rows if predicate(r)]
        require(eligible, 'Missing representative smoke data')
        return min(eligible, key=lambda r: (r['train_rows'] + r['test_rows']) * r['features'])['dataset_index']
    smoke_rows = [smallest(lambda r: r['classes'] == 2 and r['features'] <= 4),
                  smallest(lambda r: 2 < r['classes'] <= 10),
                  smallest(lambda r: r['classes'] > 10)]
    manifest = {'schema': 1, 'name': NAME, 'created_epoch': time.time(), 'models': configs,
                'rows': rows, 'membership_count': 457, 'source_inventory': identity(original),
                'code_files': own, 'expected_results': 457 * len(MODELS),
                'historical_results_read_only': True, 'original_mitra32_not_resubmitted': True,
                'excluded_taffy': ['E4/Loop1', 'Loop2'], 'metric': 'accuracy',
                'aggregation': 'each method owns its native aggregation; actual32 per fitted hierarchy node',
                'selection_policy': 'fixed Loop3step19650 and Loop4step22400; no test-based candidate selection',
                'equal_compute_claim': False, 'smoke_indices': smoke_rows,
                'per_task_timeout_seconds': 7200, 'per_task_rss_limit_bytes': 56 * 1024**3}
    manifest['manifest_id'] = digest_obj(manifest)
    atomic(ROOT / 'manifest.json', manifest)
    print(json.dumps({'prepared': True, 'models': list(configs), 'target': manifest['expected_results'],
                      'manifest': str(ROOT / 'manifest.json')}), flush=True)


def manifest_load():
    man = read(ROOT / 'manifest.json')
    require(man['manifest_id'] == digest_obj({k: v for k, v in man.items() if k != 'manifest_id'}), 'Manifest changed')
    require(set(man['models']) == set(MODELS) and len(man['rows']) == 457, 'Campaign scope changed')
    for rec in man['code_files']:
        verify(rec, full=True)
    return man


def refresh_prelaunch_code_pins():
    """Allow audited code-only correction before any job/task has been submitted."""
    for name in ('submission_state.json', 'claims', 'results', 'smoke', 'gates', 'preflight', 'bindings'):
        require(not (ROOT / name).exists(), 'Cannot change a submitted/started campaign: ' + name)
    jobs = subprocess.check_output(['squeue', '--me', '-h', '-o', '%i|%j'], text=True)
    require(not any(line.split('|')[-1] == 'c32budget' for line in jobs.splitlines()),
            'An active actual32 job exists; code refresh prohibited')
    old = read(ROOT / 'manifest.json')
    require(old['manifest_id'] == digest_obj({k: v for k, v in old.items() if k != 'manifest_id'}),
            'Prelaunch manifest integrity error')
    for row in old['rows']:
        verify(row['cache'], full=True)
    for config in old['models'].values():
        model_verify(config)
    own = [identity(rec['path']) for rec in old['code_files']]
    archive = ROOT / 'prelaunch_manifests' / (old['manifest_id'] + '.json')
    if archive.exists():
        require(read(archive) == old, 'Prelaunch archive mismatch')
    else:
        atomic(archive, old)
    new = {k: v for k, v in old.items() if k != 'manifest_id'}
    new.update(code_files=own, prelaunch_previous_manifest_id=old['manifest_id'],
               prelaunch_code_refresh_epoch=time.time())
    new['manifest_id'] = digest_obj(new)
    atomic(ROOT / 'manifest.json', new, immutable=False)
    print(json.dumps({'code_pins_refreshed_before_any_submission': True,
                      'manifest_id': new['manifest_id'], 'archived_previous': str(archive)}), flush=True)


def model_verify(cfg):
    for key, rec in cfg.items():
        if key.endswith('_identity'):
            verify(rec)
    for rec in cfg['source_files']:
        verify(rec)


def run_one(args):
    import numpy as np
    import torch
    man = manifest_load()
    require(args.model in man['models'] and 0 <= args.index < 457, 'Unknown task')
    row, cfg = man['rows'][args.index], man['models'][args.model]
    model_verify(cfg)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expected one actually visible GPU')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    path = verify(row['cache'], full=True)
    with np.load(path, allow_pickle=False) as z:
        xs, xt = np.asarray(z['X_train'], dtype=np.float32), np.asarray(z['X_test'], dtype=np.float32)
        yr, yq = np.asarray(z['y_train']).reshape(-1), np.asarray(z['y_test']).reshape(-1)
        ys, yt = yr.astype(np.int64), yq.astype(np.int64)
        require(np.array_equal(yr, ys) and np.array_equal(yq, yt), 'Noninteger classification labels')
    require(len(xs) == len(ys) and len(xt) == len(yt), 'Frozen row counts invalid')
    require(np.isfinite(xs).all() and np.isfinite(xt).all(), 'Nonfinite input')
    classes = np.unique(ys)
    require(np.array_equal(classes, np.arange(len(classes))) and set(np.unique(yt)) <= set(classes), 'Dense label contract')
    cfg = dict(cfg, dataset=row['dataset'], dataset_index=args.index)
    start = time.time()
    adapter = importlib.import_module(cfg['adapter'])
    probability, audit = adapter.predict(args.model, {'X_train': xs, 'y_train': ys, 'X_test': xt}, cfg)
    probability = np.asarray(probability)
    require(probability.shape == (len(yt), len(classes)) and np.isfinite(probability).all()
            and np.all(probability >= 0) and np.allclose(probability.sum(axis=1), 1, atol=2e-5), 'Invalid full-test probabilities')
    require(audit.get('actual32_verified') is True and audit.get('actual_ensemble_count') == 32
            and audit.get('actual_members_per_test_row') == 32, 'Actual32 contributor audit absent')
    verify(row['cache'])
    model_verify(cfg)
    score = float(np.mean(probability.argmax(axis=1) == yt))
    result = {'complete': True, 'protocol_validation': True, 'actual32_verified': True,
              'actual_ensemble_count': 32, 'model': args.model, 'dataset': row['dataset'],
              'dataset_index': args.index, 'suite': row['suite'], 'task_kind': 'classification',
              'manifest_id': man['manifest_id'], 'input_fingerprint': row['input_fingerprint'],
              'checkpoint_sha256': cfg['checkpoint_sha256'], 'checkpoint_path': cfg['checkpoint_path'],
              'accuracy': score, 'train_rows': len(ys), 'test_rows': len(yt), 'features': xs.shape[1],
              'full_test_split': True, 'ensemble_audit': audit, 'seconds': time.time() - start,
              'job': os.environ.get('SLURM_JOB_ID'), 'step': os.environ.get('SLURM_STEP_ID'),
              'node': socket.gethostname(), 'finished_epoch': time.time(),
              'gpu_visibility': {k: os.environ.get(k) for k in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')}}
    output = ROOT / ('smoke' if args.smoke else 'results') / args.model / f'row-{args.index:03d}.json'
    atomic(output, result)
    print(json.dumps({'complete': True, 'model': args.model, 'dataset': row['dataset'], 'accuracy': score,
                      'actual32_verified': True, 'output': str(output)}), flush=True)


def valid_result(path, man, model, row):
    r = read(path)
    require(r.get('complete') is True and r.get('actual32_verified') is True
            and r['actual_ensemble_count'] == 32 and r['manifest_id'] == man['manifest_id']
            and r['model'] == model and r['dataset_index'] == row['dataset_index']
            and r['input_fingerprint'] == row['input_fingerprint']
            and r['checkpoint_sha256'] == man['models'][model]['checkpoint_sha256']
            and math.isfinite(r['accuracy']) and 0 <= r['accuracy'] <= 1
            and r['test_rows'] == row['test_rows'] and r['train_rows'] == row['train_rows']
            and r['full_test_split'] is True and r['protocol_validation'] is True,
            'Existing new result failed validation')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'one', 'refresh-prelaunch-code'))
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--index', type=int)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'refresh-prelaunch-code':
        refresh_prelaunch_code_pins()
    else:
        run_one(args)


if __name__ == '__main__':
    main()
