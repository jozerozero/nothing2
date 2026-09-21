"""Freeze an isolated TabFM native-default + authorized hierarchy campaign."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from eval_one import object_digest, publish_new, require

BASE = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
NAME = 'tabfm_defaults_standard681_20260922_v1'
STAGE = BASE / 'stage' / NAME
OUT = BASE / 'evaluation' / NAME
FT = BASE / 'stage/reg_loop3_step22175_finetune50_20260921_v1'
COMMIT = 'fbb665569425fd2f490c6576b3af967876fe11ff'
WEIGHT_COMMIT = '77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83'

def identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f'Input changed while freezing {path}')
    return {'path': str(path), 'size_bytes': after.st_size, 'mtime_ns': after.st_mtime_ns,
            'sha256': h.hexdigest()}

def read(path):
    return json.loads(Path(path).read_text())

def verify_manifest(man):
    require(man['manifest_id'] == object_digest({k:v for k,v in man.items() if k != 'manifest_id'}),
            'Manifest identity mismatch')

def prepare():
    dest = OUT / 'manifest.json'
    require(not dest.exists(), 'Campaign already frozen; will not overwrite or change protocol')
    repo = Path(__file__).resolve().parent
    clf_path = FT / 'mitra_class32_457_20260921_v1/manifest.json'
    reg_path = FT / 'eval224/manifest.json'
    clf, reg = read(clf_path), read(reg_path)
    for man, size in ((clf,457),(reg,224)):
        verify_manifest(man)
        require(len(man['rows']) == size and man['membership_count'] == size, 'Membership count mismatch')
        require([r['dataset_index'] for r in man['rows']] == list(range(size)), 'Membership order mismatch')
    plan_path = BASE / 'evaluation/table6_remaining19_standard_hpo_bg8_20260912_v1/plan.json'
    plan = read(plan_path)
    rows = {r['dataset']:r for r in plan['rows'] if r['task_kind'] == 'classification'}
    require(len(rows) == 457, 'Raw classification membership mismatch')
    inputs = {}
    for r in clf['rows']:
        orig = rows[r['dataset']]
        require(orig['source_path'] == r['source_path'], 'Raw source directory mismatch')
        require((orig['train_rows'],orig['test_rows'],orig['features']) ==
                (r['train_rows'],r['test_rows'],r['features']), 'Standard split dimension mismatch')
        files = [dict(f) for f in orig['input_files'] if f['path'] != r['cache']['path']]
        require(files, 'Raw classification source files absent')
        for f in files:
            stat = Path(f['path']).stat()
            require((stat.st_size, stat.st_mtime_ns) == (f['size_bytes'], f['mtime_ns']),
                    f"Raw input metadata changed: {f['path']}")
        inputs[str(r['dataset_index'])] = files
    official = STAGE / 'official'
    commit = subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()
    require(commit == COMMIT, 'Unpinned official source')
    require(not subprocess.check_output(['git','-C',str(official),'status','--porcelain'],text=True).strip(),
            'Official source dirty')
    weights = {}
    for kind in ('classification','regression'):
        d = STAGE / 'weights' / kind
        weights[kind] = {'directory':str(d), 'config':identity(d/'config.json'),
                         'weights':identity(d/'model.safetensors')}
    files = sorted((official/'tabfm').rglob('*.py')) + [official/'pyproject.toml']
    names = ('tabfm_default_one.py','tabfm_hierarchical.py','tabfm_default_dispatch.py',
             'tabfm_default_slurm.sh','tabfm_prepare.py','eval_one.py','classification32_dispatch.py',
             'classification32_campaign.py','classification32_submit.py','pfn_mitra_one.py')
    worker_sources = [identity(repo/n) for n in names]
    hierarchy_indices = [r['dataset_index'] for r in clf['rows'] if r['classes'] > 10]
    size = lambda r: (r['train_rows'] + r['test_rows']) * r['features']
    native_smoke = min((r for r in clf['rows'] if r['classes'] <= 10), key=size)
    hierarchy_smoke = min((r for r in clf['rows'] if r['classes'] > 10),
                          key=lambda r: (r['classes'], size(r)))
    regression_smoke = sorted(reg['rows'], key=lambda r:r['work_size'])[:2]
    smoke_tasks = [{'task_kind':'classification','dataset_index':r['dataset_index']}
                   for r in (native_smoke,hierarchy_smoke)] + [
                   {'task_kind':'regression','dataset_index':r['dataset_index']} for r in regression_smoke]
    import importlib.metadata as md
    versions = {n:md.version(n) for n in ('tabfm','torch','numpy','pandas','scikit-learn','scipy',
                                        'jaxtyping','typeguard','huggingface-hub','safetensors')}
    doc = {
        'schema':1, 'name':NAME,'created_epoch':time.time(),'output_root':str(OUT),
        'classification_manifest':identity(clf_path),'regression_manifest':identity(reg_path),
        'classification_raw_inputs':inputs,'raw_source_manifest':identity(plan_path),
        'official_source':{'path':str(official),'commit':COMMIT,'files':[identity(p) for p in files]},
        'weights':weights, 'weight_repository':'google/tabfm-1.0.0-pytorch',
        'weight_revision':WEIGHT_COMMIT, 'worker_sources':worker_sources,
        'worker_python':str(STAGE/'venv/bin/python'), 'worker_script':identity(repo/'tabfm_default_one.py'),
        'versions':versions,'membership_count':681,'classification_count':457,'regression_count':224,
        'classification_suite_counts':clf.get('suite_counts'), 'regression_suite_counts':reg.get('suite_counts'),
        'hierarchical_classification_indices':hierarchy_indices,
        'smoke_tasks':smoke_tasks,
        'protocol':{'constructor':'Native TabFMClassifier / TabFMRegressor defaults, no .ensemble()',
          'classification_over10':'TabICLv2 balanced sorted-label hierarchical probability chain; native TabFM at every node',
          'n_estimators_per_node':32,'random_state':42,'native_dtype':'bfloat16','max_num_features':500,
          'max_num_rows':None,'batch_size':1,'cache_context':False,'test_split':'existing standard457/224, full test',
          'regression_targets':'original units; native internal scaler only','finetuning':False,
          'results_independent_of_previous_campaigns':True,
          'fairness_note':'Native defaults, not strict common compute budget; hierarchy costs more than one estimator ensemble'},
        'per_task_rss_limit_bytes':48*(1<<30),'per_task_timeout_seconds':7200,
        'no_automatic_retry':True,'training_or_old_results_modified':False}
    doc['manifest_id'] = object_digest(doc)
    publish_new(dest, doc)
    print(json.dumps({'manifest':str(dest),'manifest_id':doc['manifest_id'],
                      'classification':457,'regression':224,'hierarchical_tasks':len(hierarchy_indices),
                      'versions':versions}),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.parse_args(); prepare()
