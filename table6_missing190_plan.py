#!/usr/bin/env python3
"""Freeze a NEW HPO campaign for the exact 190 missing fixed-best CLS pairs.

Reads metadata only. Never copies historical scores/configurations, loads data,
changes either parent plan, or writes any existing evaluation result directory.
Scientific rows are exact copies of the pinned standard681 classification rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
FIXED_NAME = 'table6_standard457_fixedbest_bg1_gpu_recovery_20260909_v2'
R19_NAME = 'table6_remaining19_standard_hpo_bg8_20260912_v1'
NAME = 'table6_missing190_standard_hpo_20260922_v1'
FIXED_PLAN_ID = '95ed50cd7348ed167f71ff159f6af14cc351e67959d6d227f9201bdc024fbb85'
R19_PLAN_ID = '47c3d448249235ad4f7ae6248424a1f8f38094ae204d2d4ece472b4effb61aff'
METHODS = {'CatBoost': 'catboost', 'LightGBM': 'lightgbm', 'XGBoost': 'xgboost',
           'TabM': 'tabm', 'TabR': 'tabr', 'TabTransformer': 'tabtransformer',
           'TabNet': 'tabnet', 'SwitchTab': 'switchtab', 'TabCaps': 'tabcaps'}
CPU_METHODS = ('CatBoost', 'LightGBM', 'XGBoost')
SEEDS = list(range(15))
SUITES = {'talent': 200, 'BCCO': 106, 'OpenML-CC18': 62, 'PFN': 29, 'TabArena': 33, 'TabZilla': 27}
BLOCK_REASON = 'no validated saved selected_config; no defaults or new HPO authorized'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def object_digest(value):
    # Identical canonicalization to BOTH frozen parent plan implementations.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_digest(value, label):
    require(isinstance(value, str) and len(value) == 64 and
            all(char in '0123456789abcdef' for char in value), 'invalid SHA256: ' + label)


def read_pinned(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'source plan changed while reading')
    return json.loads(data), {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
                              'size_bytes': len(data)}


def verify_parent(plan, expected, label):
    value = dict(plan)
    require(value.pop('plan_id', None) == expected, label + ' plan_id differs from pinned experiment')
    require(object_digest(value) == expected, label + ' plan content hash mismatch')
    require(plan.get('seeds') == SEEDS, label + ' seed contract mismatch')


def input_fingerprint(row):
    return object_digest({'task_kind': row['task_kind'], 'dataset': row['dataset'], 'suite': row['suite'],
                          'format': row['format'], 'input_files': row['input_files'],
                          'target_feature': row.get('target_feature'), 'class_labels': row.get('class_labels')})


def verify_row(fixed, row):
    require(row.get('task_kind') == 'classification' and not row.get('blocked_reason'), 'not a usable original classification row')
    require(row.get('row_id') == 'classification::' + row['dataset'], 'unexpected canonical row_id')
    # The fixed-best rows are unmodified copies of the original class manifest.
    # Every one of their fields must be retained identically in the augmented R19 row.
    for key, value in fixed.items():
        require(key in row and row[key] == value, 'fixed/R19 row conflict: ' + row['dataset'] + '/' + key)
    for key in ('source_path', 'cache_path'):
        require(isinstance(row.get(key), str) and Path(row[key]).is_absolute(), 'missing absolute source/cache path')
    for key in ('train_rows', 'test_rows', 'features', 'classes'):
        require(type(row.get(key)) is int and row[key] > 0, 'invalid exact row metadata: ' + key)
    require(row['classes'] >= 2 and len(row['class_labels']) == row['classes'], 'class label metadata mismatch')
    require(row.get('membership_source') == 'standard457_classification', 'classification membership source changed')
    require(isinstance(row.get('input_files'), list) and len(row['input_files']) >= 2, 'frozen source files absent')
    files = row['input_files']
    require(len({record['path'] for record in files}) == len(files), 'duplicate frozen source path')
    require(row['cache_path'] in {record['path'] for record in files}, 'original numerical cache is not pinned')
    for record in files:
        require(Path(record['path']).is_absolute(), 'relative frozen input path')
        require(type(record['size_bytes']) is int and record['size_bytes'] > 0 and
                type(record['mtime_ns']) is int and record['mtime_ns'] > 0, 'invalid frozen source identity')
        if 'sha256' in record:
            check_digest(record['sha256'], record['path'])
    check_digest(row.get('input_fingerprint'), 'input_fingerprint')
    require(input_fingerprint(row) == row['input_fingerprint'], 'row input_fingerprint does not match frozen inputs')


def check_output_root(output_root, sources):
    output = Path(output_root)
    require(output.is_absolute(), 'new output_root must be absolute')
    require(not output.is_symlink(), 'new output_root must not be a symlink')
    output = output.resolve()
    require(output.name.startswith('table6_missing190_'), 'use an explicit new table6_missing190_* output namespace')
    protected = [ROOT/'evaluation'/R19_NAME, ROOT/'evaluation/table6_standard457_fixedbest_gt1_20260909_v1',
                 ROOT/'evaluation/benchmark8_seven_suites_20260820_v1']
    protected.extend(Path(record['path']).parent for record in sources.values())
    for original in protected:
        original = original.resolve()
        require(output != original and not output.is_relative_to(original) and not original.is_relative_to(output),
                'new output overlaps a frozen source/result directory: ' + str(original))
    if output.exists():
        require(output.is_dir() and not any(output.iterdir()), 'refuse to prepare against an existing nonempty output')
    return str(output)


def build_documents(fixed, r19, output_root, source_records):
    """Pure metadata construction (except checking destination safety)."""
    verify_parent(fixed, FIXED_PLAN_ID, 'fixed-best')
    verify_parent(r19, R19_PLAN_ID, 'remaining19')
    require(fixed.get('classification_memberships') == 457 and fixed.get('hpo_trials') == 0, 'wrong fixed-best scope')
    require(r19.get('classification_memberships') == 457 and r19.get('regression_memberships') == 224 and
            r19.get('hpo_trials') == 100 and len(r19['pairs']) == 12482 and not r19['blocked'], 'wrong remaining19 scope')
    require(set(source_records) == {'fixedbest_plan', 'remaining19_plan'}, 'missing parent file identities')
    for label, record in source_records.items():
        check_digest(record['sha256'], label)
        require(Path(record['path']).is_absolute() and record['size_bytes'] > 0, 'invalid parent file identity')
    original_manifest = r19['source_contract']['classification_manifest']
    require(original_manifest['sha256'] == fixed['benchmark_manifest_sha256'], 'parent classification manifest file SHA differs')
    old_rows = {row['dataset']: row for row in fixed['rows']}
    new_rows = {row['dataset']: row for row in r19['rows'] if row.get('task_kind') == 'classification'}
    require(len(old_rows) == len(fixed['rows']) == len(new_rows) == 457 and len(r19['rows']) == 681,
            'duplicate or missing parent dataset rows')
    require(set(old_rows) == set(new_rows), 'standard457 membership differs between parent plans')
    require(dict(Counter(row['suite'] for row in old_rows.values())) == SUITES, 'standard457 suite counts changed')
    for dataset in old_rows:
        verify_row(old_rows[dataset], new_rows[dataset])
    blocked = fixed['blocked']
    require(len(blocked) == 190, 'expected exactly190 original blocked pairs')
    blocked_keys = {(entry['method'], entry['dataset']) for entry in blocked}
    require(len(blocked_keys) == 190 and {entry['method'] for entry in blocked} == set(METHODS),
            'duplicate blocked pair or unexpected/missing method')
    ready = {(pair['method'], pair['dataset']) for pair in fixed['pairs']}
    require(len(ready) == len(fixed['pairs']) == 4380 and not ready & blocked_keys, 'ready/blocked conflict')
    universe = {(method, dataset) for method in (*METHODS, 'AutoGluon') for dataset in old_rows}
    require(ready | blocked_keys == universe, 'fixed-best ready/blocked universe is incomplete or contains unknown methods')
    pairs, evidence = [], []
    for entry in sorted(blocked, key=lambda value: (value['method'], value['dataset'])):
        require(set(entry) == {'method', 'dataset', 'reason'} and entry['reason'] == BLOCK_REASON,
                'unknown original blocking reason/schema; manual review required')
        method, dataset = entry['method'], entry['dataset']
        row = new_rows[dataset]
        key = hashlib.sha256((method+'\0'+row['row_id']).encode()).hexdigest()[:24]
        evidence_digest = object_digest(entry)
        pairs.append({'key': key, 'method': method, 'talent_name': METHODS[method], 'dataset': dataset,
                      'row_id': row['row_id'], 'task_kind': 'classification', 'suite': row['suite'],
                      'input_fingerprint': row['input_fingerprint'], 'work_size': row['work_size'],
                      'hpo_trials': 100, 'seeds': list(SEEDS), 'batch_size': 1024, 'max_epoch': 200,
                      'execution_mode': 'cpu' if method in CPU_METHODS else 'gpu',
                      'selection_kind': 'fresh_training_validation_hpo',
                      'legacy_fixedbest_key': hashlib.sha256((method+'\0'+dataset).encode()).hexdigest()[:24],
                      'original_blocked_record_sha256': evidence_digest})
        evidence.append({'record': deepcopy(entry), 'record_sha256': evidence_digest,
                         'fixedbest_row_sha256': object_digest(old_rows[dataset]),
                         'remaining19_row_sha256': object_digest(row)})
    require(len({pair['key'] for pair in pairs}) == 190, 'new pair key collision')
    datasets = sorted({pair['dataset'] for pair in pairs})
    plan = {'schema': 1, 'name': NAME, 'protocol': 'standard457_missing190_fresh_hpo100_seed15_v1',
            'output_root': check_output_root(output_root, source_records), 'task_kind': 'classification',
            'classification_memberships': len(datasets), 'source_classification_memberships': 457,
            'applicable_pair_target': 190, 'methods': dict(METHODS), 'cpu_methods': list(CPU_METHODS),
            'hpo_trials': 100, 'seeds': list(SEEDS), 'hpo_seed': 0,
            'hpo_sampler': 'Optuna TPE; sampler seed equals next trial number; persistent study history',
            'failed_trials_count_toward_budget': True, 'all_failed_study_has_no_selected_model': True,
            'selection_metric': 'validation Accuracy', 'test_evaluated_during_hpo': False,
            'classification_decision': 'native standardized probabilities followed by argmax; no threshold tuning',
            'batch_size': 1024, 'max_epoch': 200, 'precision': 'unchanged TALENT method default; no automatic precision reduction',
            'preprocessing': 'same pinned R19 loader: original train/validation only; exact frozen numerical-cache match',
            'test_split': 'unchanged original standard457 full test rows, identical to both frozen parent plans',
            'validation': r19['validation'], 'rows': [deepcopy(new_rows[dataset]) for dataset in datasets],
            'pairs': pairs, 'pair_counts_by_method': dict(sorted(Counter(pair['method'] for pair in pairs).items())),
            'original_blocked_evidence': evidence,
            'source_contract': {'parent_files': deepcopy(source_records), 'fixedbest_plan_id': FIXED_PLAN_ID,
                                'remaining19_plan_id': R19_PLAN_ID,
                                'classification_manifest': deepcopy(original_manifest),
                                'remaining19_loader_contract': deepcopy(r19['source_contract'])},
            'preservation': {'parent_plans_unchanged': True, 'parent_rows_unchanged': True,
                             'historical_scores_or_selected_configs_imported': False,
                             'separate_claims_studies_selected_seeds_results': True,
                             'never_publish_into_fixedbest_or_remaining19_output': True},
            'limitations': ['New HPO and historical fixed-best have different selection provenance; retain that distinction in reports.',
                            'Metadata preparation is not runtime/data/model smoke validation. Fit support for all nine methods must be separately verified.']}
    plan['plan_id'] = object_digest(plan)
    return plan


def build(fixed_plan_path, r19_plan_path, output_root):
    fixed, fixed_identity = read_pinned(fixed_plan_path)
    r19, r19_identity = read_pinned(r19_plan_path)
    return build_documents(fixed, r19, output_root, {'fixedbest_plan': fixed_identity, 'remaining19_plan': r19_identity})


def publish_new(path, plan):
    """Atomic exclusive publication. Existing files are never replaced."""
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink() and not path.exists(), 'output-plan must be new and absolute')
    output_root = Path(plan['output_root'])
    require(path.parent.resolve() == output_root.resolve(), 'output-plan must be directly inside the new output_root')
    output_root.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.missing190-plan-', suffix='.tmp', dir=output_root)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)  # Fails if another preparer won; no overwrite race.
    finally:
        temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-plan', type=Path, default=ROOT/'stage'/FIXED_NAME/'plan.json')
    parser.add_argument('--r19-plan', type=Path, default=ROOT/'stage'/R19_NAME/'plan.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--output-plan', type=Path, required=True)
    args = parser.parse_args()
    plan = build(args.fixed_plan, args.r19_plan, args.output_root)
    publish_new(args.output_plan, plan)
    print(json.dumps({'plan_id': plan['plan_id'], 'path': str(args.output_plan), 'output_root': plan['output_root'],
                      'pair_count': len(plan['pairs']), 'dataset_count': len(plan['rows']),
                      'pair_counts_by_method': plan['pair_counts_by_method']}), flush=True)


if __name__ == '__main__':
    main()
