"""Freeze 224 standard regression memberships x 50 fine-tuned checkpoints.

Preparation is CPU/file-I/O only: no model import, checkpoint deserialization,
GPU initialization, data copying, result creation, or evaluation is performed.
Use --verify-data for the audited eval_data.load(row) CPU diagnostic pass only;
its numeric arrays are NOT the native TabICL model inputs. The worker consumes
official raw DataFrames and its separately frozen native inference protocol.
All filesystem locations are explicit arguments; importing this module is inert.

Checkpoints are streamed through SHA256 ONCE each during preparation. Workers
should verify only their selected checkpoint, once per worker/checkpoint load,
not hash the full 50-checkpoint set for each of the 11,200 evaluation units.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


CONTRACT = 'reg-loop3-ft50-standard224-evaluation-v1'
SOURCE_STEP = 22175
STEPS = tuple(range(22176, 22226))
REG_COUNTS = {'talent': 100, 'BCCO': 50, 'CTR23': 33, 'TabArena': 13, 'PFN': 28}
CLASS_COUNTS = {'talent': 200, 'BCCO': 106, 'OpenML-CC18': 62, 'PFN': 29, 'TabArena': 33, 'TabZilla': 27}
DATA_PROTOCOL = 'mitra-standard681-data-v1'
VENDOR_FILES = ('standard_loader.py', 'regression_suite_worker.py',
                'official_talent_regression_worker.py', 'talent_regression_contract.py',
                'prepare_benchmark_memberships.py')
SHA256 = re.compile(r'[0-9a-f]{64}')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def object_digest(value: Any) -> str:
    """Same canonical encoding used by the frozen Table6 plan/input IDs."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _path(value: Any, *, directory: bool = False) -> Path:
    path = Path(value)
    require(path.is_absolute(), 'all configured paths must be absolute: ' + str(path))
    require(not path.is_symlink(), 'refuse symlink configured input: ' + str(path))
    path = path.resolve(strict=True)
    require(path.is_dir() if directory else path.is_file(), 'wrong input type: ' + str(path))
    return path


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def file_identity(path: Any, *, allow_empty: bool = False) -> dict[str, Any]:
    """One full sequential hash, protected against changes during the read."""
    source = _path(path)
    before = _stat_identity(source)
    require(allow_empty or before[2] > 0, 'empty immutable input: ' + str(source))
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        opened = os.fstat(handle.fileno())
        require((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) == before,
                'input replaced before hashing: ' + str(source))
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
        after_handle = os.fstat(handle.fileno())
    after = _stat_identity(source)
    require(before == after == (after_handle.st_dev, after_handle.st_ino,
                               after_handle.st_size, after_handle.st_mtime_ns),
            'input changed while hashing: ' + str(source))
    return {'path': str(source), 'size_bytes': before[2], 'mtime_ns': before[3], 'sha256': digest.hexdigest()}


def _read_json(path: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    source = Path(identity['path'])
    data = source.read_bytes()
    require(hashlib.sha256(data).hexdigest() == identity['sha256'], 'JSON changed during read: ' + str(source))
    value = json.loads(data)
    require(isinstance(value, dict), 'JSON object required: ' + str(source))
    return value, identity


def _input_fingerprint(row: dict[str, Any]) -> str:
    return object_digest({'task_kind': row['task_kind'], 'dataset': row['dataset'], 'suite': row['suite'],
                          'format': row['format'], 'input_files': row['input_files'],
                          'target_feature': row.get('target_feature'), 'class_labels': row.get('class_labels')})


def select_regression_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Retain source-plan row order and every exact membership; never filter data."""
    body = {key: value for key, value in plan.items() if key != 'plan_id'}
    require(plan.get('plan_id') == object_digest(body), 'source plan_id mismatch')
    rows = plan.get('rows')
    require(isinstance(rows, list) and len(rows) == 681, 'source must be complete standard681 plan')
    require(not plan.get('data_blocked'), 'source plan has blocked data memberships')
    require(Counter(row.get('task_kind') for row in rows) == {'classification': 457, 'regression': 224},
            'source plan classification/regression counts differ')
    require(Counter(row.get('suite') for row in rows if row.get('task_kind') == 'classification') == CLASS_COUNTS,
            'source standard457 classification membership differs')
    selected = deepcopy([row for row in rows if row['task_kind'] == 'regression'])
    require(Counter(row.get('suite') for row in selected) == REG_COUNTS, 'regression suite counts are not 100/50/33/13/28')
    identities = [row.get('row_id') for row in rows]
    require(all(isinstance(key, str) and key for key in identities) and len(set(identities)) == 681,
            'missing or duplicate source membership row_id')
    require(len({row.get('dataset') for row in selected}) == 224, 'duplicate regression dataset key')
    for row in selected:
        require(isinstance(row.get('dataset'), str) and row['dataset'], 'missing regression dataset')
        require(row['row_id'] == 'regression::' + row['dataset'], 'regression row_id/dataset mismatch')
        require(not row.get('blocked_reason'), 'blocked regression membership: ' + row['row_id'])
        expected_format = 'talent_npy' if row['suite'] == 'talent' else 'bcco_csv' if row['suite'] == 'BCCO' else 'openml_arff'
        require(row.get('format') == expected_format, 'unexpected standard regression format: ' + row['row_id'])
        require(isinstance(row.get('source_path'), str) and Path(row['source_path']).is_absolute(),
                'missing absolute source_path: ' + row['row_id'])
        files = row.get('input_files')
        require(isinstance(files, list) and files, 'missing input_files: ' + row['row_id'])
        paths = []
        for item in files:
            require(isinstance(item, dict) and isinstance(item.get('path'), str) and Path(item['path']).is_absolute(),
                    'invalid frozen input file: ' + row['row_id'])
            require(type(item.get('size_bytes')) is int and item['size_bytes'] > 0
                    and type(item.get('mtime_ns')) is int and item['mtime_ns'] >= 0,
                    'invalid frozen input metadata: ' + item['path'])
            require(not item.get('sha256') or SHA256.fullmatch(item['sha256']) is not None,
                    'invalid frozen input digest: ' + item['path'])
            paths.append(item['path'])
        require(len(set(paths)) == len(paths), 'duplicate input file: ' + row['row_id'])
        require(row.get('input_fingerprint') == _input_fingerprint(row),
                'source input_fingerprint mismatch: ' + row['row_id'])
        if row['suite'] == 'PFN':
            require(row.get('target_feature') and isinstance(row.get('official_split_path'), str)
                    and Path(row['official_split_path']).is_absolute(), 'PFN explicit target/split missing')
            require({row['source_path'], row['official_split_path']} <= set(paths), 'PFN source/split not frozen')
    return selected


def verify_input_metadata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Metadata only; full per-file SHA/array checks belong to eval_data.load."""
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        for expected in row['input_files']:
            path = _path(expected['path'])
            require(str(path) == expected['path'], 'frozen input path no longer canonical: ' + expected['path'])
            previous = records.get(str(path))
            if previous is not None:
                require(previous == expected, 'conflicting repeated input identity: ' + str(path))
                continue
            info = path.stat()
            require((info.st_size, info.st_mtime_ns) == (expected['size_bytes'], expected['mtime_ns']),
                    'frozen input metadata changed: ' + str(path))
            records[str(path)] = deepcopy(expected)
    return records


def loader_identity(eval_data: Any, vendor_dir: Any, vendor_manifest: Any) -> dict[str, Any]:
    loader = file_identity(eval_data)
    tree = ast.parse(Path(loader['path']).read_text())
    protocols = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'PROTOCOL' for target in node.targets)
                 and isinstance(node.value, ast.Constant)]
    require(protocols == [DATA_PROTOCOL] and any(isinstance(node, ast.FunctionDef) and node.name == 'load' for node in tree.body),
            'unexpected audited data-loader API/protocol')
    vendor = _path(vendor_dir, directory=True)
    bindings, receipt = _read_json(vendor_manifest)
    files = {}
    for name in VENDOR_FILES:
        require(name in bindings and isinstance(bindings[name], dict), 'missing vendor binding: ' + name)
        current = file_identity(vendor / name)
        require(current['sha256'] == bindings[name].get('sha256'), 'frozen vendor code changed: ' + name)
        files[name] = current
    return {'protocol': DATA_PROTOCOL, 'eval_data': loader, 'vendor_dir': str(vendor),
            'vendor_files': files, 'vendor_manifest': receipt, 'model_loading': 'none; data helpers only'}


def verify_data_rows(rows: list[dict[str, Any]], loader: dict[str, Any]) -> list[dict[str, Any]]:
    """Use only the audited data helper, never an estimator or Mitra model."""
    vendor = Path(loader['vendor_dir'])
    for filename in VENDOR_FILES:
        name = Path(filename).stem
        previous = sys.modules.get(name)
        if previous is not None:
            require(Path(getattr(previous, '__file__', '')).resolve() == vendor / filename,
                    'already imported a different data helper: ' + name)
    old_path = list(sys.path)
    sys.path.insert(0, str(vendor))
    try:
        spec = importlib.util.spec_from_file_location('_ft50_eval224_audited_data', loader['eval_data']['path'])
        require(spec is not None and spec.loader is not None, 'cannot load data helper')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        audits = []
        for row in rows:
            xs, ys, xt, yt, audit = module.load(row)
            require(audit.get('protocol') == DATA_PROTOCOL and audit.get('protocol_validation') is True
                    and audit.get('row_id') == row['row_id'] and audit.get('task_kind') == 'regression'
                    and audit.get('input_fingerprint') == row['input_fingerprint'], 'data audit identity mismatch')
            require(audit.get('support_subsampling_in_loader') is False and audit.get('full_test_split') is True
                    and audit.get('validation_holdout_in_loader') is False
                    and audit.get('test_labels_used_for_fit_or_routing') is False, 'data loading protocol changed')
            audits.append(audit)
            del xs, ys, xt, yt
        return audits
    finally:
        sys.path[:] = old_path


def build_manifest(*, source_plan: Any, checkpoint_dir: Any, finetune_contract: Any,
                   finetune_complete: Any, eval_data: Any, vendor_dir: Any, vendor_manifest: Any,
                   native_source: Any, inference_protocol: Any, target_transform: Any = None,
                   source_checkpoint: Any = None, verify_data: bool = False) -> dict[str, Any]:
    plan, plan_identity = _read_json(source_plan)
    rows = select_regression_rows(plan)
    input_records = verify_input_metadata(rows)
    contract, contract_identity = _read_json(finetune_contract)
    complete, complete_identity = _read_json(finetune_complete)
    require(contract.get('source_step') == SOURCE_STEP and contract.get('optimizer_updates') == 50
            and contract.get('shared_model') is True and contract.get('inspect_only') is False
            and contract.get('validation_or_test_loaded') is False, 'not the completed train-only shared FT50 contract')
    require(isinstance(contract.get('checkpoint_sha256'), str) and SHA256.fullmatch(contract['checkpoint_sha256']),
            'missing original step22175 checkpoint identity')
    require(complete.get('complete') is True and complete.get('source_step') == SOURCE_STEP
            and complete.get('optimizer_updates') == 50 and complete.get('checkpoint_count') == 50
            and complete.get('final_step') == 22225, 'incomplete or different finetuning run')
    directory = _path(checkpoint_dir, directory=True)
    names = {f'step-{step}.ckpt' for step in STEPS}
    require({path.name for path in directory.glob('step-*.ckpt')} == names, 'checkpoint set must be exactly step22176:22225')
    require(set(complete.get('checkpoint_sizes', {})) == names, 'completion checkpoint membership mismatch')
    # Contract/complete must be from this same run, not an unrelated local copy.
    require(Path(contract_identity['path']).parent == directory and Path(complete_identity['path']).parent == directory,
            'finetune contract/complete must live alongside the 50 checkpoint files')
    checkpoints = []
    for step in STEPS:
        identity = file_identity(directory / f'step-{step}.ckpt')
        require(identity['size_bytes'] == complete['checkpoint_sizes'][f'step-{step}.ckpt'],
                'checkpoint size differs from completed training receipt')
        checkpoints.append({'step': step, 'source_step': SOURCE_STEP, 'finetune_step': step - SOURCE_STEP, 'kind': 'finetuned',
                            **identity, 'training_contract_sha256': contract_identity['sha256'],
                            'source_checkpoint_sha256': contract['checkpoint_sha256'],
                            'tensor_validation': 'worker must strict-load and check checkpoint metadata before evaluation'})
    if source_checkpoint is not None:
        identity = file_identity(source_checkpoint)
        require(identity['sha256'] == contract['checkpoint_sha256'], 'optional baseline differs from the true step22175 finetuning source')
        checkpoints.insert(0, {'step': SOURCE_STEP, 'source_step': SOURCE_STEP, 'finetune_step': 0,
                               'kind': 'source_baseline', **identity,
                               'source_checkpoint_sha256': contract['checkpoint_sha256'],
                               'tensor_validation': 'worker must strict-load and verify original curr_step22175'})
    loader = loader_identity(eval_data, vendor_dir, vendor_manifest)
    transform = (file_identity(target_transform) if target_transform is not None
                 else deepcopy(loader['vendor_files']['official_talent_regression_worker.py']))
    transform_ast = ast.parse(Path(transform['path']).read_text())
    require(any(isinstance(node, ast.ClassDef) and node.name == 'RegressionTargetTransform' for node in transform_ast.body),
            'target-transform helper must expose RegressionTargetTransform')
    helpers = {'eval_data': loader['eval_data'], **loader['vendor_files'], 'target_transform': transform}
    native = _path(native_source, directory=True)
    require((native / 'tabicl').is_dir(), '--native-source must be the frozen src directory containing tabicl')
    code_files = {str(path.relative_to(native)): file_identity(path, allow_empty=True) for path in sorted(native.rglob('*.py'))}
    require(code_files, 'native source contains no Python files')
    inference, inference_identity = _read_json(inference_protocol)
    require(inference, 'inference protocol cannot be empty')
    protocol = {'data_loader_protocol': DATA_PROTOCOL,
                'support': 'TALENT official train+val; other suites unchanged official TRAIN order',
                'pfn_split': 'explicit official repeat=0/fold=0',
                'model_input': 'official raw DataFrames; audited canonicalize_features; native TabICL preprocessing',
                'feature_encoding': 'canonicalize raw DataFrame then native TabICL encoder; preserve canonical column order',
                'diagnostic_numeric_loader': 'optional eval_data.load output is diagnostic only, never supplied as model input',
                'target_units': 'original units; support-only external_gt_aware fitted by the frozen transform helper and inverted before metrics',
                'target_transform_helper': transform,
                'test': 'all official TEST rows; never used for fitting/routing/checkpoint selection',
                'membership_policy': 'all224 retained, including tasks seen during train-only finetuning',
                'checkpoint_policy': 'all50 reported; no test-driven best-checkpoint selection',
                'inference': inference, 'data_loader_code': loader,
                'native_source_sha256': object_digest({name: record['sha256'] for name, record in code_files.items()})}
    fingerprint = object_digest(protocol)
    audits = verify_data_rows(rows, loader) if verify_data else None
    for index, row in enumerate(rows):
        row['dataset_index'] = index
        row['protocol_fingerprint'] = object_digest({'protocol_fingerprint': fingerprint,
                                                     'row_id': row['row_id'], 'input_fingerprint': row['input_fingerprint']})
        if audits is not None:
            row['diagnostic_data_audit'] = audits[index]
    # Recheck cheap identities after all long hashes/data validation. Never hash
    # all checkpoint bytes twice or create an 11,200-row dummy result matrix.
    require(verify_input_metadata(rows) == input_records, 'inputs changed during preparation')
    for record in checkpoints:
        info = Path(record['path']).stat()
        require((info.st_size, info.st_mtime_ns) == (record['size_bytes'], record['mtime_ns']),
                'checkpoint changed after hashing: ' + record['path'])
    manifest = {'schema_version': 1, 'contract': CONTRACT, 'task_kind': 'regression', 'inference_loop': 3,
                'membership_count': 224, 'checkpoint_count': len(checkpoints),
                'finetuned_checkpoint_count': 50, 'baseline_checkpoint_count': int(source_checkpoint is not None),
                'evaluation_unit_count': 224 * len(checkpoints), 'finetuned_evaluation_unit_count': 11200,
                'suite_counts': REG_COUNTS, 'source_step': SOURCE_STEP,
                'source_plan': {**plan_identity, 'plan_id': plan['plan_id']},
                'rows': rows, 'checkpoints': checkpoints, 'checkpoint_directory': str(directory),
                'finetune_contract': contract_identity, 'finetune_complete': complete_identity,
                'inference_protocol_file': inference_identity, 'data_loader': loader, 'helpers': helpers,
                'native_source': {'path': str(native), 'files': code_files,
                                  'sha256': protocol['native_source_sha256']},
                'protocol': protocol, 'protocol_fingerprint': fingerprint,
                'data_validation': ('all224_diagnostic_cpu_load_passed; worker still validates native raw-frame protocol' if verify_data
                                    else 'metadata_verified; native raw-frame validation deferred to worker'),
                'membership_sha256': object_digest([{'row_id': r['row_id'], 'dataset': r['dataset'], 'suite': r['suite'],
                                                     'input_fingerprint': r['input_fingerprint']} for r in rows]),
                'checkpoint_hash_policy': 'one full SHA256 per file at prepare; workers verify only their selected checkpoint',
                'data_files_copied': False, 'models_loaded_at_prepare': False, 'results_created_at_prepare': 0}
    manifest['manifest_id'] = object_digest(manifest)
    return manifest


def publish_manifest(path: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    """Atomic no-clobber publication; an existing manifest is never refreshed."""
    destination = Path(path)
    require(destination.is_absolute(), 'output path must be absolute')
    require(not destination.exists() and not destination.is_symlink(), 'manifest already exists; never overwrite')
    require(manifest.get('manifest_id') == object_digest({k: v for k, v in manifest.items() if k != 'manifest_id'}),
            'manifest content identity mismatch before publication')
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix='.' + destination.name + '.', dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)  # Atomic and fails if another preparer published first.
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return {'manifest': str(destination.resolve()), 'manifest_id': manifest['manifest_id'],
            'manifest_sha256': hashlib.sha256(data).hexdigest(), 'membership_count': 224,
            'checkpoint_count': manifest['checkpoint_count'], 'evaluation_unit_count': manifest['evaluation_unit_count'],
            'finetuned_checkpoint_count': 50, 'baseline_checkpoint_count': manifest['baseline_checkpoint_count'],
            'data_validation': manifest['data_validation'], 'models_loaded': False}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-plan', 'checkpoint-dir', 'finetune-contract', 'finetune-complete',
                 'eval-data', 'vendor-dir', 'vendor-manifest', 'native-source', 'inference-protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--target-transform', type=Path,
                        help='optional explicit helper; defaults to the frozen vendor official_talent_regression_worker.py')
    parser.add_argument('--source-checkpoint', type=Path, help='optional exact original step22175 same-protocol baseline')
    parser.add_argument('--verify-data', action='store_true', help='CPU-only eval_data.load on all224; no estimator instantiated')
    args = vars(parser.parse_args(argv))
    output = args.pop('output')
    require(not output.exists() and not output.is_symlink(), 'manifest already exists; do not hash/checkpoint-refresh this run')
    result = publish_manifest(output, build_manifest(**args))
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
