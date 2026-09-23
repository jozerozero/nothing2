#!/usr/bin/env python3
"""Evaluate only BNG(mv), extending historical coverage_v1 by one membership.

Calls the frozen legacy v1 TALENT runner unchanged: seed0, one seed, no HPO,
20 maximum epochs and default model configuration. BNG's 50,388 train rows
exceed the former 50,000-row eligibility filter; no input row is removed.
All outputs go to the new requested destination, never the legacy campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import tempfile
import time
import traceback

LEGACY_HASHES = {
    'run_worker.py': 'e87fb7189553212a04e2f67ef4725b2f6e6241a9b40749d4d4ea7758fce7a69f',
    'table6_data.py': '93f87227f920e1cb1b1153cea736d5251901e03290fc7f175525292b11844b5d',
}
KEY = 'talent__BNG_mv'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def snapshot(records):
    found = []
    for record in records:
        p = Path(record['path'])
        before = p.stat()
        require((before.st_size, before.st_mtime_ns) ==
                (record['size_bytes'], record['mtime_ns']), 'Canonical input metadata changed: '+str(p))
        digest = sha(p)
        require(not record.get('sha256') or digest == record['sha256'], 'Canonical input hash changed')
        after = p.stat()
        require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_ino, after.st_size, after.st_mtime_ns), 'Input changed during read')
        found.append(dict(record, sha256=digest))
    return found


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--legacy-stage', type=Path, required=True)
    p.add_argument('--canonical-manifest', type=Path, required=True)
    p.add_argument('--reference-result', type=Path, required=True,
                   help='Complete native TabFM or Taffy row009 with canonical target hashes')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--inspect-only', action='store_true')
    a = p.parse_args()
    require(not a.output.exists(), 'New result destination must not exist')
    stage = a.legacy_stage.resolve(strict=True)
    for name, digest in LEGACY_HASHES.items():
        require(sha(stage/name) == digest, 'Frozen historical source differs: '+name)
    manifest = read(a.canonical_manifest)
    body = {k:v for k,v in manifest.items() if k != 'manifest_id'}
    require(manifest['manifest_id'] == hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            'Canonical manifest self hash mismatch')
    rows = [r for r in manifest['rows'] if r['dataset'] == KEY]
    require(len(rows) == 1 and rows[0]['original_dataset'] == 'BNG(mv)', 'Expected unique canonical BNG membership')
    row = rows[0]
    require(row['format'] == 'talent_npy' and row['task_kind'] == 'regression', 'Wrong input format/task')
    files = snapshot(row['input_files'])
    reference = read(a.reference_result)
    require(reference.get('complete') is True and reference['dataset'] == KEY, 'Reference must be complete canonical BNG result')
    ref_audit = reference['data_audit']
    sys.path.insert(0, str(stage/'TALENT'))
    sys.path.insert(0, str(stage))
    legacy = importlib.import_module('run_worker')
    data = importlib.import_module('table6_data')
    require(Path(legacy.__file__).resolve() == stage/'run_worker.py' and
            Path(data.__file__).resolve() == stage/'table6_data.py', 'Wrong legacy module import')
    import numpy as np
    frames, labels, split_audit = data.read_talent(Path(row['source_path']))
    labels, task_info = data.process_labels(labels, 'regression')
    numeric, categorical, features = data.infer_feature_blocks(frames)
    require(tuple(len(labels[s]) for s in ('train','val','test')) == (50388,12597,15747), 'BNG full split dimensions drifted')
    for s in ('train','val','test'):
        require(len(frames[s]) == len(labels[s]) and frames[s].shape[1] == 10, 'BNG features/labels drifted')
    target_hashes = {
        'support_targets_sha256': hashlib.sha256(np.concatenate([labels['train'],labels['val']]).astype('<f8').tobytes()).hexdigest(),
        'test_targets_sha256': hashlib.sha256(labels['test'].astype('<f8').tobytes()).hexdigest(),
    }
    for name, value in target_hashes.items():
        require(ref_audit[name] == value, 'Canonical reference target ordering/values mismatch: '+name)
    require(ref_audit['support_rows'] == 62985 and ref_audit['test_rows'] == 15747,
            'Reference full support/test dimensions mismatch')
    info = dict(name='TALENT-REG__BNG(mv)', **task_info,
                n_num_features=0 if numeric is None else numeric['train'].shape[1],
                n_cat_features=0 if categorical is None else categorical['train'].shape[1],
                train_size=50388, val_size=12597, test_size=15747)
    def select(block, splits):
        return None if block is None else {s:block[s] for s in splits}
    audit = dict(split=split_audit, features=features, info=info,
                 source_files=files, source_path=row['source_path'], **target_hashes,
                 support_rows=62985, test_rows=15747, full_test_split=True,
                 row_subsampling=False, original_training_filter_extended_for_single_membership=True)
    bundle = data.DatasetBundle(
        train_val_data=(select(numeric, ('train','val')), select(categorical, ('train','val')), select(labels, ('train','val'))),
        test_data=(select(numeric, ('test',)), select(categorical, ('test',)), select(labels, ('test',))),
        info=info, frames=frames, labels=labels, audit=audit)
    method = dict(classification=True, group='deep', paper_name='SwitchTab', regression=True,
                  runner='talent', talent_name='switchtab')
    result = dict(complete=False, protocol='coverage_v1_standard224_membership_extension',
                  budget_protocol='coverage_v1', dataset=KEY, original_dataset='BNG(mv)',
                  dataset_index=row['dataset_index'], row_id='TALENT-REG__BNG(mv)',
                  method='SwitchTab', method_record=method, task_kind='regression', suite='talent',
                  benchmark='TALENT-REG', seed=0, seed_num=1, n_trials=0, max_epochs=20,
                  legacy_sources={str(stage/k):v for k,v in LEGACY_HASHES.items()},
                  canonical_manifest_path=str(a.canonical_manifest), canonical_manifest_id=manifest['manifest_id'],
                  canonical_manifest_sha256=sha(a.canonical_manifest),
                  canonical_reference_path=str(a.reference_result), canonical_reference_sha256=sha(a.reference_result),
                  data_audit=audit, hostname=socket.gethostname(), job_id=os.getenv('SLURM_JOB_ID'),
                  protocol_extension='One new BNG(mv) membership with 50,388 training rows; old 50,000-row eligibility filter does not apply; unchanged complete published train/val/test; original campaign untouched')
    if a.inspect_only:
        print(json.dumps(dict(status='input_preflight_pass', **result), sort_keys=True)); return
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one allocated GPU required')
    torch.cuda.set_device(0)
    torch.set_num_threads(8)
    require(os.getenv('SLURM_JOB_ID'), 'Run under a genuine allocated Slurm job')
    import TALENT
    talent_root = Path(TALENT.__file__).resolve().parent
    require(talent_root.is_relative_to(stage), 'Use the historical TALENT checkout')
    result['talent_source_snapshot'] = {str(p.relative_to(talent_root)):sha(p)
                                      for p in sorted(talent_root.rglob('*'))
                                      if p.is_file() and p.suffix in ('.py','.json')}
    args = argparse.Namespace(seed=0, seed_num=1, n_trials=0, max_epochs=20, cpus_per_task=8,
                              fixed_batch_size=0, force_float=False, saint_dim=0)
    started = time.time()
    scratch = Path(tempfile.mkdtemp(prefix='switchtab-bng-', dir=os.getenv('SLURM_TMPDIR') or '/tmp'))
    signal.signal(signal.SIGALRM, legacy.timeout_handler)
    signal.alarm(1800)
    try:
        metrics, implementation = legacy.run_talent(method, bundle, scratch, args)
        require(all(math.isfinite(float(metrics[k])) for k in ('rmse','mae','r2')), 'Nonfinite result')
        require(snapshot(row['input_files']) == files, 'Input files changed while fitting')
        result.update(complete=True, status='complete', metrics=metrics, implementation=implementation,
                      elapsed_seconds=time.time()-started)
        publish(a.output, result)
        print(json.dumps(dict(status='complete', output=str(a.output), metrics=metrics), sort_keys=True))
    except Exception as exc:
        result.update(status='error', error=repr(exc), traceback=traceback.format_exc(), elapsed_seconds=time.time()-started)
        publish(a.output.with_suffix('.error.json'), result)
        raise
    finally:
        signal.alarm(0)
        shutil.rmtree(scratch)


if __name__ == '__main__':
    main()
