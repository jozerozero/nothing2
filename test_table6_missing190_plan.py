"""Stdlib CPU tests; synthetic metadata, no datasets, model imports or remote IO."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import table6_missing190_plan as builder


def fixtures():
    fixed_rows, r19_rows = [], []
    index = 0
    for suite, count in builder.SUITES.items():
        for offset in range(count):
            dataset = f'{suite}__task-{offset:03d}'
            source, cache = f'/source/{dataset}', f'/cache/{dataset}.npz'
            fixed = {'dataset': dataset, 'suite': suite, 'source_name': f'task-{offset:03d}',
                     'source_path': source, 'cache_path': cache, 'train_rows': 80,
                     'test_rows': 20, 'features': 3, 'classes': 2, 'class_labels': ['0', '1'],
                     'position': index, 'split': 'original official train/test'}
            row = {**fixed, 'task_kind': 'classification', 'row_id': 'classification::'+dataset,
                   'format': 'talent_npy', 'membership_source': 'standard457_classification',
                   'work_size': 240, 'input_files': [
                       {'path': cache, 'size_bytes': 1024, 'mtime_ns': 123},
                       {'path': source+'/y_test.npy', 'size_bytes': 256, 'mtime_ns': 124}]}
            row['input_fingerprint'] = builder.input_fingerprint(row)
            fixed_rows.append(fixed); r19_rows.append(row); index += 1
    blocked = [{'method': method, 'dataset': row['dataset'], 'reason': builder.BLOCK_REASON}
               for method in builder.METHODS for row in fixed_rows[:22 if method == 'TabR' else 21]]
    blocked_ids = {(p['method'], p['dataset']) for p in blocked}
    ready = [{'method': method, 'dataset': row['dataset']}
             for method in (*builder.METHODS, 'AutoGluon') for row in fixed_rows
             if (method, row['dataset']) not in blocked_ids]
    fixed = {'rows': fixed_rows, 'pairs': ready, 'blocked': blocked, 'seeds': list(builder.SEEDS),
             'classification_memberships': 457, 'hpo_trials': 0, 'benchmark_manifest_sha256': 'a'*64}
    r19 = {'rows': r19_rows+[{'dataset': f'reg-{i}', 'task_kind': 'regression'} for i in range(224)],
           'pairs': [{'key': str(i)} for i in range(12482)], 'blocked': [], 'seeds': list(builder.SEEDS),
           'classification_memberships': 457, 'regression_memberships': 224, 'hpo_trials': 100,
           'validation': 'original validation or seed0 train-only holdout',
           'source_contract': {'classification_manifest': {'sha256': 'a'*64, 'snapshot': '/frozen/classification.json'},
                               'standard_loader': {'sha256': 'b'*64, 'snapshot': '/frozen/standard_loader.py'}}}
    return fixed, r19


class Missing190Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='missing190-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root/'table6_missing190_fixture'
        self.sources = {'fixedbest_plan': {'path': str(self.root/'fixed/plan.json'), 'sha256': 'c'*64, 'size_bytes': 10},
                        'remaining19_plan': {'path': str(self.root/'r19/plan.json'), 'sha256': 'd'*64, 'size_bytes': 11}}
        self.fixed, self.r19 = fixtures()

    def build(self):
        fixed, r19 = deepcopy(self.fixed), deepcopy(self.r19)
        for value in (fixed, r19):
            value.pop('plan_id', None)
            value['plan_id'] = builder.object_digest(value)
        with patch.object(builder, 'FIXED_PLAN_ID', fixed['plan_id']), patch.object(builder, 'R19_PLAN_ID', r19['plan_id']):
            return builder.build_documents(fixed, r19, self.output, self.sources)

    def test_exact190_independent_protocol_and_untouched_rows(self):
        before = deepcopy((self.fixed, self.r19))
        plan = self.build()
        self.assertEqual(len(plan['pairs']), 190)
        self.assertEqual(len(plan['rows']), 22)
        self.assertEqual(plan['pair_counts_by_method'], {m: 22 if m == 'TabR' else 21 for m in builder.METHODS})
        self.assertEqual(plan['source_classification_memberships'], 457)
        self.assertTrue(all(pair['hpo_trials'] == 100 and pair['seeds'] == list(range(15)) for pair in plan['pairs']))
        self.assertFalse(plan['test_evaluated_during_hpo'])
        self.assertEqual((self.fixed, self.r19), before)
        originals = {row['dataset']: row for row in self.r19['rows']}
        self.assertTrue(all(row == originals[row['dataset']] for row in plan['rows']))
        self.assertEqual(plan, self.build())
        payload = dict(plan); expected = payload.pop('plan_id')
        self.assertEqual(builder.object_digest(payload), expected)

    def test_parent_sha_drift_rejected_before_construction(self):
        with self.assertRaisesRegex(ValueError, 'plan_id'):
            builder.build_documents(self.fixed, self.r19, self.output, self.sources)

    def test_parent_manifest_hash_conflict(self):
        self.fixed['benchmark_manifest_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'manifest file SHA differs'):
            self.build()

    def test_fixed_row_and_r19_field_conflict(self):
        self.fixed['rows'][0]['cache_path'] = '/different/cache.npz'
        with self.assertRaisesRegex(ValueError, 'row conflict'):
            self.build()

    def test_input_fingerprint_is_recomputed(self):
        self.r19['rows'][0]['input_files'][0]['size_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'input_fingerprint'):
            self.build()

    def test_missing_cache_pin_is_rejected(self):
        row = self.r19['rows'][0]
        row['input_files'][0]['path'] = '/another/file.npz'
        row['input_fingerprint'] = builder.input_fingerprint(row)
        with self.assertRaisesRegex(ValueError, 'cache is not pinned'):
            self.build()

    def test_duplicate_blocked_pair_rejected(self):
        self.fixed['blocked'][-1] = deepcopy(self.fixed['blocked'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate blocked'):
            self.build()

    def test_unexpected_method_rejected(self):
        self.fixed['blocked'][0]['method'] = 'AutoGluon'
        with self.assertRaisesRegex(ValueError, 'unexpected/missing method'):
            self.build()

    def test_unknown_reason_rejected(self):
        self.fixed['blocked'][0]['reason'] = 'retry existing failed model with different parameters'
        with self.assertRaisesRegex(ValueError, 'blocking reason'):
            self.build()

    def test_existing_ready_pair_never_recomputed(self):
        self.fixed['pairs'][0] = {key: self.fixed['blocked'][0][key] for key in ('method', 'dataset')}
        with self.assertRaisesRegex(ValueError, 'ready/blocked conflict'):
            self.build()

    def test_output_namespaces_do_not_overlap_frozen_data(self):
        self.output = self.root/'fixed/table6_missing190_bad'
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            self.build()
        self.output = self.root/'unscoped'
        with self.assertRaisesRegex(ValueError, 'namespace'):
            self.build()

    def test_existing_nonempty_output_rejected(self):
        self.output.mkdir()
        (self.output/'existing-result').write_text('preserve me')
        with self.assertRaisesRegex(ValueError, 'nonempty output'):
            self.build()
        self.assertEqual((self.output/'existing-result').read_text(), 'preserve me')

    def test_exclusive_atomic_publish_and_build_file_api(self):
        for value, label in ((self.fixed, 'fixedbest_plan'), (self.r19, 'remaining19_plan')):
            value['plan_id'] = builder.object_digest(value)
            path = Path(self.sources[label]['path']); path.parent.mkdir()
            path.write_text(json.dumps(value))
        with patch.object(builder, 'FIXED_PLAN_ID', self.fixed['plan_id']), patch.object(builder, 'R19_PLAN_ID', self.r19['plan_id']):
            plan = builder.build(self.sources['fixedbest_plan']['path'], self.sources['remaining19_plan']['path'], self.output)
        for label, record in plan['source_contract']['parent_files'].items():
            self.assertEqual(record['sha256'], hashlib.sha256(Path(record['path']).read_bytes()).hexdigest())
        destination = self.output/'plan.json'
        builder.publish_new(destination, plan)
        before = destination.read_bytes()
        with self.assertRaises(ValueError):
            builder.publish_new(destination, plan)
        self.assertEqual(destination.read_bytes(), before)
        self.assertEqual(list(self.output.iterdir()), [destination])


if __name__ == '__main__':
    unittest.main()
