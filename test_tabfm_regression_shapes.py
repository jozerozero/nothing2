import copy
import json
from pathlib import Path
import tempfile
import unittest

import tabfm_regression_shapes as shapes


class ShapeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        self.path = root / 'manifest.json'
        self.receipts = root / 'results/step-22175'
        self.receipts.mkdir(parents=True)
        self.checkpoint = {'step': 22175, 'source_step': 22175, 'kind': 'source_baseline',
                           'finetune_step': 0, 'sha256': 'original'}
        rows = [dict(dataset_index=i, dataset=f'data{i}', input_fingerprint=f'input{i}') for i in range(224)]
        data = {'rows': rows, 'checkpoints': [self.checkpoint]}
        data['manifest_id'] = shapes.frozen.digest(data)
        self.data = data
        self.path.write_text(json.dumps(data))
        self.man = {'regression_manifest': shapes._read(self.path)[1]}
        self.tasks = [dict(task_kind='regression', dataset_index=i, dataset=r['dataset'],
                           data_manifest_id=data['manifest_id'], row=copy.deepcopy(r)) for i, r in enumerate(rows)]
        self.result = {'complete': True, 'checkpoint_step': 22175, 'checkpoint': self.checkpoint,
            'strict_checkpoint_load': True, 'checkpoint_load_weights_only': True, 'task_kind': 'regression',
            'manifest_id': data['manifest_id'], 'dataset_index': 0, 'dataset': 'data0', 'input_fingerprint': 'input0',
            'data_audit': {'input_fingerprint': 'input0', 'support_rows': 100, 'test_rows': 20, 'features': 5,
                'support_subsampling': False, 'query_chunking': False, 'test_rows_filtered': 0,
                'test_targets_masked_or_imputed': False, 'support_row_order': 'unchanged official order',
                'test_row_order': 'unchanged official order'}}
        self.write_result()

    def write_result(self):
        (self.receipts / 'row-000.json').write_text(json.dumps(self.result))

    def test_exact_shapes_and_missing_audit_no_task_mutation(self):
        before = copy.deepcopy(self.tasks)
        overlay = shapes.build(self.man, self.tasks)
        self.assertEqual(len(overlay['skipped']), 223)
        verified = shapes.validate(overlay, self.man, self.tasks)
        self.assertEqual(verified['0']['train_rows'], 100)
        self.assertEqual(verified['0']['test_rows'], 20)
        self.assertEqual(verified['0']['features'], 5)
        self.assertEqual(self.tasks, before)
        self.assertNotIn('train_rows', self.tasks[0]['row'])

    def test_non_original_checkpoint_skipped(self):
        self.result['checkpoint'] = dict(self.checkpoint, finetune_step=50)
        self.write_result()
        overlay = shapes.build(self.man, self.tasks)
        self.assertEqual(overlay['rows'], {})
        self.assertIn('exact original22175', overlay['skipped'][0]['reason'])

    def test_non_full_test_or_support_skipped(self):
        for field, value in [('support_subsampling', True), ('query_chunking', True),
                             ('test_rows_filtered', 1), ('test_targets_masked_or_imputed', True)]:
            with self.subTest(field=field):
                previous = self.result['data_audit'][field]
                self.result['data_audit'][field] = value
                self.write_result()
                self.assertEqual(shapes.build(self.man, self.tasks)['rows'], {})
                self.result['data_audit'][field] = previous

    def test_wrong_fingerprint_or_nonpositive_shape_skipped(self):
        self.result['input_fingerprint'] = 'different'
        self.write_result()
        self.assertFalse(shapes.build(self.man, self.tasks)['rows'])
        self.result['input_fingerprint'] = 'input0'
        for value in (0, -1, 1.5, True):
            self.result['data_audit']['features'] = value
            self.write_result()
            self.assertFalse(shapes.build(self.man, self.tasks)['rows'])

    def test_changed_pinned_receipt_fails_closed(self):
        overlay = shapes.build(self.man, self.tasks)
        self.result['data_audit']['support_rows'] = 999
        self.write_result()
        with self.assertRaisesRegex(RuntimeError, 'Pinned shape receipt changed'):
            shapes.validate(overlay, self.man, self.tasks)

    def test_tampered_shape_or_missing_skip_fails_closed(self):
        overlay = shapes.build(self.man, self.tasks)
        overlay['rows']['0']['features'] = 4
        with self.assertRaises(RuntimeError): shapes.validate(overlay, self.man, self.tasks)
        overlay = shapes.build(self.man, self.tasks)
        overlay['skipped'].pop()
        with self.assertRaises(RuntimeError): shapes.validate(overlay, self.man, self.tasks)

    def test_augmented_runtime_task_rows_rejected(self):
        self.tasks[0]['row']['train_rows'] = 100
        with self.assertRaisesRegex(RuntimeError, 'unchanged original task rows'):
            shapes.build(self.man, self.tasks)


if __name__ == '__main__':
    unittest.main()
