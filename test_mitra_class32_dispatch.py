"""CPU-only launcher contract tests; all generated fixtures live in temp dirs."""
import contextlib
import copy
import io
import json
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mitra_class32_dispatch as dispatch


class DispatcherTests(unittest.TestCase):
    def test_child_deadline_and_parent_remaining_time(self):
        self.assertEqual(dispatch.runtime_limits(2)['soft_step_limit_seconds'], 6900)
        full = dispatch.runtime_limits(12)
        self.assertEqual(full['hard_step_limit'], '12:00:00')
        self.assertEqual(full['single_dataset_limit_seconds'], 21600)
        self.assertEqual(full['soft_step_limit_seconds'], 42900)
        now = 1700000000
        end = datetime.fromtimestamp(now + 44000).isoformat()
        dispatch.check_remaining_time({'EndTime': end}, 43200, now)
        with self.assertRaises(RuntimeError):
            dispatch.check_remaining_time({'EndTime': end}, 44000, now)
        with self.assertRaises(RuntimeError):
            dispatch.check_remaining_time({'EndTime': 'Unknown'}, 43200, now)
        for hours in (0, 73, 1.5):
            with self.assertRaises(RuntimeError):
                dispatch.runtime_limits(hours)

    def test_assignment_all_parent_counts_and_smoke(self):
        with contextlib.redirect_stdout(io.StringIO()):
            dispatch.self_test()

    def test_file_identity_rejects_changed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'
            path.write_bytes(b'original')
            frozen = dispatch.file_identity(path)
            dispatch.check_identity(frozen)
            path.write_bytes(b'changed!')
            with self.assertRaises(RuntimeError):
                dispatch.check_identity(frozen)

    def test_prepare_freezes457_readonly_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            benchmark, stage = root / 'benchmark', root / 'stage'
            (benchmark / 'data').mkdir(parents=True)
            (benchmark / 'cache').mkdir()
            stage.mkdir()
            stack.enter_context(patch.object(dispatch, 'BENCHMARK', benchmark))
            stack.enter_context(patch.object(dispatch, 'MITRA', stage))
            rows = []
            for suite, count in dispatch.COUNTS.items():
                for local_index in range(count):
                    index = len(rows)
                    dataset = f'{suite}__dataset{local_index}'
                    (benchmark / 'data' / dataset).mkdir()
                    cache = dispatch.cache_path(dataset)
                    # Manifest freezing only hashes bytes: NumPy/model loading is
                    # the separately tested worker's responsibility.
                    cache.write_bytes(f'cache fixture {index}'.encode())
                    rows.append({'position': index, 'dataset': dataset, 'suite': suite,
                                 'cache_path': str(cache), 'finite': True, 'classes': [2, 5, 12][index % 3],
                                 'train_rows': 100, 'test_rows': 20, 'features': 4,
                                 'split': 'frozen fixture split'})
            source = {'complete': True, 'protocol_validation': True, 'classification_memberships': 457,
                      'suite_counts': dispatch.COUNTS, 'rows': rows}
            benchmark_file = benchmark / 'benchmark_manifest.json'
            benchmark_file.write_text(json.dumps(source))
            weights = stage / 'weights_manifest.json'
            weights.write_text(json.dumps({'classifier': {'repo_id': 'autogluon/mitra-classifier',
                                                          'autogluon_version': '1.5.0', 'sha256': 'fake-weights'}}))
            for script in dispatch.STAGE_SCRIPTS:
                (stage / script).write_text('# fixture ' + script)
            stack.enter_context(patch.object(dispatch, 'BENCHMARK_MANIFEST_SHA', dispatch.file_sha(benchmark_file)))
            stack.enter_context(patch.object(dispatch, 'WEIGHTS_MANIFEST_SHA', dispatch.file_sha(weights)))
            stack.enter_context(patch.object(dispatch, 'STAGE_SHA', {
                script: dispatch.file_sha(stage / script) for script in dispatch.STAGE_SCRIPTS}))
            originals = [dispatch.file_identity(row['cache_path']) for row in rows]
            output = root / 'manifest.json'
            with contextlib.redirect_stdout(io.StringIO()):
                dispatch.prepare(output)
            frozen = dispatch.validate_manifest(dispatch.read(output))
            self.assertEqual(len(frozen['rows']), 457)
            self.assertEqual(frozen['suite_counts'], dispatch.COUNTS)
            self.assertEqual(frozen['n_estimators'], 32)
            self.assertEqual(frozen['seed'], 0)
            self.assertEqual([row['cache'] for row in frozen['rows']], originals)
            self.assertEqual([dispatch.file_identity(row['cache_path']) for row in rows], originals)
            snapshot = output.read_bytes()
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                dispatch.prepare(output)
            self.assertEqual(output.read_bytes(), snapshot)
            tampered = copy.deepcopy(frozen)
            tampered['rows'][0]['cache']['sha256'] = 'tampered'
            tampered['manifest_id'] = dispatch.digest({k: v for k, v in tampered.items() if k != 'manifest_id'})
            with self.assertRaises(RuntimeError):
                dispatch.validate_manifest(tampered)

    def test_actual32_result_validation_and_rejection(self):
        row = {'dataset_index': 234, 'dataset': 'binary', 'suite': 'BCCO',
               'input_fingerprint': 'inputs', 'test_rows': 56}
        manifest = {'manifest_id': 'manifest'}
        plan = {'worker_sha256': {'mitra_class32_one.py': 'worker'}, 'model_weight_sha256': 'weights'}
        audit = {'actual32_verified': True, 'actual_ensemble_count': 32, 'member_audits': list(range(32)),
                 'all_test_rows_covered': True, 'minimum_contributions_per_test_row': 32,
                 'maximum_contributions_per_test_row': 32}
        result = {**row, 'complete': True, 'model_name': 'mitra', 'task_kind': 'classification',
                  'manifest_id': 'manifest', 'worker_source_sha256': 'worker', 'model_sha256': 'weights',
                  'actual32_verified': True, 'n_estimators': 32, 'actual_ensemble_count': 32,
                  'node_count': 1, 'ensemble_audits': [audit], 'accuracy': 0.8, 'full_test_split': True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_text(json.dumps(result))
            dispatch.validate_result(path, row, manifest, plan)
            bad_cases = []
            for key, value in [('n_estimators', 8), ('model_sha256', 'other'), ('input_fingerprint', 'other'),
                               ('actual32_verified', False), ('test_rows', 55), ('accuracy', float('nan'))]:
                bad = copy.deepcopy(result); bad[key] = value; bad_cases.append(bad)
            bad = copy.deepcopy(result)
            bad['ensemble_audits'][0]['member_audits'].pop()
            bad_cases.append(bad)
            bad = copy.deepcopy(result)
            bad['ensemble_audits'][0]['minimum_contributions_per_test_row'] = 31
            bad_cases.append(bad)
            for bad in bad_cases:
                path.write_text(json.dumps(bad))
                with self.assertRaises(RuntimeError):
                    dispatch.validate_result(path, row, manifest, plan)


if __name__ == '__main__':
    unittest.main()
