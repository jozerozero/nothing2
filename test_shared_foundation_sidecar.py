"""CPU contracts only: no remote, GPU, Slurm submission, or real model runs."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import shared_foundation_sidecar as lane


class SidecarTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='foundation-sidecar-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.man = {'manifest_id': 'scientific-manifest', 'output_root': str(self.root/'canonical')}
        self.task = {'task_kind': 'regression', 'dataset_index': 3, 'dataset': 'fixture',
                     'row': {'train_rows': 80, 'test_rows': 20, 'features': 7}}
        self.owner = {'job': '123', 'rank': 0, 'node': 'node', 'uuid': '1111222233334444', 'pci': '0000:03:00.0'}
        self.plan = {'plan_id': 'runtime-plan'}
        self.budget = types.SimpleNamespace(remaining=lambda: 6000)
        self.sidecar = Path(self.man['output_root'])/'sidecars/new-lane'
        self.memory = {'own_tree_rss_bytes': lane.GIB, 'same_uid_rss_bytes': 19*lane.GIB,
                       'other_same_uid_rss_bytes': 18*lane.GIB}
        lane.STOP = None

    def run_attempt(self, launch, snapshots=None, cleanup=None, smoke=False):
        with patch.object(lane, 'snapshot', side_effect=snapshots or [deepcopy(self.memory)]*20), \
             patch.object(lane, 'cleanup_children', return_value=cleanup or {'all_owned_children_reaped': True, 'epoch': 1}), \
             patch.object(lane.q, 'launch', side_effect=launch), \
             patch.object(lane.q, 'valid_result', side_effect=lambda path, man, task: lane.q.read(path)):
            return lane.attempt(self.plan, self.root/'frozen-manifest.json', self.man,
                                self.task, self.owner, self.budget, self.sidecar, smoke=smoke)

    def write_result(self, man, task, owner, success, smoke=False):
        path = lane.q.task_path(man, 'smoke' if smoke else 'results', task)
        lane.q.atomic(man, path, {'complete': success, 'status': 'complete' if success else 'error',
                  'manifest_id': self.man['manifest_id'], 'physical_gpu': {
                      'uuid': owner['uuid'], 'pci_bus_id': owner['pci']}, 'metrics': {'rmse': 1.}})
        return success

    def test_small_task_bounds_are_scheduling_only(self):
        before = deepcopy(self.task)
        self.assertTrue(lane.small_task(self.task)['eligible'])
        self.assertEqual(self.task, before)
        for key, value in [('train_rows', 2049), ('features', 101), ('features', True), ('test_rows', None)]:
            row = deepcopy(self.task); row['row'][key] = value
            self.assertFalse(lane.small_task(row)['eligible'])

    def test_unknown_regression_shapes_skip_without_guessing_work_size(self):
        task = {**self.task, 'row': {'work_size': 1}}
        self.assertFalse(lane.small_task(task)['eligible'])
        overlay = {'3': {'train_rows': 800, 'test_rows': 200, 'features': 25}}
        self.assertTrue(lane.small_task(task, overlay)['eligible'])
        self.assertEqual(task['row'], {'work_size': 1})

    def test_hierarchy_allowed_when_exact_shape_small_no_data_truncation(self):
        task = {**deepcopy(self.task), 'task_kind': 'classification'}
        task['row']['classes'] = 20
        self.assertTrue(lane.small_task(task)['eligible'])

    def test_success_publishes_once_and_retains_canonical_claim(self):
        def launch(man, campaign, task, owner, smoke=False):
            self.assertNotEqual(man['output_root'], self.man['output_root'])
            self.assertEqual(man['manifest_id'], self.man['manifest_id'])
            self.assertEqual(campaign, self.root/'frozen-manifest.json')
            return self.write_result(man, task, owner, True, smoke)
        result = self.run_attempt(launch)
        self.assertEqual(result['state'], 'complete')
        self.assertTrue(lane.q.task_path(self.man, 'results', self.task).exists())
        self.assertTrue(lane.q.task_path(self.man, 'claims', self.task).exists())
        again = self.run_attempt(lambda *a, **k: self.fail('duplicate child launch'))
        self.assertEqual(again['state'], 'already_claimed')

    def test_resource_deferral_never_poisons_canonical_output_and_retains_own_claim(self):
        high = {**self.memory, 'own_tree_rss_bytes': 33*lane.GIB}
        def launch(man, campaign, task, owner, smoke=False):
            try:
                lane.q.process_rss(99)
            except lane.OperationalDeferral:
                return self.write_result(man, task, owner, False, smoke)
            self.fail('memory guard did not fire')
        result = self.run_attempt(launch, [self.memory, high])
        self.assertEqual(result['state'], 'retained_resource_deferral')
        self.assertFalse(lane.q.task_path(self.man, 'results', self.task).exists())
        self.assertEqual(lane.claim_identity(self.man, self.task, result['reservation']['token']), result['reservation'])
        self.assertTrue(result['reservation_retained'])
        self.assertFalse(result['automatic_retry'])
        self.assertTrue(result['recovery_requires_quiescent_authorization'])
        self.assertTrue(Path(result['attempt_output']).exists())
        proofs = list((self.sidecar/'deferrals').glob('*.json'))
        self.assertEqual(len(proofs), 1)
        self.assertEqual(lane.q.read(proofs[0]), result)
        self.assertFalse((self.sidecar/'released').exists())
        # Live frozen queues safely skip; only authorized quiescent recovery can retry.
        self.assertFalse(lane.q.claim(self.man, self.task, {'job': 'future'}))
        self.assertEqual(self.run_attempt(lambda *a, **k: self.fail('duplicate child launch'))['state'], 'already_claimed')

    def test_memory_inspection_failure_defers_instead_of_poisoning(self):
        def launch(man, campaign, task, owner, smoke=False):
            try:
                lane.q.process_rss(99)
            except lane.OperationalDeferral:
                return self.write_result(man, task, owner, False, smoke)
        result = self.run_attempt(launch, [self.memory, PermissionError('unmonitorable RAM')])
        self.assertEqual(result['state'], 'retained_resource_deferral')
        self.assertTrue(lane.q.task_path(self.man, 'claims', self.task).exists())
        self.assertFalse(lane.q.task_path(self.man, 'results', self.task).exists())

    def test_true_model_failure_keeps_claim_and_explicit_error(self):
        result = self.run_attempt(lambda man, path, task, owner, smoke=False:
                                  self.write_result(man, task, owner, False, smoke))
        self.assertEqual(result['state'], 'model_error')
        self.assertTrue(lane.q.task_path(self.man, 'claims', self.task).exists())
        value = lane.q.read(lane.q.task_path(self.man, 'results', self.task))
        self.assertEqual(value['status'], 'error')
        self.assertFalse(value['resource_deferral'])

    def test_smoke_artifacts_never_take_formal_claims(self):
        result = self.run_attempt(lambda man, path, task, owner, smoke=False:
                                  self.write_result(man, task, owner, True, smoke), smoke=True)
        self.assertEqual(result['state'], 'complete')
        self.assertFalse(lane.q.task_path(self.man, 'claims', self.task).exists())
        self.assertFalse(lane.q.task_path(self.man, 'results', self.task).exists())

    def test_foreign_claim_is_skipped_without_any_mutation(self):
        lane.q.claim(self.man, self.task, {'reservation_token': 'foreign'})
        path = lane.q.task_path(self.man, 'claims', self.task)
        before, inode = path.read_bytes(), path.stat().st_ino
        result = self.run_attempt(lambda *a, **k: self.fail('foreign claim launched'))
        self.assertEqual(result['state'], 'already_claimed')
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(path.stat().st_ino, inode)

    def test_cleanup_failure_retains_claim_and_never_publishes(self):
        with patch.object(lane, 'snapshot', return_value=self.memory), \
             patch.object(lane, 'cleanup_children', side_effect=RuntimeError('unproven cleanup')), \
             patch.object(lane.q, 'launch', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'unproven cleanup'):
                lane.attempt(self.plan, self.root/'manifest.json', self.man, self.task,
                             self.owner, self.budget, self.sidecar)
        self.assertTrue(lane.q.task_path(self.man, 'claims', self.task).exists())
        self.assertFalse(lane.q.task_path(self.man, 'results', self.task).exists())

    def test_claim_release_operation_is_not_exposed(self):
        self.assertFalse(hasattr(lane, 'release_owned'))
        self.assertNotIn('.unlink(', Path(lane.__file__).read_text())

    def test_guard_budget_and_global_rss(self):
        with self.assertRaises(lane.OperationalDeferral):
            lane.guard(self.memory, types.SimpleNamespace(remaining=lambda: 59))
        with self.assertRaisesRegex(lane.OperationalDeferral, 'same_uid'):
            lane.guard({**self.memory, 'same_uid_rss_bytes': 61*lane.GIB}, self.budget)
        with self.assertRaisesRegex(lane.OperationalDeferral, 'startup_other'):
            lane.guard({**self.memory, 'other_same_uid_rss_bytes': 21*lane.GIB}, self.budget, startup=True)

    def test_real_probe_rss_bytes_schema(self):
        plan = {'parent_job_id': '123', 'node': 'node', 'gpu': {'uuid': '1111222233334444', 'pci': '0000:03:00.0'}}
        record = {'complete': True, 'parent': '123', 'node': 'node', 'job_fields': {
            'JobId': '123', 'JobState': 'RUNNING', 'NumNodes': '1', 'NodeList': 'node',
            'UserId': 'fixture('+str(os.getuid())+')', 'AllocTRES': 'cpu=64,mem=64G,gres/gpu=8',
            'TimeLimit': '24:00:00', 'RunTime': '01:00:00'},
            'gpus': [{**plan['gpu'], 'hardware_idle': True, 'foreign_fd_owner_pids': [], 'busy_percent': 0,
                      'vram_used_bytes': 13*1024**2}], 'owned_processes': [{'rss_bytes': 19*lane.GIB}],
            'same_uid_rss_bytes': 19*lane.GIB}
        lane.validate_probe(record, plan)
        record['same_uid_rss_bytes'] += 1
        with self.assertRaisesRegex(RuntimeError, 'Inconsistent'):
            lane.validate_probe(record, plan)

    def test_existing_four_core_launcher_mask_is_accepted_only_with_matching_gate(self):
        path = self.root/'node-resource-gate.json'
        plan = {'plan_id': 'runtime', 'parent_job_id': '123', 'node': 'node'}
        gate = {'plan_id': 'runtime', 'job': '123', 'step': '9', 'node': 'node',
                'actual_cpu_ids': [4,5,6,7], 'inherited_cpu_ids': list(range(64))}
        path.write_text(json.dumps(gate))
        with patch.dict(sys.modules, {'psutil': types.SimpleNamespace()}), \
             patch.dict(os.environ, {'SLURM_STEP_ID': '9'}), \
             patch.object(os, 'sched_getaffinity', return_value={4,5,6,7}, create=True):
            self.assertEqual(lane.bind_cpus(plan, path)['selected'], [4,5,6,7])
            gate['plan_id'] = 'other'; path.write_text(json.dumps(gate))
            with self.assertRaisesRegex(RuntimeError, 'resource gate'):
                lane.bind_cpus(plan, path)


if __name__ == '__main__':
    unittest.main()
