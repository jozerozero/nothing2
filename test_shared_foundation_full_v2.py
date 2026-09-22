"""CPU mocks only. No GPU, scheduler, model or campaign mutations."""
import copy
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import shared_foundation_full_v2 as full


class FullLaneTests(unittest.TestCase):
    def setUp(self):
        self.task = {'task_kind': 'regression', 'dataset_index': 223, 'dataset': 'large',
                     'data_manifest_id': 'data', 'row': {'train_rows': 100000, 'test_rows': 50000,
                                                       'features': 1000, 'work_size': 150000000}}
        self.man = {'manifest_id': 'science', 'output_root': '/campaign', 'native_setting': 32}
        self.owner = {'uuid': 'physical', 'pci': '0000:47:00.0'}
        self.reservation = {'token': 'owned', 'path': '/campaign/claims/regression/row-223.json'}
        self.plan = {'plan_id': 'operation'}
        self.root = Path('/campaign/sidecars/new')

    def test_full_large_and_unknown_rows_eligible_without_mutation(self):
        original = copy.deepcopy(self.task)
        self.assertTrue(full.full_task(self.task)['eligible'])
        self.assertEqual(self.task, original)
        self.task['row'] = {'work_size': 1}
        self.assertTrue(full.full_task(self.task)['eligible'])
        self.assertFalse(full.full_task(self.task)['unknown_dimensions_skipped'])

    def test_scoped_legacy_globals_are_replaced_and_restored_on_exception(self):
        before = (full.base.ELIGIBILITY, full.base.small_task, full.base.attempt, full.base.source_checks)
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with full.installed_policy():
                self.assertIs(full.base.small_task, full.full_task)
                self.assertIs(full.base.run.__globals__['small_task'], full.full_task)
                self.assertIs(full.base.run.__globals__['attempt'], full.attempt)
                self.assertEqual(full.base.ELIGIBILITY, full.ELIGIBILITY)
                raise RuntimeError('test')
        self.assertEqual(before, (full.base.ELIGIBILITY, full.base.small_task, full.base.attempt, full.base.source_checks))
        self.assertFalse(full.INSTALLED)

    def test_nested_policy_rejected(self):
        with full.installed_policy():
            with self.assertRaisesRegex(RuntimeError, 'Nested'):
                with full.installed_policy(): pass

    @contextmanager
    def mocked_attempt(self, *, success=True, events=False, launch_error=None, cleanup_error=None, invalid=False):
        order, published = [], []

        @contextmanager
        def watched(_budget, captured):
            if events: captured.append({'reason': 'own32GiB'})
            yield

        def launch(man, path, task, owner, smoke=False):
            order.append('launch')
            self.assertEqual(path, Path('/campaign/manifest.json'))
            self.assertIs(task, self.task)
            self.assertEqual(man['native_setting'], 32)
            self.assertNotEqual(man['output_root'], self.man['output_root'])
            if launch_error: raise launch_error
            return success

        def cleanup():
            order.append('cleanup')
            if cleanup_error: raise cleanup_error
            return {'all_owned_children_reaped': True}

        def atomic(man, path, value):
            self.assertIn('cleanup', order)
            order.append('publish')
            published.append((Path(path), value))

        result = {'complete': True, 'physical_gpu': {'uuid': 'physical', 'pci_bus_id': '0000:47:00.0'}}
        with mock.patch.object(full.base, 'guard'), mock.patch.object(full.base, 'snapshot', return_value={}), \
             mock.patch.object(full.q, 'claim', return_value=True) as claim, \
             mock.patch.object(full.base, 'claim_identity', return_value=self.reservation), \
             mock.patch.object(full.base, 'guarded_runtime', watched), \
             mock.patch.object(full.q, 'launch', side_effect=launch), \
             mock.patch.object(full.base, 'cleanup_children', side_effect=cleanup), \
             mock.patch.object(full.q, 'valid_result', side_effect=RuntimeError('invalid') if invalid else None,
                               return_value=result), mock.patch.object(full.q, 'atomic', side_effect=atomic):
            yield published, order, claim

    def call_attempt(self, smoke=False):
        return full.attempt(self.plan, Path('/campaign/manifest.json'), self.man, self.task,
                            self.owner, object(), self.root, smoke=smoke)

    def test_success_publishes_only_after_cleanup(self):
        with self.mocked_attempt() as (published, order, _):
            result = self.call_attempt()
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(order[:2], ['launch', 'cleanup'])
        self.assertEqual(published[0][0], Path('/campaign/results/regression/row-223.json'))
        self.assertTrue(published[0][1]['complete'])

    def test_native_failure_retains_claim_no_canonical_error(self):
        with self.mocked_attempt(success=False) as (published, _, _): result = self.call_attempt()
        self.assertEqual(result['state'], 'retained_resource_deferral')
        self.assertEqual(result['reason'], 'unsuccessful_or_unverified_native_attempt')
        self.assertTrue(result['reservation_retained'])
        self.assertFalse(result['automatic_retry'])
        self.assertEqual(len(published), 1)
        self.assertIn('/deferrals/', str(published[0][0]))

    def test_runtime_failure_guard_invalid_receipt_and_signal_defer(self):
        for kwargs in ({'events': True}, {'invalid': True}, {'launch_error': SystemExit(143)},
                       {'launch_error': RuntimeError('CUDA out of memory')}):
            with self.mocked_attempt(**kwargs) as (published, _, _): result = self.call_attempt()
            self.assertEqual(result['state'], 'retained_resource_deferral')
            self.assertTrue(all('/deferrals/' in str(path) for path, _ in published))

    def test_cleanup_failure_never_publishes(self):
        with self.mocked_attempt(cleanup_error=RuntimeError('live descendant')) as (published, _, _):
            with self.assertRaisesRegex(RuntimeError, 'live descendant'): self.call_attempt()
        self.assertEqual(published, [])

    def test_smoke_failure_no_formal_claim(self):
        with self.mocked_attempt(success=False) as (published, _, claim): result = self.call_attempt(smoke=True)
        claim.assert_not_called()
        self.assertEqual(result['state'], 'operationally_deferred')
        self.assertIsNone(result['reservation'])
        self.assertEqual(len(published), 1)

    def test_foreign_existing_claim_skips_without_model_call(self):
        with self.mocked_attempt() as (published, order, claim):
            claim.return_value = False
            result = self.call_attempt()
        self.assertEqual(result['state'], 'already_claimed')
        self.assertEqual(order, [])
        self.assertEqual(published, [])

    def test_allowlist_and_runtime_contract(self):
        self.assertEqual(len(full.ALLOWED_PARENTS), 9)
        self.assertEqual(full.RESOURCE['own_rss_gib'], 32)
        self.assertEqual(full.RESOURCE['total_uid_rss_gib'], 60)
        self.assertEqual(full.RESOURCE['step_mem_gib'], 40)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'plan.json'; path.write_text('{}')
            plan = dict(parent_job_id='196092', resource=full.RESOURCE, cpus=4, mem_gib=40,
                        parent_mem_gib=64, max_step_seconds=7200, entry_script=str(Path(full.__file__).resolve()))
            with mock.patch.object(full.base, 'load_plan', return_value=(plan, ['campaigns'])), \
                 mock.patch.object(full.base, 'run', return_value={'ok': True}) as run:
                self.assertEqual(full.load_and_run(path), {'ok': True})
                run.assert_called_once()
            for changed in ({'parent_job_id': 'unapproved'}, {'cpus': 16}, {'mem_gib': 64}):
                with mock.patch.object(full.base, 'load_plan', return_value=({**plan, **changed}, [])), \
                     mock.patch.object(full.base, 'run') as run:
                    with self.assertRaises(RuntimeError): full.load_and_run(path)
                    run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
