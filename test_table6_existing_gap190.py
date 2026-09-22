"""Local CPU mocks only; no Slurm, GPU, model execution, or remote writes."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import table6_existing_gap190 as lane


class LaneTests(unittest.TestCase):
    def plan(self):
        return {'parent_job_id': '123', 'node': 'node-a', 'cpus': 16, 'mem_gib': 40,
                'parent_mem_gib': 64, 'max_step_seconds': 7200, 'sidecar_id': 'lane-one',
                'gpu': {'uuid': 'a123', 'pci': '0000:88:00.0'}, 'proof': {'verified': True}}

    def env(self):
        return {'SLURM_JOB_ID': '123', 'SLURM_NTASKS': '1', 'SLURM_PROCID': '0',
                'SLURM_LOCALID': '0', 'SLURM_STEP_ID': '5', 'ROCR_VISIBLE_DEVICES': 'GPU-a123',
                'JOB_BUDGET_JOB_ID': '123'}

    def verify(self, plan=None, env=None, remaining=6500, affinity=range(16), reader=None):
        budget = types.SimpleNamespace(monotonic_end=9000, remaining=lambda: remaining)
        with patch.object(lane.EnvironmentBudget, 'from_environment', return_value=budget):
            return lane.verify_node(plan or self.plan(), environ=env or self.env(), hostname='node-a',
                                    affinity=affinity, gpu_reader=reader or (lambda pci: 'a123'))

    def test_true_one_rank_binding_and_budget(self):
        self.assertEqual(self.verify().remaining(), 6500)
        for changed in ({'SLURM_JOB_ID': 'other'}, {'SLURM_NTASKS': '2'}, {'SLURM_STEP_ID': 'batch'},
                        {'SLURM_LOCALID': '1'}, {'ROCR_VISIBLE_DEVICES': '0'}, {'HIP_VISIBLE_DEVICES': '0'},
                        {'JOB_BUDGET_JOB_ID': 'other'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.verify(env={**self.env(), **changed})
        for remaining in (0, 120, 7418, float('nan')):
            with self.assertRaises(ValueError): self.verify(remaining=remaining)

    def test_cpu_affinity_and_physical_uuid_must_match(self):
        for affinity in (range(8), range(17)):
            with self.assertRaises(ValueError): self.verify(affinity=affinity)
        with self.assertRaises(ValueError): self.verify(reader=lambda pci: 'wrong')
        plan = self.plan(); plan['cpu_ids'] = list(range(1, 17))
        with self.assertRaises(ValueError): self.verify(plan=plan)

    def test_rss_limits_count_full_own_tree_and_conservative_parent(self):
        lane.check_rss({'own_tree_rss_bytes': 34*lane.GIB, 'same_uid_node_rss_bytes': 60*lane.GIB})
        for own, total in ((34*lane.GIB+1, 55*lane.GIB), (20*lane.GIB, 60*lane.GIB+1), (-1, 0)):
            with self.assertRaises(ValueError):
                lane.check_rss({'own_tree_rss_bytes': own, 'same_uid_node_rss_bytes': total})

    def test_immutable_audit_and_plan_source_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lane.publish(root, 'receipt', {'complete': True})
            with self.assertRaises(FileExistsError): lane.publish(root, 'receipt', {'complete': False})
            path = root/'receipt.json'
            record = {'path': str(path), 'sha256': lane.hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(lane.verify_file(record), path.resolve())
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'SHA changed'): lane.verify_file(record)

    def test_plan_pins_exact_runtime_sources_and_scientific_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            scientific = {'output_root': str(root/'scientific-output')}
            scientific['plan_id'] = lane.digest(scientific)
            scientific_path = root/'scientific-plan.json'
            scientific_path.write_text(json.dumps(scientific))
            def record(path):
                return {'path': str(path), 'sha256': lane.hashlib.sha256(path.read_bytes()).hexdigest()}
            plan = self.plan()
            plan['scientific_plan'] = record(scientific_path)
            plan['source_records'] = [record(Path(lane.__file__).resolve().with_name(name))
                                      for name in lane.REQUIRED_SOURCES]
            plan['plan_id'] = lane.digest(plan)
            path = root/'runtime-plan.json'; path.write_text(json.dumps(plan))
            with patch.object(lane, 'SCIENTIFIC_PLAN_ID', scientific['plan_id']):
                loaded, source, document = lane.load_plan(path)
                self.assertEqual(loaded, plan)
                self.assertEqual(source, scientific_path)
                self.assertEqual(document, scientific)
                bad = copy.deepcopy(plan); bad['source_records'].pop(); bad.pop('plan_id')
                bad['plan_id'] = lane.digest(bad); path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, 'missing runtime'): lane.load_plan(path)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'wrong frozen'): lane.load_plan(path)

    def test_same_scientific_namespace_and_runtime_identity_preserved(self):
        import table6_missing190_worker as original
        class Pause(BaseException): pass
        class Budget:
            def __init__(self, hard_end): self.hard_end = hard_end; self.stop_reason = None
            def request_stop(self, reason): self.stop_reason = reason
            def check(self, new_fit=False):
                if self.stop_reason: raise Pause(self.stop_reason)
        interrupted = Pause('sidecar_resource_guard: memory')
        interrupted.fit_elapsed_seconds = 6500
        short = types.SimpleNamespace(Budget=Budget, BudgetStop=Pause, _ACTIVE=None,
                                      execute=Mock(side_effect=interrupted), signal_stop=Mock())
        runtime = types.SimpleNamespace(sources={'worker': 'frozen', 'fit': 'frozen'}, short=short)
        with tempfile.TemporaryDirectory() as temporary, patch.object(original, 'Runtime', return_value=runtime), \
                patch.object(lane.signal, 'signal'), patch.dict(os.environ, {'SLURM_STEP_ID': '5'}):
            budget = types.SimpleNamespace(hard_end_epoch=100, remaining=lambda: 1000)
            output = lane.setup_runtime({}, Path('/original/plan.json'), budget, Path(temporary))
            self.assertEqual(output.sources, {'worker': 'frozen', 'fit': 'frozen'})
            with self.assertRaises(Pause) as raised: output.short.execute('unchanged-request')
            self.assertFalse(hasattr(raised.exception, 'fit_elapsed_seconds'))
            output.short.execute.__closure__  # The old execute remains delegated, not rewritten.
            receipts = list(Path(temporary).glob('resource-pause-*.json'))
            self.assertEqual(len(receipts), 1)
            self.assertTrue(json.loads(receipts[0].read_text())['original_cleanup_completed'])

    def test_resource_flag_takes_precedence_over_parent_sigterm(self):
        import table6_missing190_worker as original
        class Pause(BaseException): pass
        class Budget:
            def __init__(self, hard_end): self.stop_reason = None
            def check(self, new_fit=False):
                if self.stop_reason: raise Pause(self.stop_reason)
        short = types.SimpleNamespace(Budget=Budget, BudgetStop=Pause, execute=Mock(), signal_stop=Mock())
        runtime = types.SimpleNamespace(sources={}, short=short)
        with tempfile.TemporaryDirectory() as temporary, patch.object(original, 'Runtime', return_value=runtime), \
                patch.object(lane.signal, 'signal'), patch.dict(os.environ, {'SLURM_STEP_ID': '5'}):
            audit = Path(temporary)
            lane.publish(audit, 'resource-stop-step-5', {'reason': 'memory'})
            lane.setup_runtime({}, Path('/original/plan'), types.SimpleNamespace(hard_end_epoch=100), audit)
            short.BUDGET.stop_reason = 'signal_SIGTERM'
            with self.assertRaisesRegex(Pause, 'sidecar_resource_guard: memory'): short.BUDGET.check()

    def test_stop_reaps_owned_child_without_signalling_other_processes(self):
        child = Mock(); child.poll.return_value = None
        lane.stop_owned(child)
        child.send_signal.assert_called_once_with(signal.SIGTERM)
        child.wait.assert_called_once_with(timeout=30)

    def test_supervisor_order_keeps_nine_smokes_before_gate_and_formal(self):
        launches = []
        def spawn(argv, **kwargs):
            self.assertTrue(kwargs['start_new_session'])
            launches.append(argv)
            return types.SimpleNamespace(poll=lambda: 0, wait=lambda: 0, returncode=0)
        with tempfile.TemporaryDirectory() as temporary, patch.object(lane.signal, 'signal'), \
                patch.object(lane, 'rss_snapshot', return_value={'own_tree_rss_bytes': 1, 'same_uid_node_rss_bytes': 2}), \
                patch.object(lane.subprocess, 'Popen', side_effect=spawn):
            out = lane.supervise(Path('/pinned/plan'), self.plan(), {'output_root': temporary},
                                 types.SimpleNamespace(remaining=lambda: 6000))
        self.assertEqual(out['state'], 'worker_finished')
        self.assertEqual([argv[argv.index('--action')+1] for argv in launches],
                         ['preflight']+['smoke']*9+['gate', 'worker'])
        self.assertEqual(len({argv[-1] for argv in launches[1:10]}), 9)

    def test_supervisor_does_not_continue_after_pause_or_failed_smoke(self):
        for code in (75, 2):
            child = types.SimpleNamespace(poll=lambda: code, wait=lambda: code, returncode=code)
            with tempfile.TemporaryDirectory() as temporary, patch.object(lane.signal, 'signal'), \
                    patch.object(lane, 'rss_snapshot', return_value={'own_tree_rss_bytes': 1, 'same_uid_node_rss_bytes': 2}), \
                    patch.object(lane.subprocess, 'Popen', return_value=child) as spawn:
                args = (Path('/pinned/plan'), self.plan(), {'output_root': temporary},
                        types.SimpleNamespace(remaining=lambda: 6000))
                if code == 75:
                    self.assertEqual(lane.supervise(*args)['state'], 'paused')
                else:
                    with self.assertRaisesRegex(ValueError, 'action failed'): lane.supervise(*args)
                self.assertEqual(spawn.call_count, 1)

    def test_supervisor_expired_budget_launches_nothing(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(lane.signal, 'signal'), \
                patch.object(lane.subprocess, 'Popen') as spawn:
            result = lane.supervise(Path('/pinned/plan'), self.plan(), {'output_root': temporary},
                                    types.SimpleNamespace(remaining=lambda: 180))
            self.assertEqual(result['state'], 'paused')
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
