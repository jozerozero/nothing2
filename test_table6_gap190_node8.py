"""Local CPU contracts only; no Slurm submission, GPU/model or remote calls."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import table6_gap190_node8 as lane


class NodeTests(unittest.TestCase):
    def test_script_is_independent_full_node_hard_two_hours(self):
        text = lane.script()
        for value in ('#SBATCH --nodes=1', '#SBATCH --ntasks=8', '#SBATCH --cpus-per-task=16',
                      '#SBATCH --gpus=8', '#SBATCH --mem=512G', '#SBATCH --time=02:00:00',
                      '#SBATCH --no-requeue', '--expected-cpus 128', '--expected-mem-gib 512',
                      '--expected-seconds 7200', '--gpu-bind=none', 'for PHASE in smoke preflight gate worker'):
            self.assertIn(value, text)
        self.assertNotIn('--dependency', text)
        self.assertNotIn('208407', text)
        self.assertNotIn('208377', text)
        self.assertNotIn('table6_existing_r19', text)
        self.assertIn(str(lane.original.PLAN), text)
        self.assertEqual(subprocess.run(['bash','-n'], input=text, text=True, capture_output=True).returncode, 0)

    def test_nine_native_smokes_are_partitioned_once_over_eight_real_ranks(self):
        methods = ['method'+str(i) for i in range(9)]
        selected = [lane.methods_for_rank(methods, rank) for rank in range(8)]
        self.assertEqual(selected[0], ['method0','method8'])
        self.assertEqual(sorted(x for group in selected for x in group), methods)
        for bad in (-1, 8, True):
            with self.assertRaises(RuntimeError): lane.methods_for_rank(methods, bad)
        with self.assertRaises(RuntimeError): lane.methods_for_rank(methods[:-1], 0)

    def test_prepare_is_exclusive_and_pins_old_scientific_source_without_editing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); plan = root/'original-plan.json'; plan.write_text('frozen scientific')
            with patch.object(lane, 'STAGE', root/'new-stage'), patch.object(lane, 'LOGS', root/'new-logs'), \
                 patch.object(lane.original, 'PLAN', plan), patch.object(lane.original, 'verify') as old_verify, \
                 patch.object(lane.original, 'plan', return_value={'plan_id': lane.PLAN_ID}), \
                 patch.object(lane, 'sources', return_value={'native':'unchanged','new':'pinned'}):
                record = lane.prepare()
                self.assertEqual(record['resources']['cpus'], 128)
                self.assertEqual(record['resources']['dependency'], None)
                self.assertEqual(plan.read_text(), 'frozen scientific')
                self.assertEqual(lane.verify(), record)
                with self.assertRaisesRegex(RuntimeError, 'already exists'): lane.prepare()
                old_verify.assert_called()
                with patch.object(lane, 'sources', return_value={'native':'modified'}):
                    with self.assertRaisesRegex(RuntimeError, 'source changed'): lane.verify()

    def test_native_worker_command_never_rewrites_requests_or_plans(self):
        child = Mock(); child.poll.return_value = 0; child.wait.return_value = 0
        with patch.object(lane.subprocess, 'Popen', return_value=child) as spawn:
            lane.run_worker(['smoke','--method','CatBoost'], types.SimpleNamespace(remaining=lambda: 6000))
        self.assertEqual(spawn.call_args.args[0], [lane.PY,'-B',str(lane.REPO/'table6_missing190_worker.py'),
                          'smoke','--method','CatBoost','--plan',str(lane.original.PLAN)])
        self.assertTrue(spawn.call_args.kwargs['start_new_session'])

    def test_no_new_child_on_exhausted_shared_budget(self):
        with patch.object(lane.subprocess, 'Popen') as spawn:
            with self.assertRaises(lane.Paused):
                lane.run_worker(['smoke','--method','TabM'], types.SimpleNamespace(remaining=lambda: 180))
        spawn.assert_not_called()

    def test_nonzero_native_failure_prevents_gate_progress(self):
        child = Mock(); child.poll.return_value = 3; child.wait.return_value = 3
        with patch.object(lane.subprocess, 'Popen', return_value=child):
            with self.assertRaisesRegex(RuntimeError, 'native action failed'):
                lane.run_worker(['gate','--ranks','8'], types.SimpleNamespace(remaining=lambda: 6000))

    def test_budget_stop_signals_only_own_native_child(self):
        import table6_existing_gap190 as cleanup
        child = Mock(); child.poll.side_effect = [None, 0]
        values = iter([6000, 80])
        with patch.object(lane.subprocess, 'Popen', return_value=child), \
             patch.object(cleanup, 'stop_owned') as stop:
            with self.assertRaises(lane.Paused):
                lane.run_worker(['preflight','--mode','gpu'], types.SimpleNamespace(remaining=lambda: next(values)))
        stop.assert_called_once_with(child)

    def test_rank_environment_requires_actual_uuid_shared_monotonic_budget_and16cores(self):
        import allocated_gpu_uuid as binding
        mapping = {'expected_cpus':128, 'expected_mem_gib':512, 'expected_seconds':7200}
        expected = {'ROCR_VISIBLE_DEVICES':'GPU-aaaabbbbccccdddd', 'EXPECTED_GPU_UUID':'aaaabbbbccccdddd',
                    'EXPECTED_GPU_PCI_BUS_ID':'0000:88:00.0', 'ALLOCATED_GPU_MAPPING_ID':'map',
                    'JOB_BUDGET_END_MONOTONIC':'10000', 'JOB_BUDGET_MONOTONIC_HOST':'node',
                    'JOB_BUDGET_JOB_ID':'123', 'JOB_BUDGET_END_EPOCH':'99999'}
        budget = types.SimpleNamespace(monotonic_end=10000, remaining=lambda:6000)
        with patch.dict(os.environ, {**expected,'SLURM_PROCID':'3'}, clear=True), \
             patch.object(binding, 'rank_environment', return_value=expected), \
             patch.object(os, 'sched_getaffinity', return_value=set(range(16)), create=True), \
             patch.object(lane.EnvironmentBudget, 'from_environment', return_value=budget):
            self.assertEqual(lane.rank_contract(mapping), (3,budget))
            with patch.dict(os.environ, {'HIP_VISIBLE_DEVICES':'0'}):
                with self.assertRaisesRegex(RuntimeError, 'double masking'): lane.rank_contract(mapping)
            with patch.object(os, 'sched_getaffinity', return_value=set(range(8))):
                with self.assertRaisesRegex(RuntimeError, '16 bound'): lane.rank_contract(mapping)

    def test_native_preflight_compares_physical_pci_and_current_cpu_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); directory = root/'preflight/123'; directory.mkdir(parents=True)
            mapping = {'job':'123','node':'node','gpus':[{'pci':'0000:88:00.0'}]}
            record = {'passed':True,'job_id':'123','rank':0,'host':'node','plan_id':lane.PLAN_ID,
                      'mode':'gpu','devices':1,'physical_gpu':'00000000:88:00.0','cpu_affinity':list(range(16))}
            target = directory/'gpu-0.json'; target.write_text(json.dumps(record))
            with patch.object(lane.original, 'OUT', root), \
                 patch.object(os, 'sched_getaffinity', return_value=set(range(16)), create=True):
                self.assertEqual(lane.validate_native_preflight(mapping,0), record)
                record['physical_gpu'] = '0000:89:00.0'; target.write_text(json.dumps(record))
                with self.assertRaisesRegex(RuntimeError, 'differs from UUID'): lane.validate_native_preflight(mapping,0)


if __name__ == '__main__':
    unittest.main()
