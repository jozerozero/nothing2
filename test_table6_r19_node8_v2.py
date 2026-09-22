"""CPU-only operational tests; no Slurm/model/GPU/remote commands."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import table6_r19_node8_v2 as lane


class NodeTests(unittest.TestCase):
    def test_script_reserves_full_eight_not_per_rank_gres(self):
        body = lane.script()
        for text in ('#SBATCH --nodes=1', '#SBATCH --ntasks=8', '#SBATCH --cpus-per-task=8',
                     '#SBATCH --gpus=8', '#SBATCH --mem=512G', '#SBATCH --time=02:00:00',
                     '#SBATCH --no-requeue', '--cpus-per-task=64 --gpus=8',
                     '--expected-cpus 64 --expected-mem-gib 512 --expected-seconds 7200',
                     '--cpus-per-task=8 --gpus=8', '--gpu-bind=none', 'allocated_gpu_uuid.py bootstrap',
                     'allocated_gpu_uuid.py exec', 'rocm_gpu_entry.py', 'manage.py check-inputs'):
            self.assertIn(text, body)
        for forbidden in ('--gpus-per-task', '--gpu-bind=single', '--dependency', '--overlap', 'sbatch', 'scancel'):
            self.assertNotIn(forbidden, body)
        self.assertEqual(body.count('\nsrun '), 2)
        self.assertEqual(subprocess.run(['bash', '-n'], input=body, text=True, capture_output=True).returncode, 0)

    def test_native_command_keeps_original_full_launch_gates(self):
        self.assertEqual(lane.native_command(),
                         [lane.PY, '-B', str(lane.REPO/'table6_restart_gpu.py'), 'launch', '--mode', 'gpu'])
        self.assertNotIn('worker', lane.native_command())

    def test_prepare_is_new_namespace_pins_frozen_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            plan = {'path': '/frozen/plan.json', 'sha256': 'frozen', 'plan_id': lane.PLAN_ID}
            with patch.object(lane, 'STAGE', root/'new-stage'), patch.object(lane, 'LOGS', root/'new-logs'), \
                 patch.object(lane.original, 'verify') as original_verify, \
                 patch.object(lane, 'scientific_plan', return_value=plan), \
                 patch.object(lane, 'sources', return_value={'new': 'pinned', 'old': 'unchanged'}):
                record = lane.prepare()
                self.assertEqual(record['resources'], {'nodes': 1, 'gpus': 8, 'cpus': 64,
                    'cpus_per_rank': 8, 'mem_gib': 512, 'seconds': 7200, 'dependency': None})
                self.assertEqual(lane.verify(), record)
                original_verify.assert_called()
                with self.assertRaisesRegex(RuntimeError, 'already exists'):
                    lane.prepare()
                with patch.object(lane, 'sources', return_value={'old': 'changed'}):
                    with self.assertRaisesRegex(RuntimeError, 'source changed'):
                        lane.verify()
                with patch.object(lane, 'scientific_plan', return_value={**plan, 'sha256': 'changed'}):
                    with self.assertRaisesRegex(RuntimeError, 'plan changed'):
                        lane.verify()

    def test_scientific_contract_rejects_missing_or_changed_memberships_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'plan.json'
            plan = {'plan_id': lane.PLAN_ID, 'hpo_trials': 100, 'seeds': list(range(15)),
                    'pairs': [None]*12482, 'classification_memberships': 457, 'regression_memberships': 224}
            with patch.object(lane, 'PLAN', path):
                path.write_text(json.dumps(plan))
                self.assertEqual(lane.scientific_plan()['plan_id'], lane.PLAN_ID)
                for change in ({'seeds': list(range(14))}, {'hpo_trials': 99},
                               {'classification_memberships': 456}, {'pairs': []}):
                    path.write_text(json.dumps({**plan, **change}))
                    with self.assertRaisesRegex(RuntimeError, 'scientific contract changed'):
                        lane.scientific_plan()

    def test_sources_require_frozen_short_worker_hash(self):
        with patch.object(lane, 'sha', return_value='wrong'):
            with self.assertRaisesRegex(RuntimeError, 'short worker changed'):
                lane.sources()

    def test_rank_requires_uuid_exact8cpus_and_shared_monotonic_budget(self):
        import allocated_gpu_uuid as binding
        mapping = {'expected_cpus': 64, 'expected_mem_gib': 512, 'expected_seconds': 7200}
        env = {'ROCR_VISIBLE_DEVICES': 'GPU-aaaabbbbccccdddd', 'EXPECTED_GPU_UUID': 'aaaabbbbccccdddd',
               'EXPECTED_GPU_PCI_BUS_ID': '0000:88:00.0', 'ALLOCATED_GPU_MAPPING_ID': 'map',
               'JOB_BUDGET_END_MONOTONIC': '10000', 'JOB_BUDGET_MONOTONIC_HOST': 'node',
               'JOB_BUDGET_JOB_ID': '123', 'JOB_BUDGET_END_EPOCH': '99999',
               'SLURM_PROCID': '3', 'SLURM_CPUS_PER_TASK': '8'}
        budget = types.SimpleNamespace(monotonic_end=10000, remaining=lambda: 6000)
        with patch.dict(os.environ, env, clear=True), patch.object(binding, 'rank_environment', return_value=env), \
             patch.object(os, 'sched_getaffinity', return_value=set(range(8)), create=True), \
             patch.object(lane.EnvironmentBudget, 'from_environment', return_value=budget):
            self.assertEqual(lane.rank_contract(mapping), 3)
            with patch.dict(os.environ, {'HIP_VISIBLE_DEVICES': '0'}):
                with self.assertRaisesRegex(RuntimeError, 'double masking'):
                    lane.rank_contract(mapping)
            with patch.dict(os.environ, {'EXPECTED_GPU_UUID': 'wrong'}):
                with self.assertRaisesRegex(RuntimeError, 'environment mismatch'):
                    lane.rank_contract(mapping)
            with patch.object(os, 'sched_getaffinity', return_value=set(range(4))):
                with self.assertRaisesRegex(RuntimeError, 'exactly8'):
                    lane.rank_contract(mapping)
            with patch.object(lane.EnvironmentBudget, 'from_environment',
                              return_value=types.SimpleNamespace(monotonic_end=None, remaining=lambda: 6000)):
                with self.assertRaisesRegex(RuntimeError, 'shared monotonic'):
                    lane.rank_contract(mapping)
            for seconds in (180, 6902):
                with patch.object(lane.EnvironmentBudget, 'from_environment',
                                  return_value=types.SimpleNamespace(monotonic_end=10000, remaining=lambda: seconds)):
                    with self.assertRaisesRegex(RuntimeError, 'shared monotonic'):
                        lane.rank_contract(mapping)

    def test_handoff_requires_actual_physical_probe_and_execs_native_unchanged(self):
        import allocated_gpu_uuid as binding
        mapping = {'job': '123', 'node': 'node', 'mapping_id': 'map'}
        path = lane.STAGE/'runtime/job-123/mapping.json'
        identity = {'uuid': 'aaaabbbbccccdddd', 'pci_bus_id': '0000:88:00.0'}
        probe = types.SimpleNamespace(gpu_identity=lambda torch: identity)
        class ExecIntercept(BaseException):
            pass
        with patch.object(lane, 'verify', return_value={'source_hashes': {'native': 'unchanged'}}), \
             patch.object(binding, 'load_mapping', return_value=mapping), patch.object(lane, 'rank_contract', return_value=3), \
             patch.dict(os.environ, {'T6_BASE_STAGE': str(lane.BASE), 'SLURM_STEP_ID': '1'}, clear=True), \
             patch.object(lane.tempfile, 'mkdtemp', return_value='/tmp/private-test'), \
             patch.dict(sys.modules, {'torch': types.SimpleNamespace(), 'pfn_mitra_one': probe}), \
             patch.object(os, 'sched_getaffinity', return_value=set(range(8)), create=True), \
             patch.object(lane, 'publish') as publish, patch.object(os, 'execv', side_effect=ExecIntercept) as execute:
            with self.assertRaises(ExecIntercept):
                lane.rank_action(path)
            execute.assert_called_once_with(lane.PY, lane.native_command())
            receipt = publish.call_args.args[1]
            self.assertEqual(receipt['physical_gpu'], identity)
            self.assertTrue(receipt['native_startup_gates_not_skipped'])
            self.assertEqual(receipt['scientific_plan_id'], lane.PLAN_ID)
            self.assertEqual(receipt['rank'], 3)

    def test_foreign_mapping_path_prevents_probe_or_worker(self):
        import allocated_gpu_uuid as binding
        with patch.object(lane, 'verify', return_value={}), \
             patch.object(binding, 'load_mapping', return_value={'job': '123'}), \
             patch.object(os, 'execv') as execute:
            with self.assertRaisesRegex(RuntimeError, 'outside this deployment'):
                lane.rank_action('/tmp/other-mapping.json')
        execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
