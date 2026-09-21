"""Stdlib-only sidecar validation tests; no Slurm/remote calls."""
import copy
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import table6_restart_ag_sidecar as side
from table6_restart_deadline import EnvironmentBudget


class SidecarTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proof = {'parent_job_id': '206116', 'node': 'node308', 'allow_cpu_sidecar': True,
                      'observations': [{'observed_epoch': 100, 'available_cpus': 128,
                                        'busy_cpus': .2, 'available_memory_bytes': side.MEMORY + 100},
                                       {'observed_epoch': 105, 'available_cpus': 127,
                                        'busy_cpus': .3, 'available_memory_bytes': side.MEMORY + 100}]}

    def raw(self, **update):
        value = {'JobId': '206116', 'JobState': 'RUNNING', 'NumNodes': '1', 'NodeList': 'node308',
                 'UserId': f'user({os.getuid()})', 'NumCPUs': '128', 'MinMemoryNode': '1024G',
                 'TimeLimit': '1-00:00:00', 'RunTime': '01:00:00', 'EndTime': '2026-09-23T00:00:00'}
        value.update(update)
        return ' '.join(f'{k}={v}' for k, v in value.items())

    def test_two_fresh_capacity_observations(self):
        self.assertEqual(side.validate_proof(self.proof, '206116', 'node308', now=110), self.proof)
        for key, bad in [('available_cpus', 63), ('available_memory_bytes', side.MEMORY - 1),
                         ('busy_cpus', 64), ('observed_epoch', -1000)]:
            value = copy.deepcopy(self.proof)
            value['observations'][0][key] = bad
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                side.validate_proof(value, '206116', 'node308', now=110)
        value = copy.deepcopy(self.proof)
        value['observations'] = value['observations'][:1]
        with self.assertRaises(RuntimeError):
            side.validate_proof(value, '206116', 'node308', now=110)

    def test_proof_is_exact_parent_node_allowlist(self):
        for parent, node in [('206115', 'node308'), ('206116', 'node309')]:
            with self.assertRaises(RuntimeError):
                side.validate_proof(self.proof, parent, node, now=110)

    def test_parent_requires_owned_running_one_node_resources(self):
        self.assertEqual(side.validate_parent(self.raw(), '206116', 'node308')['NumCPUs'], '128')
        for update in [{'UserId': 'other(9999999)'}, {'JobState': 'PENDING'}, {'NumNodes': '2'},
                       {'NodeList': 'node309'}, {'NumCPUs': '63'}, {'MinMemoryNode': '64G'},
                       {'TimeLimit': '01:01:00'}]:
            with self.subTest(update=update), self.assertRaises(RuntimeError):
                side.validate_parent(self.raw(**update), '206116', 'node308')

    def test_all_node_memory_requires_actual_allocated_tres_evidence(self):
        side.validate_parent(self.raw(MinMemoryNode='0', AllocTRES='cpu=128,mem=1024G,node=1'),
                             '206116', 'node308')
        with self.assertRaises(RuntimeError):
            side.validate_parent(self.raw(MinMemoryNode='0'), '206116', 'node308')
        with self.assertRaises(RuntimeError):
            side.validate_parent(self.raw(MinMemoryNode='1024G', AllocTRES='cpu=128,mem=64G,node=1'),
                                 '206116', 'node308')

    def test_clean_environment_drops_all_foreign_slurm_and_budget(self):
        env = side.clean_environment({'SLURM_JOB_ID': 'foreign', 'SBATCH_GRES': 'gpu:8',
                                      'SRUN_CPUS_PER_TASK': '1', 'JOB_BUDGET_END_EPOCH': '999',
                                      'PMIX_SERVER_URI': 'x', 'T6_AG_PARENT_PID': '100', 'PATH': '/bin',
                                      'CUDA_VISIBLE_DEVICES': '0'})
        self.assertEqual(env['PATH'], '/bin')
        self.assertFalse(any(k.startswith(('SLURM_', 'SBATCH_', 'SRUN_', 'JOB_BUDGET_', 'PMIX_', 'T6_AG_')) for k in env))
        self.assertTrue(all(env[k] == v for k, v in side.CPU_ENV.items()))

    def test_command_exact_cpu_exclusive_two_hour_no_parent_mutation(self):
        command = side.srun_command('206116', 'node308', self.root)
        for token in ['--jobid=206116', '--nodelist=node308', '--nodes=1', '--ntasks=4',
                      '--cpus-per-task=16', '--mem=512G', '--gpus=0', '--gres=none',
                      '--exact', '--exclusive', '--immediate=10', '--time=02:00:00']:
            self.assertIn(token, command)
        self.assertNotIn('--overlap', command)
        self.assertNotIn('sbatch', command)
        self.assertNotIn('update', command)

    def test_cpu_masks_are_applied_inside_task_after_slurm_rewrites(self):
        command = side.srun_command('206116', 'node308', self.root)
        environment_index = command.index('env')
        self.assertGreater(environment_index, command.index('--export=ALL'))
        self.assertEqual(command[environment_index:environment_index + 7],
                         ['env', 'CPU_ONLY=1', 'CUDA_VISIBLE_DEVICES=', 'HIP_VISIBLE_DEVICES=-1',
                          'ROCR_VISIBLE_DEVICES=-1', 'GPU_DEVICE_ORDINAL=-1', side.PYTHON])
        self.assertEqual(side.STAGE.name, 'table6_autogluon_sidecar_20260922_v2')

    def test_budget_caps_long_parent_and_short_parent_without_renewal(self):
        value = side.capped_budget(self.raw(), '206116', 'node308', 100, 102, 1000, 'node308')
        self.assertEqual(value['remaining_seconds'], 6898)
        env = {**value['environment'], 'SLURM_JOB_ID': '206116'}
        b = EnvironmentBudget.from_environment(env, hostname='node308', monotonic_clock=lambda: 202)
        self.assertEqual(b.remaining(), 6798)
        shorter = side.capped_budget(self.raw(TimeLimit='01:20:00'), '206116', 'node308', 100, 102, 1000, 'node308')
        self.assertEqual(shorter['remaining_seconds'], 898)

    def test_launch_detached_and_parent_claim_prohibits_repeat(self):
        proof = self.root / 'proof.json'
        proof.write_text(json.dumps(self.proof))
        with patch.object(side, 'STAGE', self.root / 'stage'), patch.object(side.time, 'time', return_value=110), \
                patch.object(side, 'command', return_value=self.raw()) as command, \
                patch.object(side, 'source_hashes', return_value={'frozen': 'sha'}), \
                patch.object(side.subprocess, 'Popen', return_value=types.SimpleNamespace(pid=123)) as spawn:
            receipt = side.launch('206116', 'node308', proof, 'first')
            self.assertEqual(receipt['pid'], 123)
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            self.assertTrue((side.STAGE / 'parents/206116.json').exists())
            with self.assertRaises(FileExistsError):
                side.launch('206116', 'node308', proof, 'second')
            self.assertEqual(spawn.call_count, 1)
            self.assertTrue(all(c.args[0][:3] == ['scontrol', 'show', 'job'] for c in command.call_args_list))

    def test_uncertain_spawn_retains_parent_intent(self):
        proof = self.root / 'proof.json'
        proof.write_text(json.dumps(self.proof))
        with patch.object(side, 'STAGE', self.root / 'stage'), patch.object(side.time, 'time', return_value=110), \
                patch.object(side, 'command', return_value=self.raw()), \
                patch.object(side, 'source_hashes', return_value={}), \
                patch.object(side.subprocess, 'Popen', side_effect=OSError('uncertain')):
            with self.assertRaises(OSError):
                side.launch('206116', 'node308', proof, 'first')
            self.assertTrue((side.STAGE / 'parents/206116.json').exists())
            with self.assertRaises(FileExistsError):
                side.launch('206116', 'node308', proof, 'retry')


if __name__ == '__main__':
    unittest.main()
