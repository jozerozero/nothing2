"""Mock/stdlib evidence tests; no remote commands and no process inspection."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import shared_gpu_capacity_probe as probe


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_environment_whitelist_never_exports_secrets(self):
        result = probe.selected_environment(b'TOKEN=secret\0SSH_AUTH_SOCK=/secret\0ROCR_VISIBLE_DEVICES=GPU-abc\0SLURM_STEP_ID=7\0PATH=/bin\0')
        self.assertEqual(result, {'ROCR_VISIBLE_DEVICES': 'GPU-abc', 'SLURM_STEP_ID': '7'})
        self.assertNotIn('secret', json.dumps(result))

    def test_atomic_output_never_overwrites(self):
        path = self.root / 'sample.json'
        probe.publish(path, {'a': 1})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            probe.publish(path, {'a': 2})
        self.assertEqual(path.read_bytes(), before)

    def parent_raw(self, **updates):
        fields = {'JobId': '196093', 'JobState': 'RUNNING', 'NumNodes': '1', 'NodeList': 'node197',
                  'UserId': f'user({os.getuid()})', 'AllocTRES': 'cpu=64,gres/gpu=8,mem=64G'}
        fields.update(updates)
        return ' '.join(f'{k}={v}' for k, v in fields.items())

    def test_exact_parent_resource_scope(self):
        self.assertEqual(probe.parent_record(self.parent_raw(), '196093')['NodeList'], 'node197')
        for update in [{'JobState': 'PENDING'}, {'NumNodes': '2'}, {'UserId': 'other(9999999)'},
                       {'AllocTRES': 'cpu=64,gres/gpu=7,mem=64G'}, {'AllocTRES': 'cpu=64,gres/gpu=8,mem=512G'}]:
            with self.assertRaises(RuntimeError):
                probe.parent_record(self.parent_raw(**update), '196093')
        with self.assertRaises(RuntimeError):
            probe.parent_record(self.parent_raw(), 'not-allowed')

    def test_readonly_step_is_tiny_zero_gpu_and_cpu_masks_post_slurm(self):
        command = probe.probe_command('196093', 'node197')
        for value in ['--ntasks=1', '--cpus-per-task=1', '--mem=1G', '--gpus=0', '--gres=none', '--time=00:02:00']:
            self.assertIn(value, command)
        self.assertLess(command.index('--export=ALL'), command.index('env'))
        self.assertIn('ROCR_VISIBLE_DEVICES=-1', command)
        self.assertNotIn('sbatch', command)

    def samples(self):
        parents = []
        for parent in probe.PARENTS:
            parents.append({'parent': parent, 'node': 'n' + parent, 'complete': True,
                            'gpus': [{'uuid': 'a', 'pci': '0000:88:00.0', 'hardware_idle': True,
                                      'same_uid_fd_owner_pids': [7], 'foreign_fd_owner_pids': []}],
                            'same_uid_rss_bytes': 100, 'all_uid_gpu_fd_inspection_complete': True,
                            'idle_parent_cpu_ids_twice': [1, 2, 3, 4]})
        first = {'parents_requested': list(probe.PARENTS), 'parents': parents, 'started_epoch': 100,
                 'epoch': 110, 'source_sha256': 'source'}
        second = copy.deepcopy(first)
        second.update(started_epoch=125, epoch=135)
        return first, second

    def test_comparison_requires_separated_same_source_samples(self):
        first, second = self.samples()
        result = probe.compare_samples(first, second)
        self.assertEqual(len(result['parents']), 6)
        self.assertEqual(len(result['parents'][0]['hardware_idle_twice']), 1)
        self.assertFalse(result['parents'][0]['ownership_proven'])
        self.assertEqual(result['parents'][0]['eligible_gpus'], [])
        second['started_epoch'] = 124
        with self.assertRaises(RuntimeError):
            probe.compare_samples(first, second)
        second['started_epoch'] = 125
        second['source_sha256'] = 'different'
        with self.assertRaises(RuntimeError):
            probe.compare_samples(first, second)

    def test_cross_host_offset_does_not_poison_login_clock_comparison(self):
        first, second = self.samples()
        for row in first['parents']:
            row['epoch'] = -100
        for row in second['parents']:
            row['epoch'] = -75
        self.assertTrue(probe.compare_samples(first, second)['parents'][0]['complete'])

    def test_permission_failure_never_produces_candidate(self):
        first, second = self.samples()
        second['parents'][0]['complete'] = False
        row = probe.compare_samples(first, second)['parents'][0]
        self.assertFalse(row['complete'])
        self.assertEqual(row['eligible_gpus'], [])

    def test_explicit_other_uuid_fds_are_driver_enumeration_not_target_mask(self):
        gpus = [{'uuid': 'a'}, {'uuid': 'b'}]
        process = {'pid': 7, 'selected_environment': {'ROCR_VISIBLE_DEVICES': 'GPU-b'},
                   'gpu_fds': [{'uuid': 'a'}, {'uuid': 'b'}], 'is_probe': False}
        result = probe.ownership_evidence(gpus[0], [process], gpus)
        self.assertEqual(result['explicit_target_mask_pids'], [])
        self.assertEqual(result['other_uuid_driver_enumeration_fd_pids'], [7])
        self.assertEqual(result['ambiguous_or_future_owner_pids'], [])

    def test_ordinal_and_unrestricted_masks_remain_ambiguous(self):
        gpus = [{'uuid': 'a'}]
        for env in ({'ROCR_VISIBLE_DEVICES': '0'}, {}):
            process = {'pid': 7, 'selected_environment': env, 'gpu_fds': [], 'is_probe': False}
            self.assertEqual(probe.ownership_evidence(gpus[0], [process], gpus)['ambiguous_or_future_owner_pids'], [7])

    def test_probe_failure_is_preserved_not_retried(self):
        with patch.object(probe, 'run', side_effect=PermissionError('denied')) as run:
            result = probe.sample_parent('196093')
        self.assertFalse(result['complete'])
        self.assertIn('PermissionError', result['error'])
        self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main()
