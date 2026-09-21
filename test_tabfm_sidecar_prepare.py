import copy
import unittest
from unittest.mock import patch
import tabfm_sidecar_prepare as p


def snapshot(epoch):
    return {'parent': p.PARENT, 'node': p.NODE, 'epoch': epoch,
            'job_fields': {'JobId': p.PARENT, 'JobState': 'RUNNING',
               'UserId': 'guangyi.chen(2012)', 'NumNodes': '1', 'NodeList': p.NODE,
               'NumCPUs': '64', 'AllocTRES': 'cpu=64,mem=64G,node=1,gres/gpu=8',
               'EndTime': '2026-09-22T17:01:50'},
            'gpus': [{'uuid': p.UUID, 'pci': p.PCI, 'busy': 0, 'vram': 13631488}],
            'owned_processes': [{'rss': 14 * 1024**3}],
            'cpu_sample': {'epoch': epoch, 'processes': [
                {'pid': 100, 'created': 1000.0, 'cpu_seconds': epoch * 12}]}}


class PrepareTests(unittest.TestCase):
    def test_parent_accepts_only_exact_owned_idle_gpu(self):
        p.check_parent(snapshot(1790025913))
        for key, value in [('JobState', 'PENDING'), ('NodeList', 'auh7-1b-gpu-195'),
                           ('AllocTRES', 'cpu=64,mem=64G,gres/gpu=4')]:
            row = snapshot(1790025913)
            row['job_fields'][key] = value
            with self.assertRaises(RuntimeError):
                p.check_parent(row)

    def test_gpu_or_memory_not_idle(self):
        for field, value in [('busy', 1), ('vram', 256 * 1024**2), ('pci', '0000:47:00.0')]:
            row = snapshot(1790025913)
            row['gpus'][0][field] = value
            with self.assertRaises(RuntimeError):
                p.check_parent(row)
        row = snapshot(1790025913)
        row['owned_processes'][0]['rss'] = 21 * 1024**3
        with self.assertRaises(RuntimeError):
            p.check_parent(row)

    def make_plan(self, first, second, now):
        writes = []
        with patch.object(p, 'read', side_effect=[first, second]), \
             patch.object(p.time, 'time', return_value=now), \
             patch.object(p, 'identity', return_value={'path': '/example/sidecar.py', 'sha256': 'test'}), \
             patch.object(p, 'atomic', side_effect=lambda *a: writes.append(a)):
            p.prepare({'output_root': str(p.OUT)}, p.OUT / 'sidecars/test')
        return writes[0][2]

    def test_complete_plan_has_operational_limits(self):
        plan = self.make_plan(snapshot(1790025913), snapshot(1790025943), 1790025950)
        self.assertEqual(plan['cpu_count'], 4)
        self.assertEqual(plan['mem_gib'], 40)
        self.assertEqual(plan['free_cpu_cores'], 48)
        self.assertEqual(plan['plan_id'], p.digest({k:v for k,v in plan.items() if k != 'plan_id'}))

    def test_close_or_stale_observations_fail(self):
        for delta, now in [(10,1790025930), (30,1790026200)]:
            with self.assertRaises(RuntimeError):
                self.make_plan(snapshot(1790025913), snapshot(1790025913+delta), now)

    def test_cpu_pressure_fails(self):
        a, b = snapshot(1790025913), snapshot(1790025943)
        b['cpu_sample']['processes'][0]['cpu_seconds'] = a['cpu_sample']['processes'][0]['cpu_seconds'] + 60*30
        with self.assertRaises(RuntimeError):
            self.make_plan(a, b, 1790025950)


if __name__ == '__main__':
    unittest.main()
