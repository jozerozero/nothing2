import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import foundation_alloc8 as f


def fixture():
    mapping = {'mapping_id': 'map', 'job': '999', 'node': 'node',
               'gpus': [{'uuid': f'{i+1:016x}', 'pci': f'0000:{i+5:02x}:00.0'} for i in range(8)]}
    plan = {'plan_id': 'plan', 'worker_python': '/python', '_path': '/plan.json', 'runtime_root': '/root/runtime'}
    records = [dict(plan_id='plan', mapping_id='map', job='999', node='node', rank=i,
                    uuid=g['uuid'], pci=g['pci'], runtime_visible_count=1, actual_cpu_count=4,
                    cpu_affinity=list(range(i*4, (i+1)*4)), step='1', phase='preflight')
               for i,g in enumerate(mapping['gpus'])]
    return plan, mapping, records


class RuntimeTests(unittest.TestCase):
    def test_eight_gpu_unified_gate_and_duplicate_rejection(self):
        plan, mapping, records = fixture()
        self.assertEqual(len(f.eight_records(records, mapping, plan)), 8)
        for changed in [records[:4], records[:-1]+[records[0]],
                        [dict(r, runtime_visible_count=8) if r['rank'] == 1 else r for r in records],
                        [dict(r, uuid=records[0]['uuid']) if r['rank'] == 1 else r for r in records]]:
            with self.assertRaises(RuntimeError): f.eight_records(changed, mapping, plan)

    def test_steps_inherit_eight_then_bind_uuid_not_per_task_gres(self):
        plan, _, _ = fixture()
        boot = f.step_command(plan, 'bootstrap')
        self.assertIn('--ntasks=1', boot); self.assertIn('--cpus-per-task=64', boot)
        for mode in ('preflight', 'smoke', 'run'):
            cmd = f.step_command(plan, mode)
            self.assertIn('--ntasks=8', cmd); self.assertIn('--gpus=8', cmd)
            self.assertIn('--cpus-per-task=4', cmd); self.assertIn('--gpu-bind=none', cmd)
            self.assertTrue(any(x.endswith('allocated_gpu_uuid.py') for x in cmd))
            self.assertFalse(any(x.startswith('--gpus-per-task') for x in cmd))

    def test_memory_and_deadline_guard(self):
        budget = mock.Mock(); budget.remaining.return_value = 600
        good = dict(own_tree_rss_bytes=1, allocation_rss_bytes=1, allocation_cgroup_bytes=1, node_available_bytes=100*f.GIB)
        with mock.patch.object(f.full.base, 'STOP', None):
            f.resource_guard(good, budget)
            for changed in ({'own_tree_rss_bytes':29*f.GIB}, {'allocation_rss_bytes':225*f.GIB},
                            {'allocation_cgroup_bytes':240*f.GIB}, {'node_available_bytes':7*f.GIB}):
                with self.assertRaises(f.full.base.OperationalDeferral): f.resource_guard({**good, **changed}, budget)
            budget.remaining.return_value = 59
            with self.assertRaises(f.full.base.OperationalDeferral): f.resource_guard(good, budget)

    def test_runtime_guard_is_scoped_not_frozen_edit(self):
        original = f.full.base.snapshot, f.full.base.guard
        with f.allocation_guards():
            self.assertIs(f.full.base.snapshot, f.allocation_snapshot)
            self.assertIs(f.full.base.guard, f.resource_guard)
        self.assertEqual(original, (f.full.base.snapshot, f.full.base.guard))

    def test_gate_rejects_cpu_overlap_or_different_step(self):
        plan, mapping, records = fixture()
        for change in ({'cpu_affinity': records[0]['cpu_affinity']}, {'step': '2'}, {'phase': 'smoke'}):
            changed = copy.deepcopy(records); changed[1].update(change)
            with self.assertRaises(RuntimeError): f.eight_records(changed, mapping, plan)

    def test_job_cgroup_cache_counter_fails_closed(self):
        text = '0::/system.slice/slurmstepd.scope/job_999/step_5/user/task_0\n'
        with mock.patch.object(Path, 'read_text', side_effect=[text, '1234\n']) as reader:
            self.assertEqual(f.job_cgroup_memory('999'), 1234)
            self.assertEqual(reader.call_count, 2)
        with mock.patch.object(Path, 'read_text', return_value='0::/unrelated\n'):
            with self.assertRaises(RuntimeError): f.job_cgroup_memory('999')
        with mock.patch.object(Path, 'read_text', side_effect=[text, PermissionError('denied')]):
            with self.assertRaises(PermissionError): f.job_cgroup_memory('999')

    def test_full681_each_protocol_no_shape_exclusion(self):
        tasks = [dict(task_kind='regression' if i >= 457 else 'classification', dataset_index=i, dataset='d'+str(i),
                      row={'work_size': i+1}) for i in range(681)]
        campaigns = [(Path('/official'), {}, tasks), (Path('/budget'), {}, tasks)]
        ordered = list(f.pending_order(campaigns))
        self.assertEqual(len(ordered), 1362)
        self.assertEqual([x[0][0] for x in ordered[:4]], [Path('/official'), Path('/budget')]*2)
        self.assertIs(ordered[0][1], tasks[0])

    def test_failed_smoke_prevents_formal_launch(self):
        plan, _, _ = fixture(); plan.update(family='tabswift', plan_id='id')
        calls = []
        def run(cmd, **kwargs): calls.append(cmd)
        with mock.patch.dict(f.os.environ, {'SLURM_JOB_ID':'999', 'SLURM_JOB_NUM_NODES':'1'}, clear=True), \
             mock.patch.object(f, 'atomic'), mock.patch.object(f.subprocess, 'run', side_effect=run), \
             mock.patch.object(f, 'check_preflight'), mock.patch.object(f, 'check_smoke', side_effect=RuntimeError('smoke failed')):
            with self.assertRaisesRegex(RuntimeError, 'smoke failed'):
                f.launch_job(plan, [], Path('/plan.json'), 'tabswift')
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[-1][-3:] == ['smoke', '--plan', '/plan.json'])

    def test_slurm_contracts(self):
        for family, job in [('tabfm', 'fm681g8'), ('tabswift', 'swift2g8')]:
            source = Path(f.__file__).with_name(family+'_alloc8_slurm.sh').read_text()
            for line in ('#SBATCH --nodes=1', '#SBATCH --ntasks=8', '#SBATCH --cpus-per-task=8',
                         '#SBATCH --gpus=8', '#SBATCH --mem=256G', '#SBATCH --time=72:00:00',
                         '#SBATCH --qos=bgqos', '#SBATCH --nice=0', '#SBATCH --no-requeue', '#SBATCH --job-name='+job):
                self.assertIn(line, source)
            self.assertNotIn('--gpus-per-task', source)

    def test_generated_run_sh_fixed_path_and_exclusive_publication(self):
        source = Path(f.__file__).with_name('tabfm_alloc8_slurm.sh').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'plan.json'
            generated = f.render_run_script(source, path)
            self.assertIn('#SBATCH --chdir='+tmp, generated)
            self.assertIn('#SBATCH --output='+tmp+'/slurm-%j.out', generated)
            self.assertIn('[[ $# == 0 ]]', generated)
            self.assertNotIn('ALLOC8_PLAN="$1"', generated)
            script = Path(tmp)/'run.sh'
            f.publish_text(script, generated)
            with self.assertRaises(FileExistsError): f.publish_text(script, 'overwrite')
            self.assertEqual(script.read_text(), generated)


if __name__ == '__main__': unittest.main()
