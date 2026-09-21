"""Read-only/mock submission tests; never invoke sbatch or remote commands."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import table6_restart_ag_ops as ops
import table6_restart_deadline as deadline


class AGOpsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for key, path in [('STAGE', self.root / 'stage'), ('LOGS', self.root / 'logs'),
                          ('BASE', self.root / 'original-stage'), ('OUT', self.root / 'old-output'),
                          ('LEDGER', self.root / 'stage/submission.json')]:
            self.stack.enter_context(patch.object(ops, key, path))
        self.commands = []

    def fields(self, **updates):
        values = {'JobId': '123', 'JobName': 't6ag22cpu', 'Partition': 'faculty',
                  'Account': 'faculty-acc', 'QOS': 'bgqos', 'NumTasks': '4', 'CPUs/Task': '16',
                  'NumCPUs': '64', 'TimeLimit': '02:00:00', 'MinMemoryNode': '512G',
                  'Nice': '0', 'Requeue': '0', 'Dependency': '(null)', 'WorkDir': str(ops.STAGE),
                  'Command': str(ops.STAGE / 'run.sh'), 'StdOut': str(ops.LOGS / 'slurm-123.out'),
                  'StdErr': str(ops.LOGS / 'slurm-123.err'), 'JobState': 'PENDING', 'Reason': 'JobHeldUser',
                  'NumNodes': '1', 'NtasksPerN:B:S:C': '4:0:*:*', 'ReqTRES': 'cpu=64,mem=512G,node=1',
                  'ExcNodeList': ops.EXCLUDE, 'TresPerTask': 'cpu=16', 'TresPerNode': '(null)'}
        values.update(updates)
        return values

    def fake_run(self, args, **kwargs):
        self.commands.append(args)
        if args[:3] == ['scontrol', 'show', 'job']:
            return ' '.join(f'{k}={v}' for k, v in self.fields().items())
        if args[:3] == ['scontrol', 'show', 'hostnames']:
            return 'excluded-node'
        if args[:3] == ['scontrol', 'write', 'batch_script']:
            Path(args[-1]).write_text(ops.script())
            return ''
        if args[0] == 'sbatch':
            return '123;cluster'
        if args[0] == 'git':
            return 'source-commit'
        return ''

    def test_script_exact_cpu_contract_and_syntax(self):
        script = ops.script()
        for text in ('#SBATCH --qos=bgqos', '#SBATCH --nodes=1', '#SBATCH --ntasks=4',
                     '#SBATCH --ntasks-per-node=4', '#SBATCH --cpus-per-task=16',
                     '#SBATCH --gpus=0', '#SBATCH --mem=512G', '#SBATCH --time=02:00:00',
                     '--gpus-per-task=0 --gres=none', 'env CPU_ONLY=1 CUDA_VISIBLE_DEVICES=',
                     'HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1',
                     'nice -n 19 ionice -c 3', 'table6_restart_ag.py launch',
                     'table6_restart_deadline.py --job-id "$SLURM_JOB_ID" --format exports',
                     'eval "$BUDGET_EXPORTS"'):
            self.assertIn(text, script)
        self.assertEqual(subprocess.run(['bash', '-n'], input=script, text=True, capture_output=True).returncode, 0)

    def test_deadline_exports_reach_child_environment(self):
        result = deadline.derive_deadline(
            'JobId=123 JobState=RUNNING TimeLimit=02:00:00 RunTime=00:00:10 EndTime=2026-09-22T12:00:00',
            job_id='123', query_started_monotonic=100, query_finished_monotonic=101,
            observed_local_epoch=1000, hostname='node')
        output = io.StringIO()
        with patch.object(deadline, 'query_deadline', return_value=result), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(deadline.main(['--format', 'exports']), 0)
        env = dict(os.environ, T6_TEST_EXPORTS=output.getvalue())
        completed = subprocess.run(['bash', '-c', 'eval "$T6_TEST_EXPORTS"; env'],
                                   env=env, text=True, capture_output=True, check=True)
        for key, value in result.environment().items():
            self.assertIn(f'{key}={value}', completed.stdout.splitlines())

    def test_typed_and_per_task_gpus_rejected(self):
        ops.require_no_gpus({'ReqTRES': 'cpu=64,gres/gpu=0', 'TresPerTask': 'gres/gpu:0'})
        for field, value in [('ReqTRES', 'cpu=64,gres/gpu:mi210=1'),
                             ('AllocTRES', 'gres/gpu=8'), ('TresPerNode', 'gres/gpu:mi210:1'),
                             ('TresPerTask', 'gres/gpu:1'), ('Gres', 'gpu:mi210:8')]:
            with self.subTest(field=field), self.assertRaises(AssertionError):
                ops.require_no_gpus({field: value})

    def test_control_checks_held_identity_and_spooled_script(self):
        ops.STAGE.mkdir()
        with patch.object(ops, 'run', side_effect=self.fake_run):
            result = ops.control('123')
        self.assertEqual(result['JobState'], 'PENDING')
        self.assertEqual((ops.STAGE / 'spool-123.sh').read_text(), ops.script())

    def test_control_rejects_changed_resources_and_duplicate_identity(self):
        for update in [{'NumCPUs': '32'}, {'QOS': 'normal'}, {'TimeLimit': '04:00:00'},
                       {'MinMemoryNode': '256G'}, {'JobState': 'RUNNING'},
                       {'ReqTRES': 'cpu=64,gres/gpu:mi210=1'}]:
            def run(args, **kwargs):
                return ' '.join(f'{k}={v}' for k, v in self.fields(**update).items())
            with patch.object(ops, 'run', side_effect=run), self.assertRaises(AssertionError):
                ops.control('123')
        raw = ' '.join(f'{k}={v}' for k, v in self.fields().items()) + ' JobId=123'
        with patch.object(ops, 'run', return_value=raw), self.assertRaises(deadline.DeadlineError):
            ops.control('123')

    def test_submit_is_held_first_no_release_and_preserves_old_output(self):
        ops.OUT.mkdir()
        old = ops.OUT / 'existing.json'
        old.write_text('immutable prior result\n')
        before = old.read_bytes(), old.stat().st_mtime_ns
        with patch.object(ops, 'run', side_effect=self.fake_run), \
                patch.object(ops, 'precheck', return_value={'absent_aggregate_files': 295}), \
                patch.object(ops, 'sources', return_value={'worker': 'sha'}):
            result = ops.main('submit')
        self.assertEqual(result['state'], 'verified_held')
        submits = [args for args in self.commands if args[0] == 'sbatch']
        self.assertEqual(len(submits), 1)
        self.assertIn('--hold', submits[0])
        self.assertFalse(any(args[:2] == ['scontrol', 'release'] for args in self.commands))
        self.assertEqual((old.read_bytes(), old.stat().st_mtime_ns), before)
        with patch.object(ops, 'run') as call, self.assertRaises(AssertionError):
            ops.main('submit')
        call.assert_not_called()

    def test_uncertain_sbatch_is_not_retried_or_released(self):
        def run(args, **kwargs):
            if args[0] == 'sbatch':
                raise RuntimeError('uncertain transport')
            return self.fake_run(args, **kwargs)
        with patch.object(ops, 'run', side_effect=run), \
                patch.object(ops, 'precheck', return_value={'absent_aggregate_files': 295}), \
                patch.object(ops, 'sources', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                ops.main('submit')
        self.assertEqual(json.loads(ops.LEDGER.read_text())['state'], 'submitting')
        with patch.object(ops, 'run') as call, self.assertRaises(AssertionError):
            ops.main('submit')
        call.assert_not_called()

    def test_release_revalidates_held_before_release(self):
        ops.STAGE.mkdir()
        ops.LEDGER.write_text(json.dumps({'state': 'verified_held', 'job_id': '123'}))
        with patch.object(ops, 'verify') as verify, patch.object(ops, 'run', side_effect=self.fake_run):
            self.assertEqual(ops.main('release')['state'], 'released')
        verify.assert_called_once()
        calls = [args[:3] for args in self.commands]
        self.assertLess(calls.index(['scontrol', 'show', 'job']), calls.index(['scontrol', 'release', '123']))


if __name__ == '__main__':
    unittest.main()
