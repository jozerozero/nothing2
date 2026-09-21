"""CPU mocks only: no Slurm submission, remote command, or model fit."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import table6_missing190_ops as ops
import table6_restart_deadline as deadline


class OpsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='missing190-ops-test-')
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.root, self.stage = root, root/'stage'/ops.NAME
        self.out, self.logs = root/'evaluation'/ops.NAME, root/'logs'/ops.NAME
        self.stage.mkdir(parents=True)
        values = {'ROOT': root, 'STAGE': self.stage, 'OUT': self.out, 'LOGS': self.logs,
                  'PLAN': self.out/'plan.json', 'LEDGER': self.stage/'submission.json'}
        for key, value in values.items():
            context = patch.object(ops, key, value)
            context.start(); self.addCleanup(context.stop)
        self.plan = {'plan_id': 'new-plan'}
        for key, value in [('plan', Mock(return_value=self.plan)),
                           ('sources', Mock(return_value={'new.py': 'hash'}))]:
            context = patch.object(ops, key, value)
            context.start(); self.addCleanup(context.stop)
        self.calls = []

    def fields(self, **updates):
        values = {'JobId': '123', 'JobName': 't6gap190q', 'JobState': 'PENDING', 'Reason': 'JobHeldUser',
                  'Partition': 'faculty', 'Account': 'faculty-acc', 'QOS': 'bgqos', 'NumTasks': '1',
                  'NumCPUs': '16', 'CPUs/Task': '16', 'TimeLimit': '02:00:00', 'MinMemoryNode': '256G',
                  'Nice': '0', 'Requeue': '0', 'Command': str(self.stage/'run.sh'), 'WorkDir': str(self.stage),
                  'StdOut': str(self.logs/'slurm-123.out'), 'StdErr': str(self.logs/'slurm-123.err'),
                  'NumNodes': '1', 'ReqTRES': 'cpu=16,mem=256G,gres/gpu=1', 'TresPerTask': 'gres/gpu=1',
                  'Dependency': f'afterany:{ops.PREDECESSOR}(unfulfilled)', 'ExcNodeList': ops.EXCLUDE}
        values.update(updates)
        return ' '.join(f'{key}={value}' for key, value in values.items())

    def fake_run(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[:4] == ['scontrol', 'show', 'job', '-o']:
            if argv[4] == ops.PREDECESSOR:
                return (f'JobId={ops.PREDECESSOR} JobName=t6r22g01 JobState=RUNNING '
                        f'WorkDir={self.root}/stage/table6_completion_restart_20260922_v1')
            return self.fields()
        if argv[:3] == ['scontrol', 'show', 'hostnames']:
            return 'node193\nnode195'
        if argv[:3] == ['scontrol', 'write', 'batch_script']:
            Path(argv[4]).write_text(ops.script()); return ''
        if argv[0] == 'squeue': return ''
        if argv[0] == 'git': return 'commit'
        if argv[0] == 'bash': return ''
        if argv[0] == 'sbatch':
            self.assertEqual(json.loads(ops.LEDGER.read_text())['state'], 'submitting')
            self.assertIn('--hold', argv)
            return '123;cluster'
        if argv[:2] == ['scontrol', 'release']:
            self.assertEqual(json.loads(ops.LEDGER.read_text())['state'], 'releasing')
            return ''
        raise AssertionError(argv)

    def test_script_resource_budget_and_source_gates(self):
        text = ops.script()
        for token in ('--nodes=1', '--ntasks=1', '--cpus-per-task=16', '--gpus-per-task=1',
                      '--mem=256G', '--time=02:00:00', '--nice=0', '--no-requeue', '--signal=USR1@90',
                      f'--dependency=afterany:{ops.PREDECESSOR}', '--gpu-bind=single:1'):
            self.assertIn(token, text)
        self.assertLess(text.index('table6_restart_deadline.py'), text.index('manage.py check-inputs'))
        self.assertLess(text.index('manage.py check-inputs'), text.index('srun --exact'))
        parsed = subprocess.run(['bash', '-n'], input=text, text=True, capture_output=True)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)

    def test_submit_held_verify_release_and_no_duplicate(self):
        with patch.object(ops, 'run', self.fake_run):
            state = ops.main('submit')
            self.assertEqual(state['state'], 'verified_held')
            self.assertFalse(any(call[:2] == ['scontrol', 'release'] for call in self.calls))
            ops.verify()
            with self.assertRaisesRegex(AssertionError, 'no duplicate'):
                ops.main('submit')
            state = ops.main('release')
            self.assertEqual(state['state'], 'released')
            with self.assertRaises(AssertionError): ops.main('release')

    def test_uncertain_submit_retains_ledger_and_cannot_retry(self):
        def failed(argv, **kwargs):
            if argv[0] == 'sbatch': raise RuntimeError('uncertain transport')
            return self.fake_run(argv, **kwargs)
        with patch.object(ops, 'run', failed):
            with self.assertRaisesRegex(RuntimeError, 'uncertain'): ops.main('submit')
            self.assertEqual(json.loads(ops.LEDGER.read_text())['state'], 'submitting')
            with self.assertRaisesRegex(AssertionError, 'no duplicate'): ops.main('submit')

    def test_existing_script_never_overwritten(self):
        (self.stage/'run.sh').write_text('preserve')
        with patch.object(ops, 'run', self.fake_run), self.assertRaises(FileExistsError):
            ops.main('submit')
        self.assertEqual((self.stage/'run.sh').read_text(), 'preserve')
        self.assertFalse(ops.LEDGER.exists())

    def test_wrong_predecessor_or_duplicate_queue_rejected_before_submit(self):
        for which in ('predecessor', 'queue'):
            def changed(argv, **kwargs):
                if which == 'predecessor' and argv[:4] == ['scontrol', 'show', 'job', '-o']:
                    return f'JobId={ops.PREDECESSOR} JobName=unrelated'
                if which == 'queue' and argv[0] == 'squeue': return '123|t6gap190q|elsewhere'
                return self.fake_run(argv, **kwargs)
            with patch.object(ops, 'run', changed), self.assertRaises(AssertionError): ops.main('submit')
            self.assertFalse(ops.LEDGER.exists())

    def test_held_resources_dependency_and_spool_are_exact(self):
        (self.stage/'spool-123.sh').write_text(ops.script())
        for changed in ({'Reason': 'Priority'}, {'JobState': 'RUNNING'}, {'NumCPUs': '8'},
                        {'MinMemoryNode': '128G'}, {'Dependency': 'afterany:999(unfulfilled)'},
                        {'ReqTRES': 'cpu=16,gres/gpu=8'}, {'Requeue': '1'}, {'Nice': '1'}):
            def read(argv, **kwargs):
                if argv[:4] == ['scontrol', 'show', 'job', '-o']: return self.fields(**changed)
                return self.fake_run(argv, **kwargs)
            with patch.object(ops, 'run', read), self.assertRaises(AssertionError): ops.control('123')
        (self.stage/'spool-123.sh').write_text('modified')
        with patch.object(ops, 'run', self.fake_run), self.assertRaises(AssertionError): ops.control('123')

    def test_runtime_source_change_blocks_release(self):
        with patch.object(ops, 'run', self.fake_run): ops.main('submit')
        with patch.object(ops, 'sources', return_value={'new.py': 'changed'}), \
                self.assertRaisesRegex(AssertionError, 'source changed'):
            ops.main('release')
        self.assertEqual(json.loads(ops.LEDGER.read_text())['state'], 'verified_held')


class NodeTests(unittest.TestCase):
    def node(self, *, remaining=6000, signal_during_wait=None, failed=False):
        handlers, children, launches = {}, [], []
        class Child:
            def __init__(self, argv):
                launches.append(argv); children.append(self)
                self.returncode = None; self.signals = []; self.reaped = False
            def poll(self): return self.returncode
            def send_signal(self, signum): self.signals.append(signum)
            def wait(self, timeout=None):
                if signal_during_wait is not None: handlers[signal_during_wait](signal_during_wait, None)
                self.returncode = 2 if failed else 0
                self.reaped = True
                return self.returncode
        result = None
        with patch.object(ops, 'verify'), patch.dict(os.environ, {'SLURM_PROCID': '0', 'SLURM_NTASKS': '1'}), \
                patch.object(deadline.EnvironmentBudget, 'from_environment', return_value=types.SimpleNamespace(remaining=lambda: remaining)), \
                patch.object(ops.signal, 'signal', side_effect=lambda signum, handler: handlers.__setitem__(signum, handler)), \
                patch.object(ops.subprocess, 'Popen', side_effect=Child), \
                patch.object(ops.os, 'execv', side_effect=lambda executable, argv: launches.append(('exec', executable, argv))):
            if failed:
                with self.assertRaises(subprocess.CalledProcessError): ops.main('node')
            else:
                # A mocked execv returns; production execv replaces this process.
                try: result = ops.main('node')
                except ValueError as exc:
                    if str(exc) != 'node': raise
        return result, children, launches

    def test_nine_smokes_then_preflight_gate_then_formal_exec(self):
        _, children, launches = self.node()
        self.assertEqual(len(children), 11)
        self.assertTrue(all(child.reaped for child in children))
        self.assertEqual([argv[3] for argv in launches[:11]], ['smoke']*9+['preflight', 'gate'])
        self.assertEqual(len({argv[-1] for argv in launches[:9]}), 9)
        self.assertEqual(launches[-1][0], 'exec')
        self.assertIn('worker', launches[-1][2])

    def test_low_budget_never_launches_smoke_or_formal(self):
        result, children, launches = self.node(remaining=180)
        self.assertEqual(result['state'], 'paused_before_formal')
        self.assertEqual((children, launches), ([], []))

    def test_signal_forwarded_child_reaped_and_no_formal(self):
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            result, children, launches = self.node(signal_during_wait=signum)
            self.assertEqual(result['state'], 'paused_before_formal')
            self.assertEqual(len(children), 1)
            self.assertEqual(children[0].signals, [signum])
            self.assertTrue(children[0].reaped)
            self.assertEqual(len(launches), 1)

    def test_failed_smoke_never_launches_formal(self):
        _, children, launches = self.node(failed=True)
        self.assertEqual(len(children), 1)
        self.assertTrue(children[0].reaped)
        self.assertEqual(len(launches), 1)


if __name__ == '__main__':
    unittest.main()
