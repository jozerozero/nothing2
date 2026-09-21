"""CPU mocks only: operational replacement cannot cancel an unverified old job."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import tabfm_shorttmp_dispatch as runtime
import tabfm_replace_208210 as replacement


class RuntimeTests(unittest.TestCase):
    def test_genuine_recorded_four_rank_only(self):
        man, plan = {'output_root': '/tmp/fixture/campaign'}, {'plan_id': 'p'}
        env = {'SLURM_JOB_ID': '209999', 'SLURM_NTASKS': '4', 'SLURM_PROCID': '2',
               'SLURM_LOCALID': '2', 'SLURM_STEP_ID': '1'}
        with patch.object(runtime.frozen, 'read', return_value={'job_id': '209999', 'plan_id': 'p'}):
            self.assertEqual(runtime.verify_runtime_job(plan, man, 'run', env), '209999')
            for key, value in [('SLURM_JOB_ID', '208210'), ('SLURM_NTASKS', '1'),
                               ('SLURM_LOCALID', '0'), ('SLURM_PROCID', '4'), ('SLURM_STEP_ID', 'batch')]:
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    runtime.verify_runtime_job(plan, man, 'run', dict(env, **{key: value}))

    def test_controller_check_still_requires_recorded_job(self):
        with patch.object(runtime.frozen, 'read', return_value={'job_id': '209999', 'plan_id': 'p'}):
            self.assertEqual(runtime.verify_runtime_job({'plan_id': 'p'}, {'output_root': '/tmp/a/b'},
                                                        'check', {'SLURM_JOB_ID': '209999'}), '209999')
            with self.assertRaises(RuntimeError):
                runtime.verify_runtime_job({'plan_id': 'wrong'}, {'output_root': '/tmp/a/b'},
                                           'check', {'SLURM_JOB_ID': '209999'})

    def test_slurm_uses_wrapper_all_phases_with_same_resources(self):
        path = Path(runtime.__file__).with_name('tabfm_shorttmp_slurm.sh')
        content = path.read_text()
        for phase in ('preflight', 'check', 'smoke', 'check-smoke', 'run', 'status'):
            self.assertIn('tabfm_shorttmp_dispatch.py ' + phase + ' ', content)
        for option in ('--ntasks=4', '--gpus-per-task=1', '--cpus-per-task=4', '--mem=256G',
                       '--time=24:00:00', '--gpu-bind=single:1', '--kill-on-bad-exit=1'):
            self.assertIn(option, content)
        self.assertNotIn('tabfm_default_dispatch.py ', content)


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.man = {'output_root': str(Path(self.directory.name).resolve() / 'campaign'), 'manifest_id': 'm'}
        self.plan = {'plan_id': 'p'}
        self.root = runtime.runtime_root(self.man)
        self.root.mkdir(parents=True)
        self.commands = []
        self.cancel_state = 'CANCELLED'
        self.reject_new = False
        self.hold_error = False
        self.old_calls = 0
        self.old_started = False

    def command(self, args):
        self.commands.append(args)
        if args[0] == 'sbatch': return '209999;cluster'
        if args == ['scontrol', 'hold', '208210'] and self.hold_error:
            raise RuntimeError('hold failed')
        if args[:4] == ['scontrol', 'show', 'job', '208210']:
            return 'JobId=208210 JobState=' + self.cancel_state
        if args[:4] == ['scontrol', 'show', 'job', '209999']:
            return 'new raw'
        return ''

    def verify(self, job, script, raw, held):
        if self.reject_new:
            raise RuntimeError('wrong resource or owner')
        return {'JobState': 'PENDING', 'Reason': 'JobHeldUser' if held else 'Priority'}

    def old_record(self, man, held=False):
        self.old_calls += 1
        if self.old_started and self.old_calls >= 2:
            raise RuntimeError('old is now running; do not touch')
        return {'job_id': '208210', 'fields': {'JobState': 'PENDING'}, 'raw': 'old raw'}

    def run_replace(self):
        with patch.object(replacement.frozen, 'load_campaign', return_value=(self.man, [])), \
             patch.object(replacement, 'load_plan', return_value=(self.plan, self.man, [])), \
             patch.object(replacement, 'old_record', side_effect=self.old_record), \
             patch.object(replacement, 'command', side_effect=self.command), \
             patch.object(replacement.original, 'verify', side_effect=self.verify):
            replacement.replace(Path('/fixture/campaign.json'))

    def test_new_held_verified_before_old_hold_cancel_and_release(self):
        self.run_replace()
        submit = next(i for i, c in enumerate(self.commands) if c[0] == 'sbatch')
        verify = self.commands.index(['scontrol', 'show', 'job', '209999', '-o'])
        hold = self.commands.index(['scontrol', 'hold', '208210'])
        cancel = self.commands.index(['scancel', '208210'])
        release = self.commands.index(['scontrol', 'release', '209999'])
        self.assertLess(submit, verify)
        self.assertLess(verify, hold)
        self.assertLess(hold, cancel)
        self.assertLess(cancel, release)
        self.assertTrue((self.root / 'old_cancelled.json').is_file())
        self.assertEqual(json.loads((self.root / 'released_new.json').read_text())['job_id'], '209999')
        self.assertEqual([c for c in self.commands if c[0] == 'scancel'], [['scancel', '208210']])

    def test_new_validation_failure_leaves_old_untouched_and_candidate_held(self):
        self.reject_new = True
        with self.assertRaises(RuntimeError): self.run_replace()
        self.assertFalse(any(c[0] == 'scancel' or c[:2] in (['scontrol', 'hold'], ['scontrol', 'release'])
                             for c in self.commands))
        self.assertTrue((self.root / 'submitted_new.json').exists())
        self.assertEqual(json.loads((self.root / 'transaction_error.json').read_text())['stage'], 'verify_new_held')

    def test_old_start_race_aborts_without_any_cancellation(self):
        self.old_started = True
        with self.assertRaises(RuntimeError): self.run_replace()
        self.assertFalse(any(c[0] == 'scancel' or c[:2] == ['scontrol', 'release'] for c in self.commands))

    def test_failed_hold_never_cancels_running_job_or_releases_candidate(self):
        self.hold_error = True
        with self.assertRaises(RuntimeError): self.run_replace()
        self.assertFalse(any(c[0] == 'scancel' or c[:2] == ['scontrol', 'release'] for c in self.commands))

    def test_unconfirmed_cancellation_keeps_replacement_held(self):
        self.cancel_state = 'PENDING'
        with self.assertRaises(RuntimeError): self.run_replace()
        self.assertNotIn(['scontrol', 'release', '209999'], self.commands)
        self.assertFalse((self.root / 'old_cancelled.json').exists())

    def test_immutable_journal_forbids_retry_after_failure(self):
        self.reject_new = True
        with self.assertRaises(RuntimeError): self.run_replace()
        count = len(self.commands)
        with self.assertRaises(RuntimeError): self.run_replace()
        self.assertEqual(len(self.commands), count)


if __name__ == '__main__':
    unittest.main()
