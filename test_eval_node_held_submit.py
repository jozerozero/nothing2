"""Read-only-source review via fake Slurm; only temporary fixture files written."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import eval_node_held_submit as submitter


class SubmitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='held-node-submit-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.stage = self.root/'stage/new-gap'
        self.stage.mkdir(parents=True)
        self.script = self.stage/'run.sh'
        self.script.write_text(f'''#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=faculty-acc
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --no-requeue
#SBATCH --nice=0
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --gpus=8
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --chdir={self.stage}
#SBATCH --output={self.root}/logs/slurm-%j.out
#SBATCH --error={self.root}/logs/slurm-%j.err
set -euo pipefail
true
''')
        self.fields = {'JobId':'999','JobName':'t6g22d','Account':'faculty-acc', 'Partition':'faculty',
            'QOS':'bgqos','NumCPUs':'128','MinMemoryNode':'512G','Nice':'0','Requeue':'0','Dependency':'(null)',
            'TimeLimit':'02:00:00','WorkDir':str(self.stage),'Command':str(self.script),'NumTasks':'8',
            'CPUs/Task':'16','NumNodes':'1','ReqTRES':'cpu=128,mem=512G,gres/gpu=8',
            'StdOut':str(self.root/'logs/slurm-999.out'),'StdErr':str(self.root/'logs/slurm-999.err'),
            'UserId':'fixture('+str(os.getuid())+')','JobState':'PENDING','Reason':'JobHeldUser'}
        self.calls = []
        self.queue = ''
        self.sbatch_result = '999;cluster'
        self.spool_override = None
        self.root_patch = patch.object(submitter, 'ROOT', self.root)
        self.root_patch.start(); self.addCleanup(self.root_patch.stop)
        self.command_patch = patch.object(submitter, 'command', side_effect=self.command)
        self.command_patch.start(); self.addCleanup(self.command_patch.stop)

    @property
    def ledger(self):
        return self.stage/'node_submission.json'

    def command(self, argv):
        self.calls.append(list(argv))
        if argv[0] == 'bash': return ''
        if argv[0] == 'squeue': return self.queue
        if argv[0] == 'git': return 'source-commit'
        if argv[0] == 'sbatch':
            self.assertEqual(json.loads(self.ledger.read_text())['state'], 'submission_intent')
            self.assertIn('--hold', argv)
            return self.sbatch_result
        if argv[:3] == ['scontrol','show','job']:
            return ' '.join(k+'='+v for k,v in self.fields.items())
        if argv[:3] == ['scontrol','write','batch_script']:
            Path(argv[-1]).write_text(self.script.read_text() if self.spool_override is None else self.spool_override)
            return ''
        if argv[:2] == ['scontrol','release']:
            self.assertEqual(json.loads(self.ledger.read_text())['state'], 'release_intent')
            self.fields['Reason'] = 'Priority'
            return ''
        self.fail('unexpected command '+repr(argv))

    def submit(self):
        return submitter.submit(self.script, 'gap190', 't6g22d')

    def test_held_submission_records_id_before_validation_and_never_releases(self):
        result = self.submit()
        self.assertEqual(result['state'], 'verified_held')
        self.assertEqual(result['job_id'], '999')
        self.assertFalse(any(call[:2] == ['scontrol','release'] for call in self.calls))
        self.assertEqual((self.stage/'verified-spool-999.sh').read_text(), self.script.read_text())

    def test_release_revalidates_before_action_and_persists_intent(self):
        self.submit()
        self.calls.clear()
        result = submitter.release(self.script)
        self.assertEqual(result['state'], 'released')
        release_index = next(i for i,c in enumerate(self.calls) if c[:2] == ['scontrol','release'])
        self.assertTrue(any(c[:3] == ['scontrol','show','job'] for c in self.calls[:release_index]))
        self.assertTrue(any(c[:3] == ['scontrol','show','job'] for c in self.calls[release_index+1:]))

    def test_existing_submission_ledger_blocks_duplicate(self):
        self.submit()
        with self.assertRaisesRegex(RuntimeError, 'existing submission intent'): self.submit()
        self.assertEqual(sum(c[0] == 'sbatch' for c in self.calls), 1)

    def test_same_name_or_stage_queue_blocks_before_ledger(self):
        for queue in ('77|t6g22d|/another/stage', '78|other|'+str(self.stage)):
            self.queue = queue
            with self.assertRaisesRegex(RuntimeError, 'same-name/stage'): self.submit()
        self.assertFalse(self.ledger.exists())
        self.assertFalse(any(c[0] == 'sbatch' for c in self.calls))

    def test_uncertain_sbatch_response_retains_intent_and_refuses_retry(self):
        self.sbatch_result = 'unparseable backend response'
        with self.assertRaisesRegex(RuntimeError, 'unknown sbatch result'): self.submit()
        self.assertEqual(json.loads(self.ledger.read_text())['state'], 'submission_intent')
        with self.assertRaisesRegex(RuntimeError, 'existing submission intent'): self.submit()

    def test_resource_mismatch_retains_exact_held_job_id(self):
        self.fields['ReqTRES'] = 'cpu=128,mem=512G,gres/gpu=4'
        with self.assertRaisesRegex(RuntimeError, 'eight allocated GPUs'): self.submit()
        result = json.loads(self.ledger.read_text())
        self.assertEqual((result['state'], result['job_id']), ('submitted_held','999'))
        self.assertFalse(any(c[:2] == ['scontrol','release'] for c in self.calls))

    def test_submitted_dependency_is_rejected_while_held(self):
        self.fields['Dependency'] = 'afterany:208377(unfulfilled)'
        with self.assertRaisesRegex(RuntimeError, 'Dependency'): self.submit()
        self.assertEqual(json.loads(self.ledger.read_text())['job_id'], '999')

    def test_wrong_owner_is_rejected(self):
        self.fields['UserId'] = 'foreign('+str(os.getuid()+1)+')'
        with self.assertRaisesRegex(RuntimeError, 'wrong owner'): self.submit()

    def test_log_or_spool_mismatch_cannot_release(self):
        self.fields['StdOut'] = '/unexpected/log'
        with self.assertRaisesRegex(RuntimeError, 'log path'): self.submit()
        self.assertFalse(any(c[:2] == ['scontrol','release'] for c in self.calls))

    def test_actual_submitted_script_must_equal_prepared_bytes(self):
        self.spool_override = '#!/bin/bash\nexit 0\n'
        with self.assertRaisesRegex(RuntimeError, 'spooled script differs'): self.submit()
        self.assertEqual(json.loads(self.ledger.read_text())['state'], 'submitted_held')

    def test_script_edit_after_hold_blocks_release(self):
        self.submit()
        self.script.write_text(self.script.read_text()+'# changed\n')
        with self.assertRaisesRegex(RuntimeError, 'script changed'): submitter.release(self.script)
        self.assertFalse(any(c[:2] == ['scontrol','release'] for c in self.calls))

    def test_job_must_still_be_held_when_releasing(self):
        self.submit()
        self.fields['Reason'] = 'Priority'
        with self.assertRaisesRegex(RuntimeError, 'job not held'): submitter.release(self.script)
        self.assertFalse(any(c[:2] == ['scontrol','release'] for c in self.calls))

    def test_script_outside_allowed_stage_is_rejected(self):
        other = self.root/'outside.sh'; other.write_text(self.script.read_text())
        with self.assertRaisesRegex(RuntimeError, 'unexpected script path'):
            submitter.context(other, 'gap190', 't6g22d')

    def test_preexisting_spool_symlink_is_not_accepted_as_submission_proof(self):
        (self.stage/'verified-spool-999.sh').symlink_to(self.script)
        with self.assertRaisesRegex(RuntimeError, 'spool symlink'): self.submit()
        self.assertEqual(json.loads(self.ledger.read_text())['job_id'], '999')

    def test_release_argument_must_match_ledger_script(self):
        self.submit()
        other = self.stage/'different.sh'; other.write_text(self.script.read_text())
        with self.assertRaisesRegex(RuntimeError, 'release script differs'):
            submitter.release(other)
        self.assertFalse(any(c[:2] == ['scontrol','release'] for c in self.calls))

    def test_wrong_script_gpu_time_or_dependency_is_rejected_before_sbatch(self):
        original = self.script.read_text()
        bad_scripts = [original.replace('--gpus=8','--gpus=4'),
                       original.replace('--time=02:00:00','--time=03:00:00'),
                       original+'#SBATCH --dependency=afterany:208377\n']
        for text in bad_scripts:
            self.script.write_text(text)
            with self.assertRaises(RuntimeError): self.submit()
        self.assertFalse(self.ledger.exists())
        self.assertFalse(any(c[0] == 'sbatch' for c in self.calls))


if __name__ == '__main__':
    unittest.main()
