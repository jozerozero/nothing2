"""Mock-only tests for bounded v5 continuation; never invoke Slurm or a fitter."""
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import table6_ag_existing_v5 as v


class V5Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.node = v.ALLOWED['206117']
        self.now = dt.datetime(2026, 9, 22, 15, 0, tzinfo=dt.timezone.utc).timestamp()

    def scheduler(self, state='COMPLETED', queued=False, end='2026-09-22T13:00:00'):
        def run(command, **kwargs):
            identity = command[command.index('-j') + 1]
            if command[0] == 'sacct':
                text = f'{identity}|{state}|{end}\n'
            else:
                self.assertEqual(command[-1], '%i')
                text = f'{identity}.207\n' if queued else f'{identity}.batch\n{identity}.extern\n'
            return types.SimpleNamespace(returncode=0, stdout=text, stderr='')
        return run

    def previous(self):
        stage = self.root / 'v4'
        directory = stage / 'launches' / 'prior'
        directory.mkdir(parents=True)
        out = self.root / 'out'
        (out / 'claims').mkdir(parents=True)
        command = v.v4.srun_command('206117', self.node, 2, directory)
        plan = {'parent': '206117', 'node': self.node, 'plan_id': v.ag.PLAN,
                'ranks': 2, 'reserved_cpus_per_rank': 32, 'effective_cpus_per_rank': 16,
                'source_hashes': dict(v.PINNED), 'command': command}
        plan['operational_digest'] = v.ag.digest(plan)
        v.publish(directory / 'plan.json', plan)
        v.publish(directory / 'completion.json', {'returncode': 0})
        v.publish(directory / 'srun-intent.json', {'command': command})
        for rank in range(2):
            cpus = list(range(rank * 32, (rank + 1) * 32))
            owner = {'job_id': '206117', 'step_id': '207', 'host': self.node, 'rank': rank,
                     'plan_id': v.ag.PLAN, 'worker_sha256': v.old.WORKER_SHA}
            v.publish(directory / f'rank-{rank}.json', dict(owner, effective_fit_cpus=16,
                      slurm_cpus_per_task='32', reserved_cpu_ids=cpus, selected_cpu_ids=cpus[:16],
                      cpu_percent_samples=[[0.] * 128] * 2))
            v.publish(out / 'auxiliary' / stage.name / 'j206117-s207' / f'finished-{rank}.json',
                      {'state': 'paused', 'owner': owner})
        return stage, directory, out

    def claim(self, **changes):
        return dict({'job_id': '206117', 'step_id': '207', 'method': 'AutoGluon',
                     'state': 'paused', 'descendants_reaped': True, 'plan_id': v.ag.PLAN,
                     'worker_sha256': v.old.WORKER_SHA}, **changes)

    def test_only_reviewed_parent_and_exact_resources(self):
        v.target('206117', self.node, 2)
        for parent, node, ranks in [('206116', 'auh7-1b-gpu-308', 2),
                                     ('206117', self.node, 4), ('206117', self.node, True)]:
            with self.assertRaises(RuntimeError): v.target(parent, node, ranks)
        cmd = v.srun_command('206117', self.node, 2, self.root)
        for item in ('--gpus=0', '--gpus-per-task=0', '--gres=none', '--ntasks=2',
                     '--cpus-per-task=32', '--mem=256G', '--time=02:00:00', '--exact', '--exclusive'):
            self.assertIn(item, cmd)
        self.assertIn(str(Path(v.__file__).resolve()), cmd)
        self.assertNotIn('--overlap', cmd)

    def test_running_queued_unknown_or_recent_previous_step_refused(self):
        self.assertEqual(v.terminal_identity('206117', '207', run=self.scheduler(), now=self.now)['owner'], '206117.207')
        for scheduler in [self.scheduler(state='RUNNING'), self.scheduler(queued=True),
                          self.scheduler(end='2026-09-22T14:59:00')]:
            with self.assertRaises(RuntimeError):
                v.terminal_identity('206117', '207', run=scheduler, now=self.now)
        with self.assertRaises(RuntimeError):
            v.terminal_identity('206117', None, run=self.scheduler(), now=self.now)

    def test_clean_completed_v4_with_paused_claim_is_readonly(self):
        stage, directory, out = self.previous()
        v.publish(out / 'claims' / 'old.json', self.claim())
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with patch.object(v, 'PREVIOUS_STAGE', stage), patch.object(v.ag, 'OUT', out):
            proof = v.previous_attempt(directory, '206117', self.node, run=self.scheduler(), now=self.now)
            self.assertTrue(proof['claims_untouched'])
            self.assertEqual(len(proof['source_records']), 7)
            self.assertEqual(proof['matching_canonical_claims'][0]['state'], 'paused')
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_bad_old_claim_blocks_continuation(self):
        stage, directory, out = self.previous()
        with patch.object(v, 'PREVIOUS_STAGE', stage), patch.object(v.ag, 'OUT', out):
            for bad in [self.claim(state='running'), self.claim(descendants_reaped=False),
                        self.claim(plan_id='wrong')]:
                v.publish(out / 'claims' / 'old.json', bad, replace=True)
                with self.assertRaisesRegex(RuntimeError, 'cleanup proof'):
                    v.previous_attempt(directory, '206117', self.node, run=self.scheduler(), now=self.now)

    def test_other_active_ag_owner_blocks_even_with_own_previous_complete(self):
        stage, directory, out = self.previous()
        v.publish(out / 'claims' / 'other.json', self.claim(job_id='206116', step_id='400', state='running'))
        scheduler = self.scheduler()
        def running_other(command, **kwargs):
            if command[0] == 'sacct' and command[command.index('-j')+1] == '206116':
                return types.SimpleNamespace(returncode=0, stdout='206116|RUNNING|Unknown\n206116.400|RUNNING|Unknown\n', stderr='')
            return scheduler(command, **kwargs)
        with patch.object(v, 'PREVIOUS_STAGE', stage), patch.object(v.ag, 'OUT', out):
            with self.assertRaisesRegex(RuntimeError, 'owner nonterminal'):
                v.previous_attempt(directory, '206117', self.node, run=running_other, now=self.now)

    def test_historical_missing_step_requires_whole_parent_terminal_and_empty_queue(self):
        stage, directory, out = self.previous()
        v.publish(out / 'claims' / 'legacy.json', self.claim(job_id='181785', step_id=None, state='running'))
        scheduler = self.scheduler()
        def legacy(command, **kwargs):
            if command[command.index('-j')+1] == '181785':
                if command[0] == 'sacct':
                    return types.SimpleNamespace(returncode=0, stdout='181785|COMPLETED|2026-09-22T13:00:00\n', stderr='')
                return types.SimpleNamespace(returncode=1, stdout='', stderr='squeue: error: Invalid job id specified')
            return scheduler(command, **kwargs)
        with patch.object(v, 'PREVIOUS_STAGE', stage), patch.object(v.ag, 'OUT', out):
            proof = v.previous_attempt(directory, '206117', self.node, run=legacy, now=self.now)
            self.assertTrue(proof['terminated_historical_running_claims'][0]['terminal']['queue_empty'])

    def test_exact_native_gone_spelling_only_after_terminal_proof(self):
        stage, directory, out = self.previous()
        v.publish(out / 'claims' / 'legacy.json', self.claim(job_id='181785', step_id=None, state='running'))
        scheduler = self.scheduler()
        def legacy(state='COMPLETED', error='slurm_load_jobs error: Invalid job id specified\n', stdout=''):
            calls = []
            def run(command, **kwargs):
                if command[command.index('-j')+1] != '181785':
                    return scheduler(command, **kwargs)
                calls.append(command[0])
                if command[0] == 'sacct':
                    return types.SimpleNamespace(returncode=0, stdout=f'181785|{state}|2026-09-22T13:00:00\n', stderr='')
                return types.SimpleNamespace(returncode=1, stdout=stdout, stderr=error)
            return run, calls
        with patch.object(v, 'PREVIOUS_STAGE', stage), patch.object(v.ag, 'OUT', out):
            run, calls = legacy()
            proof = v.previous_attempt(directory, '206117', self.node, run=run, now=self.now)
            self.assertTrue(proof['terminated_historical_running_claims'][0]['terminal']['queue_empty'])
            self.assertEqual(calls, ['sacct', 'squeue', 'squeue'])
            run, calls = legacy(state='RUNNING')
            with self.assertRaisesRegex(RuntimeError, 'owner nonterminal'):
                v.previous_attempt(directory, '206117', self.node, run=run, now=self.now)
            self.assertEqual(calls, ['sacct'])
            for error, stdout in [('slurm_load_jobs error: Socket timed out\n', ''),
                                  ('slurm_load_jobs error: Invalid job id specified; permission denied\n', ''),
                                  ('slurm_load_jobs error: Invalid job id specified\n', '181785|RUNNING\n')]:
                run, _ = legacy(error=error, stdout=stdout)
                with self.assertRaisesRegex(RuntimeError, 'owner query failed'):
                    v.previous_attempt(directory, '206117', self.node, run=run, now=self.now)

    def test_missing_rank_finish_and_fatal_audit_block(self):
        stage, directory, out = self.previous()
        audit = out / 'auxiliary' / stage.name / 'j206117-s207'
        v.publish(audit / 'cleanup-fatal-any.json', {'fatal': True})
        with patch.object(v, 'PREVIOUS_STAGE', stage), patch.object(v.ag, 'OUT', out):
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                v.previous_attempt(directory, '206117', self.node, run=self.scheduler(), now=self.now)

    def test_live_process_guard_rejects_v4_and_other_v5_steps(self):
        own = {'pid': 12, 'cmdline': ['python', str(Path(v.__file__).resolve()), 'work'],
               'environ': {'SLURM_JOB_ID': '206117', 'SLURM_STEP_ID': '300', 'SLURM_NTASKS': '2'}}
        with patch.object(v.socket, 'gethostname', return_value=self.node):
            v.no_other_ag_processes([own], '206117', '300')
            for bad in [dict(own, cmdline=['python', 'table6_ag_existing_v4.py', 'work']),
                        dict(own, environ=dict(own['environ'], SLURM_STEP_ID='299'))]:
                with self.assertRaisesRegex(RuntimeError, 'another AG process'):
                    v.no_other_ag_processes([own, bad], '206117', '300')

    def test_immutable_bodies_and_source_pins(self):
        hashes = v.sources()
        self.assertEqual({key: hashes[key] for key in v.PINNED}, v.PINNED)
        self.assertEqual(v.PRIVATE['rank_entry'].__code__, v.v4.rank_entry.__code__)
        self.assertIsNot(v.PRIVATE['rank_entry'].__globals__, v.v4.rank_entry.__globals__)
        self.assertEqual(v.v4.STAGE, v.PREVIOUS_STAGE)
        self.assertEqual(v.v4.PREVIOUS_STEPS['206117'], '190')
        self.assertEqual(v.ag.SEEDS, list(range(15)))

    def test_new_supervisor_rank_seed_paths_and_effective_cpu_contract(self):
        attrs = ('STAGE', 'CAMPAIGN', 'verify_plan', 'sources', 'srun_command', 'validate_parent',
                 'exact_step_evidence', 'dynamic_preflight', 'guarded_run_seed', 'target')
        saved = {name: getattr(v.v3, name) for name in attrs}
        try:
            with patch.dict(v.PRIVATE, ORIGINAL_GUARDED=lambda *args: args):
                v.configure('/new/launch')
                native = [sys_executable(), '-B', str(v.v3.REPO/'table6_restart_ag.py'),
                          'seed', '--key', 'a'*24, '--seed', '2']
                result = v.v3.guarded_run_seed(native, '/log', 3, 'budget', 'stop', {})
                self.assertEqual(result[0][2], str(Path(v.__file__).resolve()))
                self.assertEqual(result[0][-2:], ['--launch-dir', '/new/launch'])
                with self.assertRaises(RuntimeError): v.v3.target('206116', 'auh7-1b-gpu-308', 2)
        finally:
            for name, value in saved.items(): setattr(v.v3, name, value)


def sys_executable():
    import sys
    return sys.executable


if __name__ == '__main__':
    unittest.main()
