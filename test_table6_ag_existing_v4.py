"""Read-only-runtime mock tests: reserved32/effective16 and reviewed retry."""
import copy
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import table6_ag_existing_v4 as v


class V4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.node = v.v3.ALLOWED['206116']
        self.now = dt.datetime(2026, 9, 22, 4, 0, tzinfo=dt.timezone.utc).timestamp()

    def scheduler(self, state='FAILED', live=False):
        def run(command, **kwargs):
            if command[0] == 'sacct':
                step = command[command.index('-j')+1]
                text = f'206116|RUNNING|Unknown\n{step}|{state}|2026-09-22T03:00:00\n'
            else:
                self.assertEqual(command[-1], '%i')
                text = '206116.120\n' if live else '206116.130\n'
            return types.SimpleNamespace(returncode=0, stdout=text, stderr='')
        return run

    def old_attempt(self):
        folder = self.root/'old'/'launches'/'first'
        folder.mkdir(parents=True)
        plan = {'parent': '206116', 'node': self.node, 'plan_id': v.ag.PLAN,
                'source_hashes': {'table6_ag_existing_v3.py': v.V3_SHA}}
        plan['operational_digest'] = v.ag.digest(plan)
        for name, data in [('plan.json', plan), ('completion.json', {'returncode': 1}),
                           ('srun-intent.json', {'command': ['srun']}),
                           ('rank-0.json', {'job_id': '206116', 'step_id': '120', 'host': self.node})]:
            v.publish(folder/name, data)
        (self.root/'out'/'claims').mkdir(parents=True)
        return folder

    def test_select16_only_inside_reserved32_avoiding_busy_cpus(self):
        samples = [[0.0]*128 for _ in range(2)]
        for cpu in range(8): samples[0][cpu] = 80
        for cpu in range(8, 12): samples[1][cpu] = 80
        selected = v.select_cpus(list(range(32)), samples)
        self.assertEqual(selected, list(range(12, 28)))
        self.assertTrue(set(selected) <= set(range(32)))

    def test_insufficient_or_bad_samples_fail_closed(self):
        for samples in ([[100.0]*128]*2, [[0.0]*10]*2,
                        [[float('nan')]+[0.0]*127, [0.0]*128], [[0.0]*128]):
            with self.assertRaises(RuntimeError):
                v.select_cpus(list(range(32)), samples)
        with self.assertRaises(RuntimeError):
            v.select_cpus(list(range(16)), [[0.0]*128]*2)

    def test_two_rank_reservation_uses32_cpu256g_total_without_overlap(self):
        command = v.srun_command('206116', self.node, 2, self.root)
        for item in ('--ntasks=2', '--cpus-per-task=32', '--mem=256G', '--cpu-bind=threads',
                     '--gpus=0', '--gpus-per-task=0', '--exclusive', '--exact'):
            self.assertIn(item, command)
        self.assertIn(str(Path(v.__file__).resolve()), command)
        self.assertNotIn('--overlap', command)
        with self.assertRaises(RuntimeError):
            v.srun_command('206116', self.node, 4, self.root)

    def test_effective_native16_does_not_forge_global_slurm(self):
        with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '32', 'OTHER': 'kept'}), \
                patch.object(os, 'sched_getaffinity', return_value=set(range(16)), create=True):
            proxy = v.EffectiveFitOS()
            self.assertEqual(proxy.getenv('SLURM_CPUS_PER_TASK'), '16')
            self.assertEqual(os.getenv('SLURM_CPUS_PER_TASK'), '32')
            self.assertEqual(proxy.getenv('OTHER'), 'kept')
            self.assertIs(proxy.environ, os.environ)
            self.assertEqual(proxy.getpid(), os.getpid())
        with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '32'}), \
                patch.object(os, 'sched_getaffinity', return_value=set(range(32)), create=True):
            with self.assertRaises(RuntimeError):
                v.EffectiveFitOS().getenv('SLURM_CPUS_PER_TASK')

    def test_exact_failed_step_terminal_while_parent_running(self):
        evidence = v.terminal_step('206116', '120', run=self.scheduler(), now=self.now)
        self.assertEqual(evidence['owner'], '206116.120')
        for run in (self.scheduler(state='RUNNING'), self.scheduler(live=True)):
            with self.assertRaises(RuntimeError):
                v.terminal_step('206116', '120', run=run, now=self.now)
        with self.assertRaises(RuntimeError):
            v.terminal_step('206116', '119', run=self.scheduler(), now=self.now)

    def test_paused_old206116_16_recovery_uses_supported_step_formatter(self):
        claim = {'job_id': '206116', 'step_id': '16', 'state': 'paused', 'descendants_reaped': True,
                 'worker_sha256': v.old.WORKER_SHA, 'plan_id': v.ag.PLAN, 'host': self.node, 'owner_token': 'old'}
        evidence = v.exact_step_evidence(claim, run=self.scheduler(), now=self.now)
        self.assertEqual(evidence['owner'], '206116.16')

    def test_reviewed_previous_failure_requires_no_claims_and_retains_all_files(self):
        folder = self.old_attempt()
        before = {str(p): p.read_bytes() for p in folder.iterdir()}
        with patch.object(v, 'V3_STAGE', self.root/'old'), patch.object(v.ag, 'OUT', self.root/'out'):
            proof = v.previous_attempt(folder, '206116', self.node, run=self.scheduler(), now=self.now)
            self.assertEqual(proof['matching_canonical_claims'], [])
            self.assertEqual(len(proof['source_records']), 4)
            v.publish(self.root/'out/claims/one.json', {'job_id': '206116', 'step_id': '120'})
            with self.assertRaisesRegex(RuntimeError, 'canonical claims'):
                v.previous_attempt(folder, '206116', self.node, run=self.scheduler(), now=self.now)
        self.assertEqual({str(p): p.read_bytes() for p in folder.iterdir()}, before)

    def test_previous_fit_audit_or_missing_claim_directory_blocks_retry(self):
        folder = self.old_attempt()
        with patch.object(v, 'V3_STAGE', self.root/'old'), patch.object(v.ag, 'OUT', self.root/'out'):
            audit = v.ag.OUT/'auxiliary'/v.V3_STAGE.name/'j206116-s120'
            v.publish(audit/'new-owner-a.json', {'untrusted': True})
            with self.assertRaisesRegex(RuntimeError, 'entered fitting'):
                v.previous_attempt(folder, '206116', self.node, run=self.scheduler(), now=self.now)

    def test_frozen_sources_and_native_scientific_contract_pinned(self):
        hashes = v.sources()
        self.assertEqual(hashes['table6_ag_existing_v3.py'], v.V3_SHA)
        self.assertEqual(hashes['table6_restart_ag.py'], v.old.WORKER_SHA)
        self.assertEqual(v.ag.SEEDS, list(range(15)))

    def test_step_seed_command_rewrite_only_known_frozen_entry(self):
        attrs = ('STAGE', 'CAMPAIGN', 'verify_plan', 'sources', 'srun_command', 'validate_parent',
                 'exact_step_evidence', 'dynamic_preflight', 'guarded_run_seed')
        saved = {name: getattr(v.v3, name) for name in attrs}
        try:
            with patch.object(v, 'ORIGINAL_GUARDED', return_value=(0, None)) as guarded:
                v.configure('/new/launch')
                native = [v.sys.executable, '-B', str(v.v3.REPO/'table6_restart_ag.py'),
                          'seed', '--key', 'a'*24, '--seed', '2']
                v.v3.guarded_run_seed(native, '/log', 9, 'budget', 'stop', {})
                rewritten = guarded.call_args.args[0]
                self.assertEqual(rewritten[:4], [v.sys.executable, '-B', str(Path(v.__file__).resolve()), 'seed'])
                self.assertEqual(rewritten[-2:], ['--launch-dir', '/new/launch'])
                self.assertEqual(guarded.call_args.args[2], 9)
                with self.assertRaises(RuntimeError):
                    v.v3.guarded_run_seed(['wrong'], '/log', 9, 'budget', 'stop', {})
        finally:
            for name, value in saved.items(): setattr(v.v3, name, value)


if __name__ == '__main__':
    unittest.main()
