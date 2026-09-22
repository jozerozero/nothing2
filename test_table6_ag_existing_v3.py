"""CPU/stdlib mock tests; no Slurm, remote access, or real campaign writes."""
import copy
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import table6_ag_existing_v3 as v


class ExistingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.node = v.ALLOWED['206116']
        self.proof = {'allow_cpu_sidecar': True, 'parent_job_id': '206116', 'node': self.node,
                      'observations': [self.sample(100), self.sample(115)]}
        self.claim = {'job_id': '206116', 'step_id': '16', 'host': self.node, 'owner_token': 'old',
                      'state': 'paused', 'descendants_reaped': True, 'worker_sha256': v.old.WORKER_SHA,
                      'plan_id': v.ag.PLAN}
        self.now = dt.datetime(2026, 9, 22, 4, 0, tzinfo=dt.timezone.utc).timestamp()

    def sample(self, when):
        return {'observed_epoch': when, 'parent_job_id': '206116', 'node': self.node,
                'idle_cpu_ids': list(range(64)), 'parent_cpu_ids': list(range(128)),
                'parent_memory_current_bytes': 100 * v.GIB, 'parent_memory_limit_bytes': 2048 * v.GIB,
                'available_memory_bytes': 1500 * v.GIB, 'parent_memory_source': 'cgroup_v2'}

    def raw(self, **changes):
        data = {'JobId': '206116', 'JobState': 'RUNNING', 'NumNodes': '1', 'NodeList': self.node,
                'UserId': f'user({os.getuid()})', 'NumCPUs': '128', 'AllocTRES': 'cpu=128,mem=2T,node=1',
                'TimeLimit': '1-00:00:00', 'RunTime': '01:00:00', 'EndTime': '2026-09-23T00:00:00'}
        data.update(changes)
        return ' '.join(k + '=' + val for k, val in data.items())

    def scheduler(self, state='COMPLETED', queue='206116.17|RUNNING\n', error=False):
        def run(args, **kwargs):
            self.assertEqual(kwargs['env']['TZ'], 'UTC')
            if args[0] == 'sacct':
                self.assertIn('206116.16', args)
                # The still-running parent must NOT block exact-step recovery.
                text = f'206116|RUNNING|Unknown\n206116.16|{state}|2026-09-22T03:00:00\n'
            else:
                text = queue
            return types.SimpleNamespace(returncode=int(error), stdout=text, stderr='outage' if error else '')
        return run

    def test_exact_terminal_step_can_resume_while_parent_and_other_step_running(self):
        result = v.exact_step_evidence(self.claim, run=self.scheduler(), now=self.now)
        self.assertTrue(result['parent_may_still_be_running'])
        self.assertEqual(result['owner'], '206116.16')

    def test_terminal_step_requires_cleanup_and_known_scientific_identity(self):
        for changes in ({'state': 'running'}, {'descendants_reaped': False}, {'worker_sha256': 'different'},
                        {'plan_id': 'different'}, {'owner_token': ''}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                v.exact_step_evidence(dict(self.claim, **changes), run=self.scheduler(), now=self.now)

    def test_age_alone_no_takeover_and_scheduler_errors_fail_closed(self):
        for run in (self.scheduler(state='RUNNING'), self.scheduler(queue='206116.16|RUNNING\n'),
                    self.scheduler(error=True), self.scheduler(queue='unparseable\n')):
            with self.assertRaises(RuntimeError):
                v.exact_step_evidence(self.claim, run=run, now=self.now)
        with self.assertRaisesRegex(RuntimeError, 'grace'):
            v.exact_step_evidence(self.claim, run=self.scheduler(), now=self.now - 3550)

    def test_legacy_without_step_retains_original_strict_terminal_policy(self):
        claim = dict(self.claim, step_id=None)
        with patch.object(v, 'ag_original_terminal', return_value={'old': True}) as original:
            self.assertEqual(v.exact_step_evidence(claim), {'old': True})
            self.assertEqual(original.call_args.args[0], claim)

    def test_allowlist_and_dynamic_resources(self):
        for ranks in (1, 2, 4):
            v.validate_parent(self.raw(), '206116', self.node, ranks)
            v.validate_capacity(self.proof, '206116', self.node, ranks, now=120)
            cmd = v.srun_command('206116', self.node, ranks, self.root)
            for arg in (f'--ntasks={ranks}', f'--mem={128*ranks}G', '--cpus-per-task=16',
                        '--gpus=0', '--gpus-per-task=0', '--gres=none', '--exact', '--exclusive'):
                self.assertIn(arg, cmd)
            self.assertNotIn('--overlap', cmd)
            self.assertNotIn('sbatch', cmd)
            self.assertIn('ROCR_VISIBLE_DEVICES=-1', cmd[cmd.index('env'):])
        for parent, node, ranks in (('196093', self.node, 1), ('206116', 'wrong', 1),
                                     ('206116', self.node, 3), ('206116', self.node, True)):
            with self.assertRaises(RuntimeError):
                v.target(parent, node, ranks)

    def test_parent_requires_owned_running_resources_and_enough_lifetime(self):
        for changes in ({'JobState': 'PENDING'}, {'UserId': 'other(99999999)'},
                        {'AllocTRES': 'cpu=128,mem=64G'}, {'NumCPUs': '32'}, {'TimeLimit': '02:00:00'}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                v.validate_parent(self.raw(**changes), '206116', self.node, 4)

    def test_capacity_must_be_fresh_independent_and_whole_parent(self):
        for key, bad in [('observed_epoch', 114), ('idle_cpu_ids', list(range(63))),
                         ('parent_memory_current_bytes', 1500*v.GIB),
                         ('available_memory_bytes', 540*v.GIB), ('parent_memory_source', 'own_rss')]:
            proof = copy.deepcopy(self.proof)
            proof['observations'][0][key] = bad
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                v.validate_capacity(proof, '206116', self.node, 4, now=120)
        with self.assertRaises(RuntimeError):
            v.validate_capacity(self.proof, '206116', self.node, 4, now=500)

    def test_memory_guard_counts_whole_parent_not_only_own_fit(self):
        snapshot = {'parent_memory_current_bytes': 100*v.GIB, 'parent_memory_limit_bytes': 2048*v.GIB,
                    'available_memory_bytes': 1000*v.GIB}
        v.check_memory(snapshot, 4, startup=True)
        with self.assertRaisesRegex(RuntimeError, 'whole_parent'):
            v.check_memory(dict(snapshot, parent_memory_current_bytes=2000*v.GIB), 4, startup=False)
        with self.assertRaisesRegex(RuntimeError, 'node_memory'):
            v.check_memory(dict(snapshot, available_memory_bytes=31*v.GIB), 4, startup=False)

    def test_exact_parent_cgroup_not_step_or_adjacent_job(self):
        used, limit, source = v.parent_memory_files('206116', '0::/system.slice/slurmstepd.scope/job_206116/step_20/task_0')
        self.assertEqual(str(used), '/sys/fs/cgroup/system.slice/slurmstepd.scope/job_206116/memory.current')
        self.assertEqual(source, 'cgroup_v2')
        used, limit, source = v.parent_memory_files('206116', '4:memory:/slurm/uid_1000/job_206116/step_20')
        self.assertEqual(str(used), '/sys/fs/cgroup/memory/slurm/uid_1000/job_206116/memory.usage_in_bytes')
        self.assertEqual(source, 'cgroup_v1')
        with self.assertRaises(RuntimeError):
            v.parent_memory_files('206116', '0::/slurm/job_2061160/step_20')

    def test_dynamic_gate_requires_fastai_disjoint_cpus_and_zero_gpus(self):
        def record(rank):
            return {'rank': rank, 'pass': True, 'job_id': '206116', 'step_id': '20', 'host': self.node,
                    'plan_id': v.ag.PLAN, 'devices': 0, 'nice': 19, 'environment': v.ag.CPU_ENV,
                    'fastai': 'NNFastAiTabularModel', 'cpu_affinity': list(range(rank*16, (rank+1)*16))}
        for ranks in (1, 2, 4):
            records = [record(r) for r in range(ranks)]
            self.assertEqual(len(v.validate_gate(records, '206116', '20', self.node, ranks)), 16*ranks)
        for bad in ({'fastai': ''}, {'devices': 1}, {'nice': 0}):
            with self.assertRaises(RuntimeError):
                v.validate_gate([dict(record(0), **bad)], '206116', '20', self.node, 1)
        with self.assertRaises(RuntimeError):
            v.validate_gate([record(0), dict(record(1), cpu_affinity=list(range(16)))], '206116', '20', self.node, 2)

    def test_environment_cpu_sentinels_fastai_and_no_foreign_slurm(self):
        env = v.clean_environment({'PATH': '/bin', 'SLURM_JOB_ID': 'other', 'JOB_BUDGET_END_EPOCH': '123',
                                   'CUDA_VISIBLE_DEVICES': '0', 'PYTHONHOME': '/bad'})
        self.assertNotIn('SLURM_JOB_ID', env)
        self.assertNotIn('JOB_BUDGET_END_EPOCH', env)
        self.assertNotIn('PYTHONHOME', env)
        self.assertTrue(all(env[k] == val for k, val in v.ag.CPU_ENV.items()))
        self.assertEqual(env['GPU_DEVICE_ORDINAL'], '-1')
        self.assertIn('fastai_overlay', env['PYTHONPATH'])

    def test_immutable_launch_intent_blocks_duplicate_and_uncertain_launch(self):
        proof_file = self.root / 'proof.json'
        proof_file.write_text(json.dumps(self.proof))
        with patch.object(v, 'STAGE', self.root / 'stage'), patch.object(v.time, 'time', return_value=120), \
                patch.object(v.old, 'command', return_value=self.raw()), patch.object(v, 'sources', return_value={'source': 'sha'}), \
                patch.object(v.subprocess, 'Popen', return_value=types.SimpleNamespace(pid=321)) as spawn:
            receipt = v.launch('206116', self.node, 4, proof_file, 'first')
            self.assertEqual(receipt['ranks'], 4)
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            with self.assertRaises(FileExistsError):
                v.launch('206116', self.node, 4, proof_file, 'second')
            self.assertEqual(spawn.call_count, 1)
        with patch.object(v, 'STAGE', self.root / 'uncertain'), patch.object(v.time, 'time', return_value=120), \
                patch.object(v.old, 'command', return_value=self.raw()), patch.object(v, 'sources', return_value={}), \
                patch.object(v.subprocess, 'Popen', side_effect=OSError('uncertain')):
            with self.assertRaises(OSError):
                v.launch('206116', self.node, 4, proof_file, 'first')
            self.assertTrue((v.STAGE / 'parents/206116.json').exists())

    def test_resource_pause_before_fit_does_not_create_model_error_or_start_child(self):
        with patch.object(v, 'memory_snapshot', side_effect=OSError('controller unavailable')), \
                patch.object(v.subprocess, 'Popen') as spawn:
            code, reason = v.guarded_run_seed([], self.root/'log', 5, types.SimpleNamespace(remaining=lambda: 1000),
                                             types.SimpleNamespace(reason=None), {'parent': '206116', 'allocated_memory_bytes': 2**41, 'ranks': 4})
            self.assertIsNone(code)
            self.assertIn('memory_proof', reason)
            spawn.assert_not_called()

    def test_original_worker_pins_and_scientific_seed_contract_unchanged(self):
        self.assertEqual(v.sources()['table6_restart_ag.py'], v.old.WORKER_SHA)
        self.assertEqual(v.ag.SEEDS, list(range(15)))
        self.assertEqual(v.ag.PLAN, '95ed50cd7348ed167f71ff159f6af14cc351e67959d6d227f9201bdc024fbb85')

    def test_frozen_pair_resume_preserves_prior_claim_and_reaps_on_pause(self):
        pair = {'key': 'a'*24, 'method': 'AutoGluon', 'dataset': 'D'}
        old_claim = dict(self.claim, **pair)
        path = self.root / 'claims' / (pair['key'] + '.json')
        v.publish(path, old_claim)
        before = path.read_bytes()
        owner = {'job_id': '206117', 'step_id': '50', 'rank': 0, 'pid': 9,
                 'host': v.ALLOWED['206117'], 'worker_sha256': v.old.WORKER_SHA}
        audit = self.root/'audit'; audit.mkdir()
        terminal = lambda claim: v.exact_step_evidence(claim, run=self.scheduler(), now=self.now)
        with patch.object(v.ag, 'OUT', self.root), patch.object(v.ag, 'terminal_evidence', side_effect=terminal), \
                patch.object(v.ag, 'inspect_pair', return_value=({}, False)), \
                patch.object(v.ag, 'run_seed', return_value=(None, 'allocation_budget')):
            result = v.ag.work_pair(None, {}, pair, audit, owner, types.SimpleNamespace(remaining=lambda: 1000),
                                    types.SimpleNamespace(reason=None))
        self.assertEqual(result, 'paused')
        after = v.read(path)
        self.assertEqual(after['job_id'], '206117')
        self.assertTrue(after['descendants_reaped'])
        self.assertNotEqual(after['owner_token'], 'old')
        prior = v.read(audit/f"prior-claim-{pair['key']}-r0.json")
        self.assertEqual(prior['original_text'], before.decode())
        self.assertTrue(prior['terminal_evidence']['parent_may_still_be_running'])

    def test_frozen_pair_claim_cas_rejects_changed_claim_before_takeover(self):
        pair = {'key': 'a'*24, 'method': 'AutoGluon', 'dataset': 'D'}
        path = self.root / 'claims' / (pair['key'] + '.json')
        old_claim = dict(self.claim, **pair)
        v.publish(path, old_claim)
        def racing_check(claim):
            v.publish(path, dict(claim, owner_token='raced'), replace=True)
            return {'test': True}
        owner = {'job_id': '206117', 'step_id': '50', 'rank': 0, 'host': v.ALLOWED['206117'],
                 'worker_sha256': v.old.WORKER_SHA}
        audit = self.root/'audit'; audit.mkdir()
        with patch.object(v.ag, 'OUT', self.root), patch.object(v.ag, 'terminal_evidence', side_effect=racing_check), \
                patch.object(v.ag, 'inspect_pair', return_value=({}, False)), patch.object(v.ag, 'run_seed') as fit:
            with self.assertRaisesRegex(RuntimeError, 'changed during owner checks'):
                v.ag.work_pair(None, {}, pair, audit, owner, types.SimpleNamespace(remaining=lambda: 1000),
                               types.SimpleNamespace(reason=None))
            fit.assert_not_called()
        self.assertEqual(v.read(path)['owner_token'], 'raced')


if __name__ == '__main__':
    unittest.main()
