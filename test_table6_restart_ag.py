"""Stdlib tests; no Slurm, ML dependencies or real campaign writes."""
import copy
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

import table6_restart_ag as ag


class AGTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pair = {'method': 'AutoGluon', 'dataset': 'D', 'suite': 'PFN', 'key': 'a' * 24,
                     'config': {'preset': 'best_quality'}, 'config_sha256': 'cfg', 'source': '/frozen'}
        self.audit = {'exact_frozen_cache_match': True, 'test_labels_sha256': 'testhash'}
        self.owner = {'job_id': '999', 'step_id': '0', 'rank': 0, 'host': 'node', 'worker_sha256': 'worker'}
        self.budget = types.SimpleNamespace(remaining=lambda: 1000)
        self.stop = ag.Stop()
        self.stop.reason = None
        self.patcher = patch.object(ag, 'OUT', self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def seed(self, seed):
        return {'complete': True, 'plan_id': ag.PLAN, 'method': 'AutoGluon',
                'dataset': 'D', 'suite': 'PFN', 'seed': seed, 'selected_config_sha256': 'cfg',
                'selected_config_source': '/frozen', 'hpo_trials': 0,
                'data_audit': self.audit, 'metrics': {'ACC': .8, 'AUC': .9, 'F1': .7}}

    def save_seed(self, seed):
        path = self.root / 'seeds' / self.pair['key'] / f'{seed:02d}.json'
        ag.publish(path, self.seed(seed))
        return path

    def save_complete(self):
        for seed in ag.SEEDS:
            self.save_seed(seed)
        value = {'complete': True, 'plan_id': ag.PLAN, 'method': 'AutoGluon', 'dataset': 'D',
                 'suite': 'PFN', 'hpo_trials': 0, 'seed_count': 15, 'seeds': ag.SEEDS,
                 'selected_config': self.pair['config'], 'selected_config_sha256': 'cfg',
                 'selected_config_source': '/frozen', 'data_audit': self.audit,
                 'metrics_mean': {'ACC': .8, 'AUC': .9, 'F1': .7}}
        path = self.root / 'results/AutoGluon/D.json'
        ag.publish(path, value)
        return path

    def test_publication_never_overwrites(self):
        path = self.root / 'seed.json'
        ag.publish(path, {'a': 1})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            ag.publish(path, {'a': 2})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_complete_read_only_and_all_seeds_required(self):
        result = self.save_complete()
        before = result.stat().st_mtime_ns, result.read_bytes()
        self.assertTrue(ag.inspect_pair(self.pair)[1])
        self.assertEqual((result.stat().st_mtime_ns, result.read_bytes()), before)
        (self.root / 'seeds' / self.pair['key'] / '14.json').unlink()
        with self.assertRaisesRegex(RuntimeError, '15 matching seeds'):
            ag.inspect_pair(self.pair)

    def test_invalid_existing_seed_is_not_missing(self):
        path = self.save_seed(0)
        value = self.seed(0)
        value['metrics']['ACC'] = float('nan')
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, 'invalid seed metric'):
            ag.inspect_pair(self.pair)

    def test_partial_seed_audits_must_match(self):
        self.save_seed(0)
        value = self.seed(1)
        value['data_audit'] = dict(self.audit, test_labels_sha256='changed')
        ag.publish(self.root / 'seeds' / self.pair['key'] / '01.json', value)
        with self.assertRaisesRegex(RuntimeError, 'audits differ'):
            ag.inspect_pair(self.pair)

    def test_wrong_aggregate_mean_rejected(self):
        path = self.save_complete()
        value = json.loads(path.read_text())
        value['metrics_mean']['ACC'] = .81
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, 'mean mismatch'):
            ag.inspect_pair(self.pair)

    def mock_owner(self, jobstate='TIMEOUT', stepstate='CANCELLED', ended='2026-09-01T00:00:00', queue=''):
        def run(args, **kwargs):
            self.assertEqual(kwargs['env']['TZ'], 'UTC')
            stdout = queue if args[0] == 'squeue' else f'1|{jobstate}|{ended}\n1.2|{stepstate}|{ended}\n'
            return types.SimpleNamespace(returncode=0, stdout=stdout, stderr='')
        return run

    def test_terminal_job_and_exact_step_plus_grace(self):
        now = dt.datetime(2026, 9, 1, 0, 2, tzinfo=dt.timezone.utc).timestamp()
        value = ag.terminal_evidence({'job_id': '1', 'step_id': '2'}, run=self.mock_owner(), now=now)
        self.assertEqual([v['owner'] for v in value['terminal']], ['1', '1.2'])
        with self.assertRaisesRegex(RuntimeError, 'grace'):
            ag.terminal_evidence({'job_id': '1'}, run=self.mock_owner(), now=now - 1)

    def test_active_unknown_and_failed_owner_checks_fail_closed(self):
        for claim, run in [({'job_id': '1', 'step_id': '2'}, self.mock_owner(stepstate='RUNNING')),
                           ({'job_id': '1'}, self.mock_owner(queue='1|RUNNING')),
                           ({'job_id': '1', 'step_id': '3'}, self.mock_owner())]:
            with self.assertRaises(RuntimeError):
                ag.terminal_evidence(claim, run=run)
        with self.assertRaises(RuntimeError):
            ag.terminal_evidence({'job_id': '../bad'}, run=self.mock_owner())

    def test_null_step_uses_proven_terminal_job(self):
        value = ag.terminal_evidence({'job_id': '1', 'step_id': None}, run=self.mock_owner())
        self.assertEqual([v['owner'] for v in value['terminal']], ['1'])

    def test_expired_job_queue_exception_is_narrow_and_after_accounting(self):
        commands = []
        def run(args, **kwargs):
            commands.append(args[0])
            if args[0] == 'sacct':
                return self.mock_owner()(args, **kwargs)
            return types.SimpleNamespace(returncode=1, stdout='', stderr='squeue: error: Invalid job id specified\n')
        self.assertTrue(ag.terminal_evidence({'job_id': '1'}, run=run)['queue_empty'])
        self.assertEqual(commands[0], 'sacct')
        def outage(args, **kwargs):
            if args[0] == 'sacct':
                return self.mock_owner()(args, **kwargs)
            return types.SimpleNamespace(returncode=1, stdout='', stderr='Unable to contact slurm controller')
        with self.assertRaisesRegex(RuntimeError, 'owner query failed'):
            ag.terminal_evidence({'job_id': '1'}, run=outage)

    def gate_rows(self):
        return [{'pass': True, 'rank': i, 'job_id': '99', 'step_id': '0', 'host': 'node',
                 'plan_id': ag.PLAN, 'devices': 0, 'nice': 19, 'environment': ag.CPU_ENV,
                 'cpu_affinity': list(range(i * 16, (i + 1) * 16))} for i in range(4)]

    def test_gate_requires_four_disjoint_ranks_cpu_only(self):
        rows = self.gate_rows()
        self.assertEqual(len(ag.validate_gate(rows, '99', '0', 'node')), 64)
        for change in ('overlap', 'gpu', 'host', 'rank'):
            broken = copy.deepcopy(rows)
            if change == 'overlap':
                broken[1]['cpu_affinity'] = broken[0]['cpu_affinity']
            elif change == 'gpu':
                broken[1]['devices'] = 1
            elif change == 'host':
                broken[1]['host'] = 'other'
            else:
                broken[1]['rank'] = 0
            with self.assertRaises(RuntimeError):
                ag.validate_gate(broken, '99', '0', 'node')

    def test_signal_and_deadline_prevent_child_start(self):
        self.stop.signal(signal.SIGTERM, None)
        with patch.object(ag.subprocess, 'Popen') as spawn:
            self.assertEqual(ag.run_seed([], self.root / 'log', 4, self.budget, self.stop),
                             (None, 'signal_SIGTERM'))
            spawn.assert_not_called()
        self.stop.reason = None
        self.budget.remaining = lambda: 179
        self.assertEqual(ag.run_seed([], self.root / 'log', 4, self.budget, self.stop),
                         (None, 'allocation_budget'))

    def test_actual_fit_error_writes_once_and_stops_rank(self):
        with patch.object(ag, 'run_seed', return_value=(2, None)), patch.object(ag, 'process_snapshot', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'fit failed'):
                ag.work_pair(Mock(), {}, self.pair, self.root / 'audit', self.owner, self.budget, self.stop)
        errors = list((self.root / 'errors').glob('*.json'))
        self.assertEqual(len(errors), 1)
        self.assertFalse(ag.read(errors[0])['retry'])
        claim = ag.read(self.root / 'claims' / (self.pair['key'] + '.json'))
        self.assertEqual(claim['state'], 'failed')

    def test_deadline_pause_does_not_create_model_error(self):
        with patch.object(ag, 'run_seed', return_value=(None, 'allocation_budget')):
            status = ag.work_pair(Mock(), {}, self.pair, self.root / 'audit', self.owner, self.budget, self.stop)
        self.assertEqual(status, 'paused')
        self.assertFalse((self.root / 'errors').exists())
        self.assertTrue(ag.read(self.root / 'claims' / (self.pair['key'] + '.json'))['descendants_reaped'])

    def test_held_claim_is_never_touched(self):
        path = self.root / 'claims' / (self.pair['key'] + '.lock')
        path.parent.mkdir()
        with path.open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(ag.work_pair(Mock(), {}, self.pair, self.root / 'audit', self.owner,
                                         self.budget, self.stop), 'locked')
        self.assertFalse(path.with_suffix('.json').exists())

    def test_complete_pair_never_calls_original_combine_or_fitter(self):
        path = self.save_complete()
        before = path.read_bytes(), path.stat().st_mtime_ns
        original = Mock()
        with patch.object(ag, 'run_seed') as fit:
            self.assertEqual(ag.work_pair(original, {}, self.pair, self.root / 'audit', self.owner,
                                         self.budget, self.stop), 'complete')
        original.combine_pair.assert_not_called()
        fit.assert_not_called()
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_pause_releases_lock_only_after_child_run_cleanup(self):
        events = []
        def seed(command, log, fd, budget, stop):
            path = self.root / 'claims' / (self.pair['key'] + '.lock')
            with path.open('a+') as other:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            events.append('child_cleanup_complete')
            return None, 'signal_SIGUSR1'
        with patch.object(ag, 'run_seed', side_effect=seed):
            self.assertEqual(ag.work_pair(Mock(), {}, self.pair, self.root / 'audit', self.owner,
                                         self.budget, self.stop), 'paused')
        with (self.root / 'claims' / (self.pair['key'] + '.lock')).open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            events.append('claim_unlocked')
        self.assertEqual(events, ['child_cleanup_complete', 'claim_unlocked'])

    def test_old_claim_archived_byte_exact_before_takeover(self):
        path = self.root / 'claims' / (self.pair['key'] + '.json')
        ag.publish(path, {'method': 'AutoGluon', 'dataset': 'D', 'job_id': '1', 'state': 'running'})
        before = path.read_bytes()
        with patch.object(ag, 'terminal_evidence', return_value={'terminal': True}), \
                patch.object(ag, 'run_seed', return_value=(None, 'allocation_budget')):
            ag.work_pair(Mock(), {}, self.pair, self.root / 'audit', self.owner, self.budget, self.stop)
        archive = ag.read(next((self.root / 'audit').glob('prior-claim-*.json')))
        self.assertEqual(archive['original_text'].encode(), before)
        self.assertEqual(ag.read(path)['job_id'], '999')

    def test_unproven_owner_never_replaced(self):
        path = self.root / 'claims' / (self.pair['key'] + '.json')
        ag.publish(path, {'method': 'AutoGluon', 'dataset': 'D', 'job_id': '1', 'state': 'running'})
        before = path.read_bytes()
        with patch.object(ag, 'terminal_evidence', side_effect=RuntimeError('RUNNING')):
            result = ag.work_pair(Mock(), {}, self.pair, self.root / 'audit', self.owner, self.budget, self.stop)
        self.assertEqual(result, 'owner_unproven')
        self.assertEqual(path.read_bytes(), before)

    def test_descendant_tree_excludes_other_same_uid_processes(self):
        rows = {2: {'ppid': 1}, 3: {'ppid': 2}, 4: {'ppid': 3}, 5: {'ppid': 99}}
        self.assertEqual(set(ag.descendants(rows, 1)), {2, 3, 4})

    def test_child_cleanup_is_before_run_return(self):
        self.root.joinpath('audit').mkdir()
        child = Mock()
        child.poll.return_value = 0
        events = []
        with patch.object(ag.subprocess, 'Popen', return_value=child), \
                patch.object(ag, 'clean_children', side_effect=lambda c: events.append('reaped')):
            rc, reason = ag.run_seed(['python'], self.root / 'audit/log', 10, self.budget, self.stop)
            events.append('returned')
        self.assertEqual((rc, reason), (0, None))
        self.assertEqual(events, ['reaped', 'returned'])

    def test_signal_racing_child_exit_remains_pause(self):
        child = Mock()
        def poll():
            self.stop.reason = 'signal_SIGUSR1'
            return -signal.SIGUSR1
        child.poll.side_effect = poll
        with patch.object(ag.subprocess, 'Popen', return_value=child), patch.object(ag, 'clean_children'):
            rc, reason = ag.run_seed(['python'], self.root / 'log', 10, self.budget, self.stop)
        self.assertEqual(reason, 'signal_SIGUSR1')


if __name__ == '__main__':
    unittest.main()
