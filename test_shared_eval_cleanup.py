"""Owned-child cleanup regression tests; mocks only, no real process signals."""
import contextlib
import os
from pathlib import Path
import signal
import types
import unittest
from unittest.mock import Mock, call, patch

import shared_eval_launch as launcher


class GoneProcess(Exception):
    pass


class EndMockHold(BaseException):
    """Test harness only: stand in for scheduler SIGKILL after proving wait."""


class CleanupTests(unittest.TestCase):
    def runtime(self, snapshots, waitpid=None):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        me = Mock()
        me.children.side_effect = snapshots
        psutil = types.SimpleNamespace(Process=Mock(return_value=me),
                                       NoSuchProcess=GoneProcess, wait_procs=Mock(),
                                       process_iter=Mock(side_effect=AssertionError('must not enumerate peer processes')))
        stack.enter_context(patch.dict('sys.modules', {'psutil': psutil}))
        wait = stack.enter_context(patch.object(launcher.os, 'waitpid',
                                                side_effect=waitpid or ChildProcessError()))
        return me, psutil, wait

    def test_term_cleans_reverse_descendant_order_and_reaps(self):
        events = []
        leader = Mock(send_signal=Mock(side_effect=lambda sig: events.append(('leader', sig))))
        fit = Mock(send_signal=Mock(side_effect=lambda sig: events.append(('fit', sig))))
        me, psutil, wait = self.runtime([[leader, fit], []], [(22, 0), (0, 0)])
        launcher.cleanup_owned_descendants()
        self.assertEqual(events, [('fit', signal.SIGTERM), ('leader', signal.SIGTERM)])
        psutil.Process.assert_called_once_with(os.getpid())
        psutil.process_iter.assert_not_called()
        psutil.wait_procs.assert_called_once_with([leader, fit], timeout=10)
        self.assertEqual(me.children.call_args_list, [call(recursive=True), call(recursive=True)])
        self.assertEqual(wait.call_args_list, [call(-1, os.WNOHANG), call(-1, os.WNOHANG)])

    def test_kill_rescans_and_includes_newly_adopted_session(self):
        leader, orphan_fit = Mock(), Mock()
        me, psutil, _ = self.runtime([[leader], [orphan_fit], [orphan_fit], []])
        launcher.cleanup_owned_descendants()
        leader.send_signal.assert_called_once_with(signal.SIGTERM)
        orphan_fit.send_signal.assert_called_once_with(signal.SIGKILL)
        self.assertEqual(psutil.wait_procs.call_args_list,
                         [call([leader], timeout=10), call([orphan_fit], timeout=10)])
        self.assertEqual(me.children.call_count, 4)

    def test_exited_process_race_is_tolerated_not_arbitrary_failure(self):
        child = Mock()
        child.send_signal.side_effect = GoneProcess()
        self.runtime([[child], []])
        launcher.cleanup_owned_descendants()

    def test_uninspectable_children_fail_closed(self):
        self.runtime(PermissionError('cannot inspect own descendants'))
        with self.assertRaises(PermissionError):
            launcher.cleanup_owned_descendants()

    def test_permission_error_signalling_owned_child_is_not_swallowed(self):
        child = Mock()
        child.send_signal.side_effect = PermissionError('cannot signal owned child')
        self.runtime([[child]])
        with self.assertRaises(PermissionError):
            launcher.cleanup_owned_descendants()

    def test_surviving_child_cannot_be_reported_clean(self):
        child = Mock()
        self.runtime([[child], [child], [child], [child], [child]])
        with self.assertRaisesRegex(RuntimeError, 'owned descendants survived shutdown'):
            launcher.cleanup_owned_descendants()
        self.assertEqual(child.send_signal.call_args_list, [call(signal.SIGTERM), call(signal.SIGKILL)])

    def test_subreaper_must_be_enabled_successfully(self):
        libc = Mock()
        libc.prctl.return_value = 0
        with patch.object(launcher.ctypes, 'CDLL', return_value=libc) as load:
            launcher.enable_subreaper()
        load.assert_called_once_with(None, use_errno=True)
        libc.prctl.assert_called_once_with(36, 1, 0, 0, 0)
        libc.prctl.return_value = -1
        with patch.object(launcher.ctypes, 'CDLL', return_value=libc), self.assertRaises(RuntimeError):
            launcher.enable_subreaper()

    def test_failed_cleanup_holds_instead_of_returning_or_rethrowing(self):
        with patch.object(launcher, 'cleanup_owned_descendants', side_effect=PermissionError('unknown descendants')), \
                patch.object(launcher, 'publish') as publish, \
                patch.object(launcher.signal, 'pause', side_effect=[None, None, EndMockHold()]) as pause:
            with self.assertRaises(EndMockHold):
                launcher.cleanup_or_hold(Path('/audit'), {'plan_id': 'p'})
        self.assertEqual(pause.call_count, 3)
        self.assertEqual(publish.call_args.args[0], Path('/audit/cleanup-fatal.json'))
        self.assertEqual(publish.call_args.args[1]['state'], 'lease_retained_waiting_for_step_cgroup_teardown')

    def test_unwritable_cleanup_receipt_still_holds_lease(self):
        with patch.object(launcher, 'cleanup_owned_descendants', side_effect=RuntimeError('survivor')), \
                patch.object(launcher, 'publish', side_effect=OSError('disk full')), \
                patch.object(launcher.signal, 'pause', side_effect=EndMockHold()) as pause:
            with self.assertRaises(EndMockHold):
                launcher.cleanup_or_hold(Path('/audit'), {})
        pause.assert_called_once()


if __name__ == '__main__':
    unittest.main()
