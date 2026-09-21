"""CPU/stdlib-only deadline tests: no Slurm, remote access, or filesystem writes."""
import contextlib
import io
import json
import math
import subprocess
import unittest
from unittest import mock

import table6_restart_deadline as deadline


class DurationTests(unittest.TestCase):
    def test_supported_formats(self):
        for text, seconds in (("02:00:00", 7200), ("00:00:00", 0),
                              ("1-02:03:04", 93784), ("72:00:00", 259200),
                              ("12:34", 754), ("0-23:59:59", 86399)):
            with self.subTest(text=text):
                self.assertEqual(deadline.parse_duration(text), seconds)

    def test_reject_unknown_unlimited_negative_ambiguous_and_malformed(self):
        for text in ("UNLIMITED", "UNKNOWN", "Unknown", "N/A", "7200", "-1:00:00",
                     "00:60:00", "00:00:60", "1-24:00:00", "1-02:03", "", None):
            with self.subTest(text=text):
                with self.assertRaises(deadline.DeadlineError):
                    deadline.parse_duration(text)

    def test_end_time_timezone_is_explicit(self):
        utc = deadline.parse_end_time("2026-09-14T22:59:07")
        self.assertEqual(utc, 1789426747)
        self.assertEqual(utc, deadline.parse_end_time("2026-09-14T22:59:07Z"))
        self.assertEqual(utc, deadline.parse_end_time("2026-09-15T02:59:07+04:00"))


class DeadlineTests(unittest.TestCase):
    def fields(self, **overrides):
        fields = {"JobId": "188641", "JobState": "RUNNING", "TimeLimit": "02:00:00",
                  "RunTime": "00:00:00", "EndTime": "2026-09-14T22:59:07",
                  "NumTasks": "8", "CPUs/Task": "8", "NumNodes": "1"}
        fields.update(overrides)
        return " ".join(f"{key}={value}" for key, value in fields.items())

    def derive(self, raw=None, **kwargs):
        options = dict(job_id="188641", query_started_monotonic=100,
                       query_finished_monotonic=102, observed_local_epoch=1789419328.3347573,
                       hostname="node294")
        options.update(kwargs)
        return deadline.derive_deadline(self.fields() if raw is None else raw, **options)

    def test_recorded_7418_second_failure_is_diagnostic_not_new_budget(self):
        value = self.derive()
        self.assertAlmostEqual(value.slurm_end_minus_local_epoch_seconds, 7418.6652427, places=5)
        self.assertAlmostEqual(value.slurm_end_vs_runtime_projection_seconds, 218.6652427, places=5)
        self.assertEqual(value.safe_remaining_seconds, 7200 - 2 - 300)
        self.assertEqual(value.monotonic_deadline, 7000)
        self.assertAlmostEqual(value.local_epoch_deadline - value.observed_local_epoch, 6898)
        self.assertLess(value.local_epoch_deadline, value.slurm_end_epoch)

    def test_existing_runtime_and_query_latency_both_subtracted(self):
        value = self.derive(self.fields(RunTime="00:10:00"), query_finished_monotonic=107.5)
        self.assertEqual(value.safe_remaining_seconds, 7200 - 600 - 7.5 - 300)

    def test_day_form_explicit_limit_contract(self):
        value = self.derive(self.fields(TimeLimit="1-02:00:00", RunTime="1-00:00:00"),
                            expected_limit_seconds=26 * 3600)
        self.assertEqual(value.safe_remaining_seconds, 6898)

    def test_state_identity_and_limit_contracts_fail_closed(self):
        for fields in ({"JobState": "PENDING"}, {"JobState": "COMPLETED"},
                       {"JobId": "188642"}, {"TimeLimit": "03:00:00"},
                       {"TimeLimit": "UNLIMITED"}, {"RunTime": "UNKNOWN"},
                       {"EndTime": "Unknown"}):
            with self.subTest(fields=fields):
                with self.assertRaises(deadline.DeadlineError):
                    self.derive(self.fields(**fields))

    def test_missing_duplicate_and_multiple_job_responses_rejected(self):
        for raw in (self.fields().replace("RunTime=00:00:00", ""),
                    self.fields() + " RunTime=00:00:01", self.fields() + "\n" + self.fields()):
            with self.assertRaises(deadline.DeadlineError):
                self.derive(raw)

    def test_expired_and_insufficient_remaining_rejected(self):
        for runtime in ("02:00:00", "02:00:01", "01:58:00", "01:53:00"):
            with self.subTest(runtime=runtime):
                with self.assertRaises(deadline.DeadlineError):
                    self.derive(self.fields(RunTime=runtime))

    def test_exact_minimum_boundary_is_closed(self):
        # 7200 - 6778 - 2 - 300 = 120, the required minimum is exclusive.
        with self.assertRaises(deadline.DeadlineError):
            self.derive(self.fields(RunTime="01:52:58"))
        self.assertEqual(self.derive(self.fields(RunTime="01:52:57")).safe_remaining_seconds, 121)

    def test_invalid_margin_clocks_and_job_ids_rejected(self):
        for options in ({"safety_margin_seconds": 299}, {"safety_margin_seconds": math.nan},
                        {"observed_local_epoch": math.inf}, {"query_finished_monotonic": 99},
                        {"query_started_monotonic": -1}, {"job_id": ""},
                        {"job_id": "188641;sbatch"}, {"minimum_remaining_seconds": -1}):
            with self.subTest(options=options):
                with self.assertRaises(deadline.DeadlineError):
                    self.derive(**options)

    def test_resources_are_allocation_wide_not_single_rank_ownership(self):
        value = self.derive(self.fields(NumTasks="8", NumCPUs="64", TresPerTask="gres/gpu=1"))
        self.assertEqual(value.job_id, "188641")

    def test_end_time_past_local_wall_does_not_override_runtime_source(self):
        value = self.derive(observed_local_epoch=1789426757)
        self.assertEqual(value.slurm_end_minus_local_epoch_seconds, -10)
        self.assertEqual(value.safe_remaining_seconds, 6898)

    def test_monotonic_preferred_even_after_wall_clock_jump(self):
        value = self.derive()
        env = {**value.environment(), "SLURM_JOB_ID": "188641"}
        budget = deadline.EnvironmentBudget.from_environment(
            env, hostname="node294", wall_clock=lambda: value.local_epoch_deadline + 100000,
            monotonic_clock=lambda: 200)
        self.assertEqual(budget.hard_end_epoch, value.local_epoch_deadline)
        self.assertEqual(budget.remaining(), 6800)

    def test_late_rank_does_not_receive_fresh_duration(self):
        env = self.derive().environment()
        late = deadline.EnvironmentBudget.from_environment(
            env, hostname="node294", monotonic_clock=lambda: 6990)
        self.assertEqual(late.remaining(), 10)
        ended = deadline.EnvironmentBudget.from_environment(
            env, hostname="node294", monotonic_clock=lambda: 7001)
        self.assertEqual(ended.remaining(), -1)

    def test_explicit_legacy_epoch_supported_but_no_fallback(self):
        budget = deadline.EnvironmentBudget.from_environment(
            {"JOB_BUDGET_END_EPOCH": "1000"}, wall_clock=lambda: 900)
        self.assertEqual(budget.remaining(), 100)
        for env in ({}, {"JOB_BUDGET_END_EPOCH": "nan"}):
            with self.assertRaises(deadline.DeadlineError):
                deadline.EnvironmentBudget.from_environment(env)

    def test_different_node_or_allocation_rejected(self):
        env = self.derive().environment()
        with self.assertRaises(deadline.DeadlineError):
            deadline.EnvironmentBudget.from_environment(env, hostname="node295")
        with self.assertRaises(deadline.DeadlineError):
            deadline.EnvironmentBudget.from_environment({**env, "SLURM_JOB_ID": "9"}, hostname="node294")

    def test_query_is_single_read_only_response_with_explicit_utc(self):
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 0, self.fields(), ""))
        with mock.patch.dict(deadline.os.environ, {"TZ": "Asia/Dubai", "SLURM_TIME_FORMAT": "relative"}):
            value = deadline.query_deadline("188641", run=run, wall_clock=lambda: 1789419328.3347573,
                                            monotonic_clock=mock.Mock(side_effect=[100, 102]))
        run.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["scontrol", "show", "job", "-o", "188641"])
        self.assertEqual(kwargs["env"]["TZ"], "UTC")
        self.assertNotIn("SLURM_TIME_FORMAT", kwargs["env"])
        self.assertEqual(value.query_elapsed_seconds, 2)

    def test_query_error_and_timeout_never_grant_budget(self):
        for run in (mock.Mock(return_value=subprocess.CompletedProcess([], 1, "", "unknown job")),
                    mock.Mock(side_effect=subprocess.TimeoutExpired("scontrol", 30))):
            with self.assertRaises(deadline.DeadlineError):
                deadline.query_deadline("188641", run=run)

    def test_cli_exports_and_failure_exit(self):
        output, error = io.StringIO(), io.StringIO()
        with mock.patch.object(deadline, "query_deadline", return_value=self.derive()), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            self.assertEqual(deadline.main(["--job-id", "188641", "--format", "exports"]), 0)
        self.assertIn("export JOB_BUDGET_END_MONOTONIC=7000.000000000", output.getvalue())
        self.assertEqual(json.loads(error.getvalue())["job_id"], "188641")
        output, error = io.StringIO(), io.StringIO()
        with mock.patch.object(deadline, "query_deadline", side_effect=deadline.DeadlineError("expired")), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            self.assertEqual(deadline.main(["--job-id", "188641"]), 2)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(json.loads(error.getvalue())["event"], "allocation_deadline_rejected")


if __name__ == "__main__":
    unittest.main()
