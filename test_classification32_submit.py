"""Submission safety tests use a fake scheduler; never contact Slurm."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import classification32_submit as submitter


class Scheduler:
    def __init__(self, bad_field=None, fail_release=None, active=False, uncertain=False):
        self.calls, self.jobs = [], {}
        self.bad_field, self.fail_release = bad_field, fail_release
        self.active, self.uncertain = active, uncertain

    def __call__(self, args):
        self.calls.append(list(args))
        if args[0] == "squeue":
            return "700|c32budget|RUNNING" if self.active else "42|unrelated|RUNNING"
        if args[:3] == ["scontrol", "show", "hostnames"]:
            return "\n".join(sorted(submitter.EXCLUDED_NODES)) if args[3] == submitter.EXCLUDE else args[3]
        if args[0] == "sbatch":
            job = str(1000 + len(self.jobs))
            self.jobs[job] = False
            return "ambiguous response" if self.uncertain else job
        if args[:2] == ["scontrol", "release"]:
            if args[2] == self.fail_release:
                raise RuntimeError("Fake release failure")
            self.jobs[args[2]] = True
            return ""
        if args[:3] == ["scontrol", "show", "job"]:
            job = args[3]
            root, stage = submitter.campaign.ROOT, submitter.campaign.STAGE
            fields = {"JobId": job, "JobName": "c32budget", "Partition": "faculty",
                      "Account": "faculty-acc", "QOS": "bgqos", "Nice": "0", "Requeue": "0",
                      "NumCPUs": "16", "NumTasks": "4", "CPUs/Task": "4", "MinMemoryNode": "256G",
                      "Command": str(stage / "repo/classification32_slurm.sh"), "WorkDir": str(stage / "repo"),
                      "StdOut": str(root / f"logs/job-{job}.out"), "StdErr": str(root / f"logs/job-{job}.err"),
                      "NumNodes": "1", "TimeLimit": "12:00:00", "UserId": "guangyi.chen(123)",
                      "Dependency": "(null)", "NtasksPerN:B:S:C": "4:0:*:*", "ReqTRES": "cpu=16,mem=256G,node=1,gres/gpu=4",
                      "TresPerTask": "cpu=4,gres/gpu=1", "ExcNodeList": submitter.EXCLUDE, "NodeList": "(null)",
                      "JobState": "PENDING", "Reason": "Priority" if self.jobs[job] else "JobHeldUser"}
            if self.bad_field:
                fields[self.bad_field[0]] = self.bad_field[1]
            return " ".join(f"{key}={value}" for key, value in fields.items())
        raise AssertionError(args)


class SubmitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="classification32-submit-")
        folder = Path(self.temp.name).resolve()
        self.root, self.stage = folder / "evaluation", folder / "stage"
        self.root.mkdir()
        (self.stage / "repo").mkdir(parents=True)
        (self.stage / "repo/classification32_slurm.sh").write_text("#!/bin/bash\nexit 0\n")
        self.patches = [patch.object(submitter.campaign, "ROOT", self.root),
                        patch.object(submitter.campaign, "STAGE", self.stage),
                        patch.object(submitter.campaign, "manifest_load", return_value={"manifest_id": "f" * 64})]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def state(self):
        return json.loads((self.root / "submission_state.json").read_text())

    def test_all_four_verified_held_before_release_and_resources_fixed(self):
        scheduler = Scheduler()
        summary = submitter.submit(scheduler)
        self.assertEqual(summary["status"], "submitted_released")
        state = self.state()
        self.assertEqual(len(state["jobs"]), 4)
        first_release = next(i for i, args in enumerate(scheduler.calls) if args[:2] == ["scontrol", "release"])
        self.assertEqual(sum(args[0] == "sbatch" for args in scheduler.calls[:first_release]), 4)
        for index, job in enumerate(state["jobs"]):
            self.assertTrue(job["held_verified"] and job["released"])
            self.assertIn("scontrol_before_release", job)
            self.assertIn("scontrol_after_release", job)
            for option in ("--hold", "--partition=faculty", "--account=faculty-acc", "--qos=bgqos",
                           "--ntasks=4", "--gpus-per-task=1", "--cpus-per-task=4", "--mem=256G",
                           "--time=12:00:00", "--no-requeue", "--nice=0"):
                self.assertIn(option, job["command"])
            self.assertIn(f"--export=ALL,CLASS32_SHARD={index},PYTHONHASHSEED=0", job["command"])
        self.assertFalse(any(args[0] == "scancel" for args in scheduler.calls))

    def test_repeat_is_report_only_without_scheduler_calls(self):
        first = Scheduler()
        submitter.submit(first)
        def forbidden(_args):
            raise AssertionError("Repeated invocation contacted scheduler")
        result = submitter.submit(forbidden)
        self.assertTrue(result["idempotent_noop"])
        self.assertEqual(len(self.state()["jobs"]), 4)

    def test_verification_failure_keeps_immediately_journaled_held_id(self):
        scheduler = Scheduler(bad_field=("NumCPUs", "8"))
        with self.assertRaisesRegex(RuntimeError, "NumCPUs"):
            submitter.submit(scheduler)
        state = self.state()
        self.assertEqual([job["job_id"] for job in state["jobs"]], ["1000"])
        self.assertIn("scontrol_before_release", state["jobs"][0])
        self.assertFalse(any(args[:2] == ["scontrol", "release"] for args in scheduler.calls))
        self.assertTrue(submitter.submit(lambda _: self.fail("No automatic retry"))["idempotent_noop"])

    def test_partial_release_failure_reports_residual_ids_without_cancellation(self):
        scheduler = Scheduler(fail_release="1001")
        with self.assertRaisesRegex(RuntimeError, "release failure"):
            submitter.submit(scheduler)
        summary = submitter.status_summary(self.state())
        self.assertEqual(summary["release_confirmed_job_ids"], ["1000"])
        self.assertEqual(summary["held_or_release_unconfirmed_job_ids"], ["1001", "1002", "1003"])
        self.assertFalse(any(args[0] == "scancel" for args in scheduler.calls))

    def test_active_same_name_blocks_before_journal_or_submission(self):
        scheduler = Scheduler(active=True)
        with self.assertRaisesRegex(RuntimeError, "Existing c32budget"):
            submitter.submit(scheduler)
        self.assertFalse((self.root / "submission_state.json").exists())
        self.assertFalse(scheduler.jobs)

    def test_ambiguous_submission_blocks_automatic_resubmission(self):
        scheduler = Scheduler(uncertain=True)
        with self.assertRaisesRegex(RuntimeError, "uncertain"):
            submitter.submit(scheduler)
        self.assertTrue(self.state()["submission_uncertain"])
        self.assertEqual(self.state()["pending_shard"], 0)
        self.assertTrue(submitter.submit(lambda _: self.fail("Ambiguous attempt must not be repeated"))["idempotent_noop"])

    def test_resource_contract_rejects_wrong_ownership_qos_time_or_gpu(self):
        script = self.stage / "repo/classification32_slurm.sh"
        for key, value in (("UserId", "somebody(1)"), ("QOS", "stqos"), ("TimeLimit", "1-00:00:00"),
                           ("ReqTRES", "cpu=16,mem=256G,node=1,gres/gpu=8"), ("Requeue", "1")):
            scheduler = Scheduler(bad_field=(key, value))
            scheduler.jobs["1000"] = False
            raw = scheduler(["scontrol", "show", "job", "1000", "-o"])
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                submitter.verify_job("1000", raw, script, held=True, run=scheduler)


if __name__ == "__main__":
    unittest.main()
