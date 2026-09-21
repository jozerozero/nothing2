#!/usr/bin/env python3
"""Submit only this new actual32 campaign: four audited, initially-held jobs.

The campaign manifest must already have been prepared. No preparation, remote
transport, retries, job cancellation, or mutation of other jobs occurs here.
An immutable initial journal plus an exclusive lock makes repeated invocation
a report-only no-op, including after partial or uncertain submission failure.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time

import classification32_campaign as campaign


JOB_NAME = "c32budget"
EXCLUDED_IDS = (193, 195, 207, 216, 228, 239, 274, 287, 292, 296)
EXCLUDED_NODES = {f"auh7-1b-gpu-{number}" for number in EXCLUDED_IDS}
EXCLUDE = "auh7-1b-gpu-[" + ",".join(map(str, EXCLUDED_IDS)) + "]"
require = campaign.require


def command(args):
    return subprocess.check_output(args, text=True).strip()


def fields_from_raw(raw):
    return dict(re.findall(r"([^\s=]+)=([^\s]+)", raw))


def require_no_existing_jobs(run=command):
    raw = run(["squeue", "--me", "-h", "-o", "%i|%j|%T"])
    matches = [line for line in raw.splitlines() if len(line.split("|")) >= 2
               and line.split("|")[1] == JOB_NAME]
    require(not matches, "Existing c32budget jobs require inspection; refusing duplicate campaign: " + "; ".join(matches))


def verify_job(job_id, raw, script, *, held, run=command):
    fields = fields_from_raw(raw)
    expected = {"JobId": str(job_id), "JobName": JOB_NAME, "Partition": "faculty",
                "Account": "faculty-acc", "QOS": "bgqos", "Nice": "0", "Requeue": "0",
                "NumCPUs": "16", "NumTasks": "4", "CPUs/Task": "4", "MinMemoryNode": "256G",
                "Command": str(script), "WorkDir": str(campaign.STAGE / "repo"),
                "StdOut": str(campaign.ROOT / "logs" / f"job-{job_id}.out"),
                "StdErr": str(campaign.ROOT / "logs" / f"job-{job_id}.err")}
    for key, value in expected.items():
        require(fields.get(key) == value, f"Job {job_id} resource contract mismatch: {key}={fields.get(key)!r}, expected {value!r}")
    require(fields.get("NumNodes") in ("1", "1-1"), f"Job {job_id} must reserve one node")
    require(fields.get("TimeLimit") in ("12:00:00", "0-12:00:00"), f"Job {job_id} time limit must be12h")
    require(fields.get("UserId", "").startswith("guangyi.chen("), f"Job {job_id} ownership mismatch")
    require(fields.get("Dependency") == "(null)", f"Job {job_id} has unexpected dependencies")
    require(fields.get("NtasksPerN:B:S:C", "").split(":")[0] == "4", f"Job {job_id} must run four tasks per node")
    require("gres/gpu=4" in fields.get("ReqTRES", "").split(","), f"Job {job_id} must request four GPUs")
    require(fields.get("TresPerTask") == "cpu=4,gres/gpu=1", f"Job {job_id} must reserve one GPU/four CPUs per task")
    excluded = set(run(["scontrol", "show", "hostnames", fields.get("ExcNodeList", "")]).splitlines())
    require(EXCLUDED_NODES <= excluded, f"Job {job_id} lost required excluded nodes")
    for key in ("NodeList", "SchedNodeList"):
        if fields.get(key) not in (None, "(null)", "None"):
            assigned = set(run(["scontrol", "show", "hostnames", fields[key]]).splitlines())
            require(not assigned & EXCLUDED_NODES, f"Job {job_id} was assigned an excluded node")
    if held:
        require(fields.get("JobState") == "PENDING" and fields.get("Reason") == "JobHeldUser",
                f"Job {job_id} is not safely held before release")
    return fields


def sbatch_command(script, shard):
    require(shard in range(4), "Shard must be0..3")
    return ["sbatch", "--hold", "--parsable", "--job-name=" + JOB_NAME,
            "--partition=faculty", "--account=faculty-acc", "--qos=bgqos", "--nodes=1",
            "--ntasks=4", "--ntasks-per-node=4", "--gpus-per-task=1", "--cpus-per-task=4",
            "--mem=256G", "--time=12:00:00", "--no-requeue", "--nice=0", "--exclude=" + EXCLUDE,
            "--chdir=" + str(campaign.STAGE / "repo"),
            "--output=" + str(campaign.ROOT / "logs/job-%j.out"),
            "--error=" + str(campaign.ROOT / "logs/job-%j.err"),
            f"--export=ALL,CLASS32_SHARD={shard},PYTHONHASHSEED=0", str(script), str(shard)]


def status_summary(state):
    jobs = state.get("jobs", [])
    released = [job["job_id"] for job in jobs if job.get("released") is True]
    unresolved = [job["job_id"] for job in jobs if job.get("released") is not True]
    return {"status": state.get("status"), "recorded_job_ids": [job["job_id"] for job in jobs],
            "release_confirmed_job_ids": released, "held_or_release_unconfirmed_job_ids": unresolved,
            "submission_uncertain": state.get("submission_uncertain", False),
            "error": state.get("error"), "automatic_retry": False,
            "note": "Journal state only; no cancellations performed. Inspect scheduler before any manual recovery."}


def submit(run=command):
    root = campaign.ROOT
    require(root.is_dir(), "New campaign root absent; prepare must run separately first")
    state_path = root / "submission_state.json"
    lock_path = root / "submission.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            state = campaign.read(state_path)
            require(state.get("campaign") == campaign.NAME and state.get("job_name") == JOB_NAME,
                    "Existing submission journal belongs to another campaign")
            return {"idempotent_noop": True, **status_summary(state)}
        manifest = campaign.manifest_load()
        script = (campaign.STAGE / "repo/classification32_slurm.sh").resolve(strict=True)
        require(script.is_file() and script.read_text().startswith("#!"), "Main-provided Slurm shell is missing/invalid")
        require_no_existing_jobs(run)
        (root / "logs").mkdir(exist_ok=True)
        state = {"campaign": campaign.NAME, "job_name": JOB_NAME, "status": "preparing_submission",
                 "manifest_id": manifest["manifest_id"], "script": campaign.identity(script),
                 "started_epoch": time.time(), "expected_jobs": 4, "jobs": [],
                 "resources_per_job": {"nodes": 1, "tasks": 4, "gpus": 4, "cpus": 16,
                     "ram_gib": 256, "hours": 12, "partition": "faculty", "account": "faculty-acc",
                     "qos": "bgqos", "nice": 0, "requeue": False},
                 "excluded_nodes": sorted(EXCLUDED_NODES), "submission_uncertain": False}
        campaign.atomic(state_path, state, immutable=True)
        try:
            for shard in range(4):
                campaign.verify(state["script"], full=True)
                state.update(status="submitting_held", pending_shard=shard, submission_uncertain=True)
                campaign.atomic(state_path, state, immutable=False)
                args = sbatch_command(script, shard)
                raw_submit = run(args)
                match = re.fullmatch(r"([0-9]+)(?:;[^\s;]+)?", raw_submit)
                require(match is not None, "Unparseable sbatch response; submission outcome uncertain: " + raw_submit)
                job_id = match.group(1)
                require(job_id not in {job["job_id"] for job in state["jobs"]}, "Scheduler returned a duplicate job ID")
                job = {"job_id": job_id, "shard": shard, "command": args,
                       "submitted_epoch": time.time(), "sbatch_response": raw_submit,
                       "released": False, "held_verified": False}
                state["jobs"].append(job)
                state.update(submission_uncertain=False)
                # Persist the returned held ID BEFORE any verification can fail.
                campaign.atomic(state_path, state, immutable=False)
                raw = run(["scontrol", "show", "job", job_id, "-o"])
                job["scontrol_before_release"] = raw
                campaign.atomic(state_path, state, immutable=False)
                job["verified_contract"] = verify_job(job_id, raw, script, held=True, run=run)
                job["held_verified"] = True
                campaign.atomic(state_path, state, immutable=False)
            require(len(state["jobs"]) == 4 and all(job["held_verified"] for job in state["jobs"]),
                    "All four held jobs must pass verification before any release")
            state.update(status="verified_all_held", pending_shard=None)
            campaign.atomic(state_path, state, immutable=False)
            # Re-check all four immediately before releasing the first job.
            for job in state["jobs"]:
                raw = run(["scontrol", "show", "job", job["job_id"], "-o"])
                job["scontrol_final_held_check"] = raw
                campaign.atomic(state_path, state, immutable=False)
                verify_job(job["job_id"], raw, script, held=True, run=run)
            for job in state["jobs"]:
                state.update(status="releasing", pending_release_job_id=job["job_id"])
                campaign.atomic(state_path, state, immutable=False)
                run(["scontrol", "release", job["job_id"]])
                job.update(released=True, released_epoch=time.time())
                campaign.atomic(state_path, state, immutable=False)
                raw = run(["scontrol", "show", "job", job["job_id"], "-o"])
                job["scontrol_after_release"] = raw
                campaign.atomic(state_path, state, immutable=False)
                verified = verify_job(job["job_id"], raw, script, held=False, run=run)
                require(verified.get("Reason") != "JobHeldUser", "Scheduler still reports user hold after release")
            state.update(status="submitted_released", completed_epoch=time.time(), pending_release_job_id=None)
            campaign.atomic(state_path, state, immutable=False)
        except Exception as exc:
            state.update(status="failed_requires_manual_inspection", error=f"{type(exc).__name__}: {exc}",
                         failed_epoch=time.time())
            campaign.atomic(state_path, state, immutable=False)
            print(json.dumps(status_summary(state)), flush=True)
            raise
        return status_summary(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("submit", "status"))
    args = parser.parse_args()
    if args.mode == "status":
        print(json.dumps(status_summary(campaign.read(campaign.ROOT / "submission_state.json"))), flush=True)
    else:
        print(json.dumps(submit()), flush=True)


if __name__ == "__main__":
    main()
