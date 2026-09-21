#!/usr/bin/env python3
"""Manually prepare/replace only owned pending208210 with a verified held job.

No parent allocations are touched. New submission failure leaves old208210
untouched. Every transition is immutable; ambiguous outcomes stop for inspection.
No automatic retry, cleanup/cancellation of candidates, or journal overwrite.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

import tabfm_default_dispatch as frozen
import tabfm_submit as original
from tabfm_prepare import OUT, identity
from tabfm_shorttmp_dispatch import KIND, NAMES, load_plan, runtime_root

OLD = "208210"


def command(argv):
    return subprocess.check_output(argv, text=True, timeout=60).strip()


def old_record(man, held=False):
    repo = Path(__file__).resolve().parent
    raw = command(["scontrol", "show", "job", OLD, "-o"])
    checked = original.verify(OLD, repo / "tabfm_default_slurm.sh", raw, held=held)
    frozen.require(checked.get("JobState") == "PENDING", "Old208210 is no longer pending; do not touch it")
    # A running sidecar/parent is never an eligible replacement target.
    frozen.require(checked["Command"] == str(repo / "tabfm_default_slurm.sh")
                   and Path(man["output_root"]).resolve() == OUT.resolve(), "Wrong original campaign/job")
    return {"job_id": OLD, "raw": raw, "fields": checked, "epoch": time.time()}


def write(man, root, name, value):
    frozen.atomic(man, root / name, value)


def prepare(campaign_path):
    man, _ = frozen.load_campaign(campaign_path)
    root, repo = runtime_root(man), Path(__file__).resolve().parent
    pending = old_record(man)
    plan = {"schema": 1, "kind": KIND, "original_pending_job_id": OLD,
            "campaign_identity": identity(campaign_path), "manifest_id": man["manifest_id"],
            "worker_source_sha256": man["worker_script"]["sha256"],
            "source_records": [identity(repo / name) for name in NAMES],
            "operational_change": {"child_environment_keys": ["TMPDIR"], "node_local_base": "/tmp",
                                   "private_mode": "0700", "scientific_sources_unchanged": True},
            "resources": {"gpus": 4, "cpus": 16, "mem_gib": 256, "hours": 24},
            "created_epoch": time.time(), "old_pending_snapshot": pending,
            "reason": "Valid short node-local TMPDIR avoids observed pre-model HIP allocation crash; frozen scientific protocol unchanged"}
    plan["plan_id"] = frozen.digest(plan)
    write(man, root, "runtime_plan.json", plan)
    print(json.dumps({"runtime_plan": str(root / "runtime_plan.json"), "plan_id": plan["plan_id"],
                      "state": "prepared_only_no_submission_or_job_change"}), flush=True)


def submission_command(man, root, repo, campaign_path):
    return ["sbatch", "--hold", "--parsable", "--job-name=tabfm681", "--partition=faculty",
            "--account=faculty-acc", "--qos=bgqos", "--nodes=1", "--ntasks=4", "--ntasks-per-node=4",
            "--gpus-per-task=1", "--cpus-per-task=4", "--mem=256G", "--time=1-00:00:00",
            "--no-requeue", "--nice=0", "--exclude=" + original.EXCLUDE, "--chdir=" + str(repo),
            "--output=" + str(OUT / "logs/job-%j.out"), "--error=" + str(OUT / "logs/job-%j.err"),
            "--export=ALL,PYTHONHASHSEED=0", str(repo / "tabfm_shorttmp_slurm.sh"),
            str(campaign_path), str(root / "runtime_plan.json")]


def replace(campaign_path):
    # Never re-enter after a partial/ambiguous transaction. Inspect its receipts.
    man, _ = frozen.load_campaign(campaign_path)
    root, repo = runtime_root(man), Path(__file__).resolve().parent
    plan, man, _ = load_plan(root / "runtime_plan.json", campaign_path)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "transaction.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen.require(not (root / "submission_attempt.json").exists(), "Replacement attempted; inspect, do not retry")
        before = old_record(man)
        argv = submission_command(man, root, repo, Path(campaign_path).resolve())
        write(man, root, "submission_attempt.json", {"plan_id": plan["plan_id"], "epoch": time.time(),
              "old_pending_snapshot": before, "command": argv, "old_untouched_until_new_verified_held": True})
        stage = "submit_new_held"
        try:
            answer = command(argv)
            match = re.fullmatch(r"([0-9]+)(?:;[^\s;]+)?", answer)
            frozen.require(match is not None, "Ambiguous sbatch response; do not retry or cancel any job")
            job = match.group(1)
            frozen.require(job != OLD, "Scheduler returned original job identity")
            write(man, root, "submitted_new.json", {"job_id": job, "plan_id": plan["plan_id"],
                  "manifest_id": man["manifest_id"], "epoch": time.time(), "sbatch_response": answer})
            stage = "verify_new_held"
            raw = command(["scontrol", "show", "job", job, "-o"])
            checked = original.verify(job, repo / "tabfm_shorttmp_slurm.sh", raw, held=True)
            write(man, root, "verified_new_held.json", {"job_id": job, "plan_id": plan["plan_id"],
                  "raw": raw, "fields": checked, "epoch": time.time()})
            stage = "hold_old_pending"
            pending = old_record(man)
            write(man, root, "old_hold_intent.json", {"job_id": OLD, "epoch": time.time(), "snapshot": pending})
            command(["scontrol", "hold", OLD])
            held = old_record(man, held=True)
            write(man, root, "old_held.json", held)
            stage = "cancel_old_only"
            write(man, root, "old_cancel_intent.json", {"job_id": OLD, "epoch": time.time(), "new_verified_job_id": job})
            command(["scancel", OLD])
            old_raw = command(["scontrol", "show", "job", OLD, "-o"])
            old_fields = dict(re.findall(r"([^\s=]+)=([^\s]+)", old_raw))
            frozen.require(old_fields.get("JobId") == OLD and old_fields.get("JobState") == "CANCELLED",
                           "Old cancellation not confirmed; keep replacement held")
            write(man, root, "old_cancelled.json", {"job_id": OLD, "raw": old_raw, "fields": old_fields,
                  "epoch": time.time(), "new_verified_job_id": job})
            stage = "reverify_new_before_release"
            raw = command(["scontrol", "show", "job", job, "-o"])
            original.verify(job, repo / "tabfm_shorttmp_slurm.sh", raw, held=True)
            # Recheck all pinned runtime/scientific sources before release.
            load_plan(root / "runtime_plan.json", campaign_path)
            write(man, root, "new_release_intent.json", {"job_id": job, "epoch": time.time(), "plan_id": plan["plan_id"]})
            stage = "release_new"
            command(["scontrol", "release", job])
            raw = command(["scontrol", "show", "job", job, "-o"])
            checked = original.verify(job, repo / "tabfm_shorttmp_slurm.sh", raw, held=False)
            frozen.require(checked.get("JobState") in ("PENDING", "RUNNING", "CONFIGURING"),
                           "Replacement not in an active schedulable state")
            write(man, root, "released_new.json", {"job_id": job, "plan_id": plan["plan_id"], "raw": raw,
                  "fields": checked, "epoch": time.time(), "replaced_job_id": OLD})
            print(json.dumps({"replacement_job_id": job, "replaced_job_id": OLD,
                              "state": checked["JobState"], "plan_id": plan["plan_id"]}), flush=True)
        except BaseException as exc:
            write(man, root, "transaction_error.json", {"plan_id": plan["plan_id"], "stage": stage,
                  "epoch": time.time(), "error": repr(exc), "traceback": traceback.format_exc(),
                  "automatic_cleanup_or_retry_performed": False})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "replace"))
    parser.add_argument("--campaign", type=Path, default=OUT / "manifest.json")
    args = parser.parse_args(argv)
    (prepare if args.mode == "prepare" else replace)(args.campaign.resolve())


if __name__ == "__main__":
    main()
