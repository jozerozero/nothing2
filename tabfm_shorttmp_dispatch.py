#!/usr/bin/env python3
"""Versioned operational entry: unchanged TabFM dispatcher, private short TMPDIR.

The scientific manifest, worker, defaults, original data, claims and result
validation remain frozen. The separate runtime plan pins this operational
extension. Only the replacement job recorded by the guarded submission may run.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import tabfm_default_dispatch as frozen

RUN = "shorttmp_208210_v1"
KIND = "tabfm-short-node-local-tmp-operational-v1"
NAMES = ("tabfm_shorttmp_dispatch.py", "tabfm_shorttmp_slurm.sh", "tabfm_replace_208210.py",
         "tabfm_local_tmp.py", "tabfm_default_dispatch.py", "tabfm_submit.py",
         "tabfm_prepare.py", "classification32_submit.py")


def runtime_root(man):
    return Path(man["output_root"]) / "runtime_replacements" / RUN


def load_plan(path, campaign_path):
    plan = frozen.read(path)
    frozen.require(plan.get("plan_id") == frozen.digest({k: v for k, v in plan.items() if k != "plan_id"}),
                   "Operational plan content identity mismatch")
    frozen.require(plan.get("kind") == KIND and plan.get("original_pending_job_id") == "208210",
                   "Wrong operational replacement scope")
    campaign_path = Path(campaign_path).resolve()
    frozen.require(frozen.verify_file(plan["campaign_identity"]) == campaign_path,
                   "Operational plan references another scientific manifest")
    man, tasks = frozen.load_campaign(campaign_path)
    frozen.require(plan["manifest_id"] == man["manifest_id"]
                   and plan["worker_source_sha256"] == man["worker_script"]["sha256"],
                   "Scientific worker/manifest changed")
    root = runtime_root(man)
    frozen.require(Path(path).resolve() == (root / "runtime_plan.json").resolve(), "Unexpected runtime plan location")
    repo = Path(__file__).resolve().parent
    checked = {frozen.verify_file(record) for record in plan["source_records"]}
    frozen.require(len(plan["source_records"]) == len(NAMES) and checked == {repo / name for name in NAMES},
                   "Runtime source identities incomplete/duplicated/unexpected")
    frozen.require(plan["resources"] == {"gpus": 4, "cpus": 16, "mem_gib": 256, "hours": 24},
                   "Replacement resource policy changed")
    frozen.require(plan["operational_change"] == {"child_environment_keys": ["TMPDIR"],
                   "node_local_base": "/tmp", "private_mode": "0700",
                   "scientific_sources_unchanged": True}, "Operational scope broadened")
    return plan, man, tasks


def verify_runtime_job(plan, man, mode, env):
    record = frozen.read(runtime_root(man) / "submitted_new.json")
    job = env.get("SLURM_JOB_ID", "")
    frozen.require(job.isdigit() and job != "208210" and record["job_id"] == job
                   and record["plan_id"] == plan["plan_id"], "Runtime is not the recorded replacement job")
    if mode in ("preflight", "smoke", "run"):
        rank = env.get("SLURM_PROCID", "")
        frozen.require(env.get("SLURM_NTASKS") == "4" and rank in {"0", "1", "2", "3"}
                       and env.get("SLURM_LOCALID") == rank and env.get("SLURM_STEP_ID", "").isdigit(),
                       "Replacement requires four real Slurm ranks, never synthetic rank variables")
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "check", "smoke", "check-smoke", "run", "status"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runtime-plan", type=Path, required=True)
    args = parser.parse_args(argv)
    plan, man, _ = load_plan(args.runtime_plan, args.campaign)
    job = verify_runtime_job(plan, man, args.mode, os.environ)
    if args.mode in ("preflight", "smoke", "run"):
        # Imports above are CPU-only. Activate before the frozen GPU preflight.
        from tabfm_local_tmp import activate
        audit_dir = runtime_root(man) / "runtime" / job / os.environ["SLURM_STEP_ID"] / args.mode / ("rank-" + os.environ["SLURM_PROCID"])
        activate(man, plan, audit_dir)
    frozen.main([args.mode, "--campaign", str(args.campaign.resolve())])


if __name__ == "__main__":
    main()
