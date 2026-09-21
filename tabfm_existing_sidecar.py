#!/usr/bin/env python3
"""Guarded low-priority single-GPU TabFM lane in authorized parent196092.

No frozen sources, manifest, model defaults, allocation or other job is changed.
Sequential full-data smoke precedes shared formal claims. Operational memory
guards may stop this lane, never reduce context/ensemble size or retry a claim.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import traceback
import tabfm_default_dispatch as frozen

GIB = 1024**3
PARENT = "196092"
NODE = "auh7-1b-gpu-196"
GPU = {"uuid": "8b8b827ace9944c1", "pci": "0000:88:00.0"}
ACTIVE_STEP = None
require = frozen.require


class ResourceGuardError(RuntimeError):
    pass


def load_plan(path, fresh=False):
    p = frozen.read(path)
    require(p.get("plan_id") == frozen.digest({k: v for k, v in p.items() if k != "plan_id"}), "Plan digest mismatch")
    require(str(p["parent_job_id"]) == PARENT and p["node"] == NODE and p["gpus"] == [GPU],
            "Only authorized parent196092/node196/single pinned GPU is allowed")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", p["sidecar_id"]), "Unsafe sidecar id")
    require(p["max_lanes"] == 1 and p["cpu_count"] == 4 and p["mem_gib"] == 40
            and p["parent_gpu_count"] == 8 and p["parent_mem_gib"] == 64
            and p["all8_gpu_reservation_verified"] is True and p["resources_available_verified"] is True,
            "Requires one4CPU/40GiB child inside verified64GiB/eight-GPU parent")
    require(p["free_cpu_cores"] >= 4 and 0 <= p["startup_other_rss_bytes"] <= 20 * GIB, "Insufficient CPU/RSS headroom")
    now = time.time()
    require(math.isfinite(p["deadline_epoch"]) and p["deadline_epoch"] > now, "Sidecar deadline expired")
    probes = p["external_idle_probes"]
    require(len(probes) == 2 and probes[1]["epoch"] - probes[0]["epoch"] >= 15, "Need two idle probes>=15s apart")
    for item in probes:
        require(item["uuid"] == GPU["uuid"] and item["pci"] == GPU["pci"] and item["gpu_busy_percent"] == 0
                and 0 <= item["used_vram_bytes"] < 128 * 1024**2, "External probe GPU not genuinely idle")
    if fresh:
        require(0 <= now - p["probe_epoch"] <= 300 and 0 <= now - probes[-1]["epoch"] <= 300
                and p["deadline_epoch"] - now > 3 * 3600, "Stale probe or parent remaining<=3h")
    require(frozen.verify_file(p["sidecar_source"]) == Path(__file__).resolve(), "Sidecar source identity mismatch")
    for source in p.get("source_records", []):
        frozen.verify_file(source)
    man, tasks = frozen.load_campaign(p["campaign_path"])
    require("manifest_id" not in p or p["manifest_id"] == man["manifest_id"], "Plan campaign mismatch")
    return p, man, tasks


def sidecar_root(plan, man):
    return Path(man["output_root"]) / "sidecars" / plan["sidecar_id"]


def memory_gib(value):
    match = re.fullmatch(r"([0-9.]+)([KMGT]?)", value)
    require(match is not None, "Unknown Slurm memory quantity")
    number, suffix = match.groups()
    return float(number) * {"K": 1/1024**2, "": 1/1024, "M": 1/1024, "G": 1, "T": 1024}[suffix]


def verify_parent(plan, raw):
    fields = dict(re.findall(r"([^\s=]+)=([^\s]+)", raw))
    require(fields.get("JobId") == PARENT and fields.get("JobState") == "RUNNING"
            and fields.get("NumNodes") == "1" and fields.get("NodeList") == NODE, "Parent allocation changed")
    require(fields.get("UserId", "").startswith("guangyi.chen("), "Wrong parent user")
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    require(int(tres.get("gres/gpu", "0")) == 8 and int(tres.get("cpu", "0")) == 64
            and memory_gib(tres.get("mem", "0")) == 64, "Parent GPU/CPU/64GiB reservation changed")
    require(fields.get("EndTime") not in (None, "Unknown", "N/A", "UNLIMITED"), "Parent expiry unknown")
    parsed_end = datetime.datetime.fromisoformat(fields["EndTime"])
    if parsed_end.tzinfo is None:
        parsed_end = parsed_end.replace(tzinfo=datetime.timezone.utc)
    end = parsed_end.timestamp()
    require(plan["deadline_epoch"] <= end - 30, "Sidecar deadline must precede parent expiry")
    return fields


def query_parent(plan):
    proc = subprocess.run(["scontrol", "show", "job", PARENT, "-o"], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    return {"raw": proc.stdout, "fields": verify_parent(plan, proc.stdout), "epoch": time.time()}


def gpu_idle_record():
    device = Path("/sys/bus/pci/devices") / GPU["pci"]
    record = {"uuid": (device / "unique_id").read_text().strip().lower(), "pci": GPU["pci"],
              "used_vram_bytes": int((device / "mem_info_vram_used").read_text().strip()),
              "gpu_busy_percent": int((device / "gpu_busy_percent").read_text().strip()), "epoch": time.time()}
    require(record["uuid"] == GPU["uuid"], "Internal GPU UUID changed")
    return record


def check_idle(record, baseline=0):
    require(record["gpu_busy_percent"] == 0 and 0 <= record["used_vram_bytes"] - baseline < 128 * 1024**2,
            "Selected GPU no longer idle; do not share another worker's GPU")


def rss_snapshot():
    import psutil
    process = psutil.Process(os.getpid())
    own_ids = {process.pid, *(child.pid for child in process.children(recursive=True))}
    uid, own, total = os.getuid(), 0, 0
    for proc in psutil.process_iter():
        try:
            if proc.uids().real != uid:
                continue
            rss = proc.memory_info().rss
            total += rss
            if proc.pid in own_ids:
                own += rss
        except psutil.NoSuchProcess:
            pass
        # AccessDenied fails closed: do not silently undercount RAM.
    return {"own_tree_rss_bytes": own, "same_uid_rss_bytes": total,
            "other_same_uid_rss_bytes": max(0, total - own), "epoch": time.time()}


def guard_snapshot(snapshot, startup=False):
    reasons = []
    if snapshot["own_tree_rss_bytes"] > 32 * GIB:
        reasons.append("own_process_tree_exceeds32GiB")
    if snapshot["same_uid_rss_bytes"] > 60 * GIB:
        reasons.append("same_uid_node_RSS_exceeds60GiB")
    if startup and snapshot["other_same_uid_rss_bytes"] > 20 * GIB:
        reasons.append("startup_other_RSS_exceeds20GiB")
    if reasons:
        raise ResourceGuardError("resource_guard: " + ",".join(reasons))


def available_cpu_cores():
    import psutil
    usage = psutil.cpu_percent(interval=.5, percpu=True)
    allowed = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set(range(len(usage)))
    return sum(usage[index] < 25 for index in allowed if index < len(usage))


def lane_environment(plan, env):
    env = dict(env)
    require(env.get("SLURM_JOB_ID") == PARENT and env.get("SLURM_NTASKS") == "1"
            and env.get("SLURM_PROCID") == env.get("SLURM_LOCALID") == "0"
            and env.get("SLURM_STEP_ID", "").isdigit(), "A genuine one-rank child step is required")
    env["TABFM_SIDECAR_ORIGINAL_VISIBILITY"] = json.dumps({k: env.get(k) for k in
        ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")})
    env["ROCR_VISIBLE_DEVICES"] = "GPU-" + GPU["uuid"]
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update(EXPECTED_GPU_UUID=GPU["uuid"], EXPECTED_GPU_PCI_BUS_ID=GPU["pci"], PYTHONHASHSEED="0",
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4")
    return env


@contextmanager
def operational_watchdog(events):
    original = frozen.process_rss
    def checked_rss(_pid):
        snapshot = {}
        try:
            snapshot = rss_snapshot()
            guard_snapshot(snapshot)
        except Exception as exc:
            reason = "resource_guard: " + str(exc)
            events.append(dict(snapshot, reason=reason, inspection_exception=type(exc).__name__))
            raise ResourceGuardError(reason) from exc
        return snapshot["own_tree_rss_bytes"]
    frozen.process_rss = checked_rss
    try:
        yield
    finally:
        frozen.process_rss = original


def smoke_manifest(plan, man):
    # Only runtime output routing differs; children get the ORIGINAL campaign.
    return dict(man, output_root=str(sidecar_root(plan, man) / "smoke_run"))


def guarded_launch(plan, man, task, owner, events, smoke=False):
    guard_snapshot(rss_snapshot())
    with operational_watchdog(events):
        return frozen.launch(man, Path(plan["campaign_path"]), task, owner, smoke=smoke)


def lane(plan, man, tasks):
    env = lane_environment(plan, os.environ)
    os.environ.clear()
    os.environ.update(env)
    require(socket.gethostname().split(".")[0] == NODE, "Wrong physical node")
    os.nice(19)
    root = sidecar_root(plan, man)
    frozen.atomic(man, root / "lane_claim.json", {"plan_id": plan["plan_id"], "pid": os.getpid(), "epoch": time.time()})
    parent, memory = query_parent(plan), rss_snapshot()
    guard_snapshot(memory, startup=True)
    require(available_cpu_cores() >= 4, "Fewer than four currently idle allowed CPU cores")
    idle = gpu_idle_record()
    check_idle(idle)
    import torch
    from pfn_mitra_one import gpu_identity
    check_idle(gpu_idle_record())  # Recheck after import, before GPU initialization.
    actual = gpu_identity(torch)
    owner = {"rank": 0, "lane_id": 0, "job": PARENT, "step": os.environ["SLURM_STEP_ID"], "node": NODE,
             "uuid": actual["uuid"], "pci": actual["pci_bus_id"], "manifest_id": man["manifest_id"],
             "plan_id": plan["plan_id"], "single_lane_sidecar": True, "slurm_task_count": 1}
    baseline = gpu_idle_record()["used_vram_bytes"]
    frozen.atomic(man, root / "preflight.json", {"owner": owner, "actual_gpu": actual, "parent": parent,
        "startup_memory": memory, "startup_gpu_idle": idle, "post_probe_vram_baseline": baseline,
        "parent_gpu_reservation_count": 8, "step_inherited_gpu_access_count": 8, "actual_model_gpu_count": 1,
        "operational_guards": {"child_mem_gib": 40, "own_tree_gib": 32, "same_uid_gib": 60,
                               "startup_other_gib": 20, "nice": os.getpriority(os.PRIO_PROCESS, 0)}})
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, frozen.stop)
    signal.alarm(max(1, int(plan["deadline_epoch"] - time.time())))
    events, smoke_man = [], smoke_manifest(plan, man)
    selected = frozen.smoke_tasks(man, tasks)
    summary = {"plan_id": plan["plan_id"], "manifest_id": man["manifest_id"], "owner": owner,
               "attempted": 0, "succeeded": 0, "failed": 0, "reason": None}
    try:
        for task in selected:
            require(frozen.claim(smoke_man, task, owner, smoke=True), "Own smoke already claimed; no implicit retry")
            require(guarded_launch(plan, smoke_man, task, owner, events, smoke=True), "Sidecar smoke failed; bulk forbidden")
        for task in selected:
            frozen.valid_result(frozen.task_path(smoke_man, "smoke", task), man, task)
        frozen.atomic(man, root / "smoke_gate.json", {"plan_id": plan["plan_id"], "manifest_id": man["manifest_id"],
            "owner": owner, "tasks": [{k: t[k] for k in ("task_kind", "dataset_index", "dataset")} for t in selected],
            "full_original_test_splits": True, "sequential_single_gpu": True})
        for task in sorted(tasks, key=lambda t: (frozen.work_size(t), t["task_kind"], t["dataset_index"])):
            if plan["deadline_epoch"] - time.time() < 7320:
                summary["reason"] = "less_than_full7200s_task_budget_plus120s_margin_remains"
                break
            if events:
                summary["reason"] = "resource_guard_triggered_no_restart"
                break
            guard_snapshot(rss_snapshot())
            output = frozen.task_path(man, "results", task)
            if output.exists():
                if frozen.read(output).get("complete") is True:
                    frozen.valid_result(output, man, task)
                require(frozen.task_path(man, "claims", task).exists(), "Cannot adopt unclaimed output")
                continue
            if not frozen.claim(man, task, owner):
                continue
            ok = guarded_launch(plan, man, task, owner, events)
            summary["attempted"] += 1
            summary["succeeded" if ok else "failed"] += 1
        summary["reason"] = summary["reason"] or "no unclaimed work; not campaign completion"
    except BaseException as exc:
        summary.update(reason=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        summary.update(finished_epoch=time.time(), resource_guard_events=events)
        frozen.atomic(man, root / "lane_finished.json", summary)
    return summary


def step_command(plan, man, plan_path):
    minutes = int((plan["deadline_epoch"] - time.time() - 30) // 60)
    require(minutes >= 180, "More than three hours of parent lifetime required")
    # Inherit the parent's eight authorized devices so the selected noncontiguous
    # physical UUID is accessible. ROCr+HIP verification restricts actual use to1.
    return ["srun", "--jobid=" + PARENT, "--overlap", "--exact", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=4", "--mem=40G", "--gpus=8", "--gpu-bind=none", "--cpu-bind=none",
            "--kill-on-bad-exit=1", "--immediate=60", "--time=" + str(minutes), "--export=ALL",
            man["worker_python"], str(plan["sidecar_source"]["path"]), "--plan", str(Path(plan_path).resolve()), "--mode", "lane"]


def stop(sig, _frame):
    frozen.stop_child(ACTIVE_STEP)  # Own new srun group only, never parent allocation.
    raise SystemExit(128 + sig)


def driver(plan_path):
    global ACTIVE_STEP
    plan, man, _ = load_plan(plan_path, fresh=True)
    root, command = sidecar_root(plan, man), step_command(plan, man, plan_path)
    start = {"plan_id": plan["plan_id"], "manifest_id": man["manifest_id"], "plan": plan,
             "command": command, "parent": query_parent(plan), "started_epoch": time.time(), "pid": os.getpid()}
    frozen.atomic(man, root / "launch_receipt.json", start)
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL", "SLURM_GPUS_PER_TASK", "SLURM_TRES_PER_TASK"):
        env.pop(key, None)
    env.update(PYTHONHASHSEED="0", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    receipt = dict(start)
    try:
        with (root / "sidecar.log").open("x") as handle:
            ACTIVE_STEP = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            receipt["launcher_pid"] = ACTIVE_STEP.pid
            while ACTIVE_STEP.poll() is None:
                remaining = plan["deadline_epoch"] - time.time()
                require(remaining > 0, "Sidecar deadline reached")
                try:
                    ACTIVE_STEP.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    frozen.atomic(man, root / "driver_heartbeat.json", dict(receipt, epoch=time.time()), immutable=False)
            receipt["exit_code"] = ACTIVE_STEP.returncode
            require(ACTIVE_STEP.returncode == 0, "Sidecar child step failed; no automatic restart")
    except BaseException as exc:
        frozen.stop_child(ACTIVE_STEP)
        receipt.update(success=False, error=repr(exc), traceback=traceback.format_exc())
        raise
    else:
        receipt["success"] = True
    finally:
        ACTIVE_STEP = None
        receipt["finished_epoch"] = time.time()
        frozen.atomic(man, root / "driver_finished.json", receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mode", choices=("driver", "lane"), default="driver")
    args = parser.parse_args(argv)
    if args.mode == "driver":
        driver(args.plan.resolve())
    else:
        plan, man, tasks = load_plan(args.plan, fresh=True)
        print(json.dumps(lane(plan, man, tasks), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
