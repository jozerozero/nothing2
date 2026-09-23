#!/usr/bin/env python3
"""One authorized existing physical GPU; FP32 TabSwift, no scheduler/video control.

Two protocols, eight current-lane full-data smokes before any formal claim.
Frozen scientific workers, CAS claims and immutable results are unchanged.
Unsuccessful attempts retain their claims; no retries, claim release or overwrite.
The external controller acquires the video's own cooperative GPU flock (best)
or separately pauses one authorized child with an independent restore watchdog.
This entry only verifies that lease and preserves the cooperative FD in children.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import time

import tabfm_default_dispatch as q
import tabswift_dispatch as swift
import shared_foundation_full_v2 as full
from table6_restart_deadline import EnvironmentBudget
from tabswift_one_v2 import protocol_settings

GIB = 1024**3
RESOURCE = {"gpus": 1, "cpus": 4, "step_mem_gib": 64, "own_rss_gib": 48,
            "parent_min_mem_gib": 1024, "parent_memory_reserve_gib": 128,
            "node_available_min_gib": 64, "gpu_free_min_gib": 16,
            "max_seconds": 7200, "shutdown_reserve_seconds": 60}
REQUIRED = ("tabswift_existing_fp32.py", "tabswift_one_v2.py", "tabswift_dispatch.py",
            "shared_foundation_full_v2.py", "shared_foundation_sidecar.py",
            "tabfm_default_dispatch.py", "tabfm_local_tmp.py", "table6_restart_deadline.py",
            "classification32_dispatch.py", "classification32_campaign.py",
            "classification32_submit.py", "pfn_mitra_one.py", "eval_one.py")
ACTIVE_GATE = None
LEASE_ROOT = Path("/vast/users/guangyi.chen/causal_group/jinyuan.hu/eec-bench/EECBench/experiments/holocine_generation_200_20260922/gpu_leases")


def identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    q.require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
              (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "Source changed during identity read")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns}


def campaign_contract(science, campaigns):
    q.require(science.get("existing_allocations_only") is True
              and science.get("new_allocation_submission_allowed") is False,
              "Only the isolated existing-allocation FP32 science plan is supported")
    q.require([m["protocol"]["variant"] for _, m, _ in campaigns] == ["official16", "budget32x8"],
              "Both original protocols required in fixed order")
    for _, man, tasks in campaigns:
        q.require(len(tasks) == 681 and sum(t["task_kind"] == "classification" for t in tasks) == 457,
                  "Original full classification457/regression224 scope required")
        q.require(Path(man["worker_script"]["path"]).name == "tabswift_one_v2.py", "FP32 worker required")
        for task in ("classification", "regression"):
            protocol_settings(man, task)
    q.require(campaigns[0][1]["classification_manifest"] == campaigns[1][1]["classification_manifest"]
              and campaigns[0][1]["regression_manifest"] == campaigns[1][1]["regression_manifest"],
              "Scientific split identities differ across protocols")


def prepare(science_path, run_id, parent, node, gpu_uuid, gpu_pci):
    q.require(str(parent) == "214135" and re.fullmatch(r"[A-Za-z0-9_-]+", run_id),
              "Only authorized existing parent214135 and safe unique run ID allowed")
    q.require(re.fullmatch(r"auh7-1b-gpu-\d+", node), "Invalid single physical node")
    q.require(re.fullmatch(r"[0-9a-f]{16}", gpu_uuid) and
              re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", gpu_pci), "Invalid GPU identity")
    science_path = Path(science_path).resolve(strict=True)
    science, campaigns = swift.load_plan(science_path)
    campaign_contract(science, campaigns)
    root = Path(campaigns[0][1]["output_root"])/"existing_gpu"/run_id
    sources = {}
    for record in [*science["source_records"], *science["campaign_manifests"], identity(science_path),
                   *(identity(Path(__file__).with_name(name)) for name in REQUIRED)]:
        path = q.verify_file(record)
        if str(path) in sources:
            q.require(sources[str(path)]["sha256"] == record["sha256"], "Conflicting source identities")
        else:
            sources[str(path)] = record
    plan = {"schema": "tabswift_existing_fp32_v1", "run_id": run_id, "parent_job_id": str(parent),
            "node": node, "gpu": {"uuid": gpu_uuid, "pci": gpu_pci}, "resource": RESOURCE,
            "runtime_root": str(root), "gate_path": str(root/"node-resource-gate.json"),
            "science_plan": identity(science_path), "science_plan_id": science["plan_id"],
            "runtime_script": identity(__file__), "worker_python": science["worker_python"],
            "source_records": [sources[p] for p in sorted(sources)],
            "existing_allocations_only": True, "new_allocation_submission_allowed": False,
            "video_actions_performed_by_entry": False, "full_rows_unchanged": True,
            "automatic_retry": False, "created_epoch": time.time()}
    plan["plan_id"] = q.digest(plan)
    q.atomic(campaigns[0][1], root/"plan.json", plan)
    load_plan(root/"plan.json")
    return {"plan_path": str(root/"plan.json"), "runtime_root": str(root),
            "gate_path": plan["gate_path"], "plan_id": plan["plan_id"], "submitted": False}


def load_plan(path):
    path = Path(path).resolve(strict=True)
    plan = q.read(path)
    q.require(plan.get("plan_id") == q.digest({k: v for k, v in plan.items() if k != "plan_id"}),
              "Runtime plan digest mismatch")
    q.require(plan["schema"] == "tabswift_existing_fp32_v1" and plan["resource"] == RESOURCE
              and plan["parent_job_id"] == "214135" and plan["existing_allocations_only"] is True
              and plan["new_allocation_submission_allowed"] is False, "Runtime authority/contract changed")
    verified = [q.verify_file(record) for record in plan["source_records"]]
    q.require(len(verified) == len(set(verified)) and
              {Path(__file__).with_name(name).resolve() for name in REQUIRED} <= set(verified),
              "Missing/duplicate runtime source pins")
    q.require(q.verify_file(plan["runtime_script"]) == Path(__file__).resolve(), "Wrong runtime entry")
    science_path = q.verify_file(plan["science_plan"])
    q.require(science_path in verified, "Science plan unpinned")
    science, campaigns = swift.load_plan(science_path)
    campaign_contract(science, campaigns)
    q.require(science["plan_id"] == plan["science_plan_id"] and
              all(man["worker_python"] == plan["worker_python"] for _, man, _ in campaigns), "Science/runtime changed")
    root = q.checked_path(campaigns[0][1], Path(plan["runtime_root"]))
    q.require(root == Path(campaigns[0][1]["output_root"])/"existing_gpu"/plan["run_id"]
              and path == root/"plan.json" and Path(plan["gate_path"]) == root/"node-resource-gate.json",
              "Runtime/gate path outside fixed isolated namespace")
    return plan, campaigns


def process_identity(pid):
    pid = int(pid)
    q.require(pid > 1, "Invalid process identity")
    root = Path("/proc")/str(pid)
    fields = (root/"stat").read_text().rsplit(") ", 1)[1].split()
    uid = root.stat().st_uid
    children = set()
    for task in (root/"task").iterdir():
        children.update(int(p) for p in (task/"children").read_text().split())
    return {"pid": pid, "uid": uid, "start_ticks": int(fields[19]), "state": fields[0],
            "children": sorted(children), "cmdline": (root/"cmdline").read_bytes().replace(b"\0", b" ").decode()}


def verify_lease(gate):
    if gate.get("release_method") == "cooperative_gpu_lease":
        return verify_cooperative_lease(gate)
    owners = gate["paused_video_owners"]
    q.require(len(owners) == 1 and gate["authorized_single_gpu_release"] is True
              and gate["no_other_gpu_owners"] is True, "Only one explicitly authorized paused video worker allowed")
    expected = owners[0]
    actual = process_identity(expected["pid"])
    q.require(expected["method"] == "SIGSTOP" and actual["state"] in ("T", "t")
              and not actual["children"] and actual["uid"] == expected["uid"] == os.getuid()
              and actual["start_ticks"] == expected["start_ticks"]
              and "node_manager" not in actual["cmdline"], "Paused video worker lease changed or is a manager")
    watchdog = gate["resume_watchdog"]
    observed = process_identity(watchdog["pid"])
    q.require(watchdog["restore_on_exit"] is True and observed["uid"] == watchdog["uid"] == os.getuid()
              and observed["start_ticks"] == watchdog["start_ticks"]
              and observed["state"] not in ("T", "t", "Z", "X")
              and observed["pid"] not in (actual["pid"], os.getpid()), "Restoration watchdog is not alive")
    return {"paused_worker": actual, "watchdog": observed}


def flock_holder_present(raw, *, pid, device, inode):
    for line in raw.splitlines():
        words = line.split()
        if len(words) < 6 or words[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
            continue
        major, minor, ino = words[5].split(":")
        if (words[4] == str(pid) and int(major, 16) == os.major(device)
                and int(minor, 16) == os.minor(device) and int(ino) == inode):
            return True
    return False


def verify_cooperative_lease(gate):
    q.require(gate["paused_video_owners"] == [] and gate["no_other_gpu_owners"] is True,
              "Cooperative lease must not pause video or hide other GPU owners")
    lease = gate["cooperative_lease"]
    expected_path = LEASE_ROOT/(gate["node"]+"_GPU-"+gate["gpu"]["uuid"]+".lock")
    q.require(Path(lease["path"]) == expected_path and not expected_path.is_symlink(), "Wrong cooperative video lease path")
    descriptor = int(os.environ.get("TABSWIFT_GPU_LEASE_FD", "-1"))
    q.require(descriptor >= 3 and descriptor == lease["fd"], "Missing inherited cooperative lease descriptor")
    held = os.fstat(descriptor)
    actual = process_identity(lease["holder_pid"])
    holder_fd = (Path("/proc")/str(lease["holder_pid"])/"fd"/str(lease["fd"])).stat()
    inode = expected_path.stat()
    q.require(actual["uid"] == lease["uid"] == os.getuid()
              and actual["start_ticks"] == lease["start_ticks"] and actual["state"] not in ("Z", "X")
              and (held.st_dev, held.st_ino) == (holder_fd.st_dev, holder_fd.st_ino)
                  == (inode.st_dev, inode.st_ino) == (lease["device"], lease["inode"]),
              "Cooperative holder/inherited inode changed")
    q.require(flock_holder_present(Path("/proc/locks").read_text(), pid=lease["holder_pid"],
                                  device=held.st_dev, inode=held.st_ino), "Kernel lacks expected exclusive cooperative flock")
    other_fd = os.open(expected_path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        try:
            fcntl.flock(other_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(other_fd, fcntl.LOCK_UN)
            raise RuntimeError("Independent descriptor acquired supposedly held cooperative flock")
    finally:
        os.close(other_fd)
    return {"cooperative_lease_verified": True, "holder": actual, "descriptor": descriptor,
            "device": held.st_dev, "inode": held.st_ino}


def read_gate(plan):
    gate_path = Path(plan["gate_path"])
    gate = q.read(gate_path)
    q.require(gate.get("schema") == "tabswift_existing_gpu_gate_v1"
              and gate["plan_id"] == plan["plan_id"] and gate["job"] == plan["parent_job_id"]
              and gate["node"] == plan["node"] and gate["step"] == os.environ["SLURM_STEP_ID"]
              and gate["uid"] == os.getuid() and gate["gpu"] == plan["gpu"], "Controller gate identity mismatch")
    elapsed = time.monotonic()-gate["created_monotonic"]
    q.require(0 <= elapsed <= 60, "Controller gate not fresh on this node")
    cpus = gate["cpu_ids"]
    q.require(len(cpus) == len(set(cpus)) == 4 and set(cpus) == set(os.sched_getaffinity(0)),
              "Actual CPU affinity differs from four-core controller lease")
    verify_lease(gate)
    return gate, identity(gate_path)


def memory_snapshot():
    import psutil
    if ACTIVE_GATE is not None:
        verify_lease(ACTIVE_GATE)
    mine = psutil.Process(os.getpid())
    own = sum(p.memory_info().rss for p in [mine, *mine.children(recursive=True)] if p.is_running())
    job = os.environ["SLURM_JOB_ID"]
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    paths = [line.split("::", 1)[1] for line in lines if line.startswith("0::") and "/job_"+job+"/" in line]
    q.require(len(paths) == 1, "Cannot identify unique Slurm job cgroup")
    components = Path(paths[0]).parts
    q.require(components.count("job_"+job) == 1, "Ambiguous allocation cgroup")
    parent = Path("/sys/fs/cgroup")/Path(*components[1:components.index("job_"+job)+1])
    maximum = (parent/"memory.max").read_text().strip()
    q.require(maximum.isdigit(), "Unbounded/unknown parent memory limit")
    return {"own_tree_rss_bytes": own, "parent_memory_current": int((parent/"memory.current").read_text()),
            "parent_memory_max": int(maximum), "node_available_bytes": psutil.virtual_memory().available,
            "epoch": time.time()}


def resource_guard(memory, budget, startup=False):
    if full.base.STOP is not None or budget.remaining() <= RESOURCE["shutdown_reserve_seconds"]:
        raise full.base.OperationalDeferral("signal_or_bounded_step_deadline")
    for key in ("own_tree_rss_bytes", "parent_memory_current", "parent_memory_max", "node_available_bytes"):
        q.require(type(memory[key]) is int and memory[key] >= 0, "Invalid memory observation")
    q.require(memory["parent_memory_max"] >= RESOURCE["parent_min_mem_gib"]*GIB,
              "This entry requires the large-memory existing parent")
    if memory["own_tree_rss_bytes"] > RESOURCE["own_rss_gib"]*GIB:
        raise full.base.OperationalDeferral("own_tree_RSS_exceeds48GiB")
    if memory["parent_memory_max"]-memory["parent_memory_current"] < RESOURCE["parent_memory_reserve_gib"]*GIB:
        raise full.base.OperationalDeferral("parent_cgroup_raw_headroom_below128GiB")
    if memory["node_available_bytes"] < RESOURCE["node_available_min_gib"]*GIB:
        raise full.base.OperationalDeferral("node_available_memory_below64GiB")


def checked_fp32_result(path, man, task):
    result = swift.validated_result(path, man, task)
    precision = result.get("precision", {})
    q.require(result.get("protocol") == man["protocol"] and precision.get("native_use_amp") is False
              and precision.get("requested_precision") == "fp32"
              and precision.get("parameter_dtypes") == ["torch.float32"]
              and precision.get("autocast_enabled_observations") == 0
              and precision.get("forward_hook_calls", 0) > 0
              and precision.get("checked_float_tensor_inputs", 0) > 0
              and precision.get("checked_float_tensor_outputs", 0) > 0
              and result.get("constructor_settings", {}).get("use_amp") is False,
              "Result lacks actual FP32/no-AMP forward evidence")
    if task["task_kind"] == "classification":
        proba = result["ensemble_audit"].get("probability_validation", {})
        q.require(proba.get("native_dtype") == "float32" and proba.get("absolute_row_sum_tolerance") == 2e-5
                  and proba.get("probabilities_renormalized") is False
                  and 0 <= proba.get("maximum_absolute_row_sum_error", float("inf")) <= 2e-5,
                  "Missing strict unmodified FP32 probabilities")
    return result


@contextmanager
def runtime_guards():
    previous = full.base.snapshot, full.base.guard, q.valid_result, q.subprocess
    original_subprocess = q.subprocess

    class InheritedLeaseSubprocess:
        def __getattr__(self, name):
            return getattr(original_subprocess, name)

        def Popen(self, *args, **kwargs):
            if ACTIVE_GATE is not None and ACTIVE_GATE.get("release_method") == "cooperative_gpu_lease":
                verify_cooperative_lease(ACTIVE_GATE)
                descriptor = int(os.environ["TABSWIFT_GPU_LEASE_FD"])
                kwargs["pass_fds"] = tuple(sorted({*kwargs.get("pass_fds", ()), descriptor}))
            return original_subprocess.Popen(*args, **kwargs)

    full.base.snapshot, full.base.guard, q.valid_result = memory_snapshot, resource_guard, checked_fp32_result
    q.subprocess = InheritedLeaseSubprocess()
    try:
        yield
    finally:
        full.base.snapshot, full.base.guard, q.valid_result, q.subprocess = previous


@contextmanager
def gpu_lock(plan, first_man):
    path = q.checked_path(first_man, Path(first_man["output_root"])/"existing_gpu_locks"/
                          (plan["parent_job_id"]+"-"+plan["gpu"]["uuid"]+".lock"))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)  # Do not unlink the inode: future lockers must share it.


def gpu_before_context(plan):
    root = Path("/sys/bus/pci/devices")/plan["gpu"]["pci"]
    gpu = {"uuid": (root/"unique_id").read_text().strip().lower(),
           "busy": int((root/"gpu_busy_percent").read_text()),
           "vram_used": int((root/"mem_info_vram_used").read_text()),
           "vram_total": int((root/"mem_info_vram_total").read_text())}
    q.require(gpu["uuid"] == plan["gpu"]["uuid"] and gpu["busy"] == 0
              and 0 <= gpu["vram_used"] <= gpu["vram_total"]
              and gpu["vram_total"]-gpu["vram_used"] >= RESOURCE["gpu_free_min_gib"]*GIB,
              "Released GPU identity/idle/free-memory changed")
    if ACTIVE_GATE is not None and ACTIVE_GATE.get("release_method") == "cooperative_gpu_lease":
        q.require(gpu["vram_used"] < 128*1024**2, "Cooperatively leased GPU still has another live GPU context")
    return gpu


def run_queues(plan, campaigns, owner, budget):
    summary = {"plan_id": plan["plan_id"], "owner": owner, "smokes": [], "attempts": [], "state": "starting"}
    roots = [Path(man["output_root"])/"existing_gpu"/plan["run_id"] for _, man, _ in campaigns]
    try:
        with runtime_guards():
            for (path, man, tasks), root in zip(campaigns, roots):
                for task in q.smoke_tasks(man, tasks):
                    value = full.attempt(plan, path, man, task, owner, budget, root, smoke=True)
                    summary["smokes"].append(value)
                    q.require(value["state"] == "complete", "Current GPU smoke failed; no formal dispatch")
                    checked_fp32_result(value["attempt_output"], man, task)
            q.require(len(summary["smokes"]) == 8, "Both protocols need all four current-GPU smokes")
            q.atomic(campaigns[0][1], Path(plan["runtime_root"])/"smoke-gate.json",
                     {"plan_id": plan["plan_id"], "owner": owner, "passed": True, "smokes": summary["smokes"]})
            for variant, task in swift.pending_order(campaigns):
                if budget.remaining() <= 120 or full.base.STOP is not None:
                    raise full.base.OperationalDeferral("no_new_task_window_or_signal")
                path, man, _ = campaigns[variant]
                output = q.task_path(man, "results", task)
                if output.exists():
                    q.require(q.task_path(man, "claims", task).exists(), "Existing result lacks canonical claim")
                    if q.read(output).get("complete") is True:
                        checked_fp32_result(output, man, task)
                    continue
                value = full.attempt(plan, path, man, task, owner, budget, roots[variant])
                if value["state"] != "already_claimed":
                    summary["attempts"].append(value)
                if value["state"] in ("retained_resource_deferral", "operationally_deferred"):
                    raise full.base.OperationalDeferral(value["state"]+"; claim retained, no automatic retry")
            summary["state"] = "no_unclaimed_work_not_campaign_completion"
    except full.base.OperationalDeferral as exc:
        summary.update(state="operationally_deferred", reason=str(exc))
    except BaseException as exc:
        summary.update(state="failed", reason=repr(exc))
        raise
    finally:
        summary["cleanup"] = full.base.cleanup_children()
        summary["finished_epoch"] = time.time()
        for root, (_, man, _) in zip(roots, campaigns):
            q.atomic(man, root/"finished.json", summary)
    return summary


def run(plan, campaigns):
    global ACTIVE_GATE
    q.require(os.environ.get("SLURM_JOB_ID") == plan["parent_job_id"]
              and os.environ.get("SLURM_NTASKS") == "1"
              and os.environ.get("SLURM_PROCID") == os.environ.get("SLURM_LOCALID") == "0"
              and os.environ.get("SLURM_STEP_ID", "").isdigit()
              and socket.gethostname().split(".")[0] == plan["node"], "One-rank step on the planned parent/node required")
    budget = EnvironmentBudget.from_environment()
    q.require(budget.monotonic_end is not None and 120 < budget.remaining() <= 6901,
              "Need same-node controller budget, at most2h minus300s")
    gate, gate_identity = read_gate(plan)
    ACTIVE_GATE = gate
    full.base.STOP = None
    memory = memory_snapshot(); resource_guard(memory, budget, startup=True)
    with gpu_lock(plan, campaigns[0][1]):
        root = Path(plan["runtime_root"])
        q.atomic(campaigns[0][1], root/"lane-claim.json", {"plan_id": plan["plan_id"],
                 "job": plan["parent_job_id"], "step": os.environ["SLURM_STEP_ID"], "pid": os.getpid(), "epoch": time.time()})
        idle = gpu_before_context(plan)
        os.environ["ROCR_VISIBLE_DEVICES"] = "GPU-"+plan["gpu"]["uuid"]
        for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "PYTHONPATH", "PYTHONHOME"):
            os.environ.pop(key, None)
        os.environ.update(EXPECTED_GPU_UUID=plan["gpu"]["uuid"], EXPECTED_GPU_PCI_BUS_ID=plan["gpu"]["pci"],
            PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4")
        os.nice(19)
        from tabfm_local_tmp import activate
        tmp = activate(campaigns[0][1], plan, root/"runtime_environment")
        import torch
        from pfn_mitra_one import gpu_identity
        gpu = gpu_identity(torch)
        q.require(gpu["uuid"] == plan["gpu"]["uuid"] and gpu["pci_bus_id"] == plan["gpu"]["pci"],
                  "Actual physical GPU differs from released GPU")
        torch.set_num_threads(4)
        probe = torch.ones((4, 4), device="cuda", dtype=torch.float32)
        q.require(probe.dtype == torch.float32 and float(probe.sum().cpu()) == 16, "Native GPU FP32 probe failed")
        del probe
        owner = {"job": plan["parent_job_id"], "node": plan["node"], "step": os.environ["SLURM_STEP_ID"],
                 "rank": 0, "uuid": gpu["uuid"], "pci": gpu["pci_bus_id"], "plan_id": plan["plan_id"],
                 "actual_cpu_ids": gate["cpu_ids"], "single_real_gpu": True}
        q.atomic(campaigns[0][1], root/"preflight.json", {"plan_id": plan["plan_id"], "owner": owner,
                 "physical_gpu": gpu, "gpu_before_context": idle, "memory": memory, "resource": RESOURCE,
                 "controller_gate": gate_identity, "TMPDIR": tmp, "budget_monotonic_end": budget.monotonic_end,
                 "fp32_tensor_probe_passed": True, "video_actions_performed_by_entry": False})
        full.base.subreaper()
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(number, full.base.signal_stop)
        return run_queues(plan, campaigns, owner, budget)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify", "run"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--science-plan", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--parent")
    parser.add_argument("--node")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--gpu-pci")
    args = parser.parse_args(argv)
    q.require(__debug__, "Optimized Python unsupported")
    if args.mode == "prepare":
        q.require(all((args.science_plan, args.run_id, args.parent, args.node, args.gpu_uuid, args.gpu_pci)),
                  "Missing immutable prepare binding arguments")
        result = prepare(args.science_plan, args.run_id, args.parent, args.node, args.gpu_uuid, args.gpu_pci)
    else:
        q.require(args.plan is not None, "--plan required")
        plan, campaigns = load_plan(args.plan)
        result = {"valid": True, "plan_id": plan["plan_id"]} if args.mode == "verify" else run(plan, campaigns)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
