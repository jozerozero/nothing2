#!/usr/bin/env python3
"""Isolated native-TabFM queue: frozen681, four real GPUs, fresh bounded children.

Claims, results, preflight and attempts are immutable. A failed or interrupted
claim is never stolen or retried implicitly. Only this campaign root is writable.
This module imports only two read-only GPU helpers from the historical dispatcher.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

from classification32_dispatch import normalize_visibility, gpu_record


COUNTS = {"classification": 457, "regression": 224}
CHILD = None


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_file(record):
    path = Path(record["path"])
    require(path.is_absolute() and path.is_file(), f"Missing absolute frozen file: {path}")
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), f"File changed while hashing: {path}")
    require(h.hexdigest() == record["sha256"], f"Frozen SHA changed: {path}")
    for field, actual in (("size_bytes", after.st_size), ("mtime_ns", after.st_mtime_ns)):
        require(field not in record or record[field] == actual, f"Frozen {field} changed: {path}")
    return path.resolve()


def checked_path(man, path):
    root = Path(man["output_root"]).resolve()
    path = Path(path)
    require(not path.is_symlink() and path.resolve().is_relative_to(root)
            and path.resolve() != root, "Writes must stay strictly within isolated output_root")
    return path


def atomic(man, path, value, immutable=True):
    path = checked_path(man, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def load_campaign(path):
    man = read(path)
    require(man.get("manifest_id") == digest({k: v for k, v in man.items() if k != "manifest_id"}),
            "Campaign identity mismatch")
    root = Path(man["output_root"])
    require(root.is_absolute() and root.name not in ("", "outputs", "evaluation", "stage", "repo")
            and len(root.parts) >= 4 and root.resolve() != Path.home().resolve(), "Unsafe output root")
    require(Path(man["worker_python"]).is_absolute() and Path(man["worker_python"]).is_file(),
            "Missing frozen worker Python")
    verify_file(man["worker_script"])
    for record in man.get("worker_sources", []):
        verify_file(record)
    require(man["per_task_rss_limit_bytes"] == 48 * 1024**3
            and man["per_task_timeout_seconds"] == 7200, "Per-task budget differs from 48GiB/7200s")
    tasks = []
    for kind, count in COUNTS.items():
        data = read(verify_file(man[kind + "_manifest"]))
        require(data.get("manifest_id") == digest({k: v for k, v in data.items() if k != "manifest_id"}),
                f"{kind} input manifest digest mismatch")
        rows = data["rows"]
        indices = [r.get("dataset_index", r.get("position")) for r in rows]
        require(len(rows) == count and indices == list(range(count))
                and len({r["dataset"] for r in rows}) == count, f"Frozen {kind}{count} scope/order changed")
        for index, row in enumerate(rows):
            require(row.get("task_kind", kind) == kind, f"Incorrect task kind at {kind}:{index}")
            tasks.append({"task_kind": kind, "dataset_index": index, "dataset": row["dataset"],
                          "data_manifest_id": data["manifest_id"], "row": row})
    return man, tasks


def task_path(man, area, task):
    return Path(man["output_root"]) / area / task["task_kind"] / f"row-{task['dataset_index']:03d}.json"


def work_size(task):
    row = task["row"]
    if all(key in row for key in ("train_rows", "test_rows", "features")):
        return max(1, (row["train_rows"] + row["test_rows"]) * row["features"])
    require(isinstance(row.get("work_size"), (int, float)) and row["work_size"] > 0,
            f"Missing frozen work-size estimate: {task['dataset']}")
    return row["work_size"]


def smoke_tasks(man, tasks):
    by_key = {(t["task_kind"], t["dataset_index"]): t for t in tasks}
    if "smoke_tasks" in man:
        selected = [by_key[(t["task_kind"], t["dataset_index"])] for t in man["smoke_tasks"]]
    else:
        ordinary = [t for t in tasks if t["task_kind"] == "classification" and t["row"]["classes"] <= 10]
        hierarchy = [t for t in tasks if t["task_kind"] == "classification" and t["row"]["classes"] > 10]
        regression = sorted((t for t in tasks if t["task_kind"] == "regression"), key=work_size)
        require(ordinary and hierarchy and len(regression) >= 2, "Missing smoke protocol coverage")
        selected = [min(ordinary, key=work_size), min(hierarchy, key=work_size), *regression[:2]]
    require(len(selected) == 4 and len({(t["task_kind"], t["dataset_index"]) for t in selected}) == 4
            and selected[0]["task_kind"] == selected[1]["task_kind"] == "classification"
            and selected[0]["row"]["classes"] <= 10 < selected[1]["row"]["classes"]
            and all(t["task_kind"] == "regression" for t in selected[2:]), "Smoke must cover native class, hierarchy and two regressions")
    return selected


def preflight(man):
    record = dict(gpu_record(), manifest_id=man["manifest_id"])
    atomic(man, Path(man["output_root"]) / "preflight" / record["job"] / f"rank-{record['rank']}.json", record)
    return record


def check_preflight(man):
    job = os.environ["SLURM_JOB_ID"]
    records = [read(Path(man["output_root"]) / "preflight" / job / f"rank-{rank}.json") for rank in range(4)]
    require([r["rank"] for r in records] == list(range(4))
            and all(r["manifest_id"] == man["manifest_id"] and r["job"] == job for r in records)
            and len({r["node"] for r in records}) == len({r["step"] for r in records}) == 1
            and len({r["uuid"] for r in records}) == len({r["pci"] for r in records}) == 4,
            "Four ranks on four distinct physical GPUs in one node/step required")
    return records


def binding(man):
    expected = check_preflight(man)
    record = dict(gpu_record(), manifest_id=man["manifest_id"])
    reference = expected[record["rank"]]
    require(all(record[k] == reference[k] for k in ("node", "uuid", "pci", "job")),
            "Run rank binding differs from verified physical allocation")
    atomic(man, Path(man["output_root"]) / "bindings" / record["job"] / record["step"] /
           f"rank-{record['rank']}.json", record)
    return record


def valid_result(path, man, task):
    result = read(path)
    expected = {"complete": True, "status": "complete", "manifest_id": man["manifest_id"],
                "task_kind": task["task_kind"], "dataset_index": task["dataset_index"],
                "dataset": task["dataset"], "data_manifest_id": task["data_manifest_id"],
                "worker_source_sha256": man["worker_script"]["sha256"], "full_test_split": True}
    require(all(result.get(k) == value for k, value in expected.items()), f"Result contract failed: {path}")
    if "input_fingerprint" in task["row"]:
        require(result.get("input_fingerprint") == task["row"]["input_fingerprint"], "Result input identity mismatch")
    require(result.get("data_audit", {}).get("full_test_split") is True
            and isinstance(result.get("ensemble_audit"), dict)
            and isinstance(result.get("actual_ensemble_count"), int) and result["actual_ensemble_count"] > 0,
            "Missing actual native ensemble / full-test audit")
    metric_names = ("accuracy",) if task["task_kind"] == "classification" else ("rmse", "mae", "r2")
    metric = result.get("metrics", {})
    require(all(isinstance(metric.get(k), (int, float)) and not isinstance(metric[k], bool)
                and math.isfinite(metric[k]) for k in metric_names), "Missing/nonfinite full-test metrics")
    require(0 <= metric["accuracy"] <= 1 if task["task_kind"] == "classification"
            else metric["rmse"] >= 0 and metric["mae"] >= 0, "Invalid metric range")
    return result


def claim(man, task, owner, smoke=False):
    path = task_path(man, "smoke_claims" if smoke else "claims", task)
    record = dict(owner, manifest_id=man["manifest_id"], task_kind=task["task_kind"],
                  dataset_index=task["dataset_index"], dataset=task["dataset"], claimed_epoch=time.time())
    try:
        atomic(man, path, record)
    except FileExistsError:
        old = read(path)
        require(all(old.get(k) == record[k] for k in ("manifest_id", "task_kind", "dataset_index", "dataset")),
                f"Existing claim belongs to another protocol/task: {path}")
        return False
    return True


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=15)


def stop(sig, _frame):
    stop_child(CHILD)
    raise SystemExit(128 + sig)


def process_rss(pid):
    import psutil
    try:
        parent = psutil.Process(pid)
        processes = [parent] + parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    # AccessDenied deliberately propagates: an unmonitorable worker must stop.
    return total


def worker_environment(man, owner):
    env = dict(os.environ)
    env.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_HIP_ALLOC_CONF="expandable_segments:True", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    env.update(EXPECTED_GPU_UUID=owner["uuid"], EXPECTED_GPU_PCI_BUS_ID=owner["pci"])
    # Do not inherit a previous experiment's Python/module injection or GPU aliases.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        env.pop(key, None)
    temp_root = checked_path(man, Path(man["output_root"]) / "tmp" / str(owner["job"]) / str(owner["rank"]))
    temp_root.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(temp_root)
    return env


def launch(man, campaign_path, task, owner, smoke=False):
    global CHILD
    phase = "smoke" if smoke else "formal"
    root = Path(man["output_root"])
    output = task_path(man, "smoke" if smoke else "results", task)
    require(not output.exists() and not output.is_symlink(), "Never overwrite an existing worker receipt")
    identity = f"{phase}-{task['task_kind']}-{task['dataset_index']:03d}-{uuid.uuid4().hex}"
    candidate = checked_path(man, root / "worker_outputs" / (identity + ".json"))
    log = checked_path(man, root / "logs" / (identity + ".log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"manifest_id": man["manifest_id"], "phase": phase, "task_kind": task["task_kind"],
               "dataset_index": task["dataset_index"], "dataset": task["dataset"], "job": owner["job"],
               "rank": owner["rank"], "physical_gpu": owner, "log": str(log), "output": str(output),
               "worker_output": str(candidate),
               "started_epoch": time.time(), "peak_rss": 0, "exit_code": None, "reason": None}
    started = time.monotonic()
    fatal = None
    try:
        verify_file(man["worker_script"])
        cmd = [man["worker_python"], str(man["worker_script"]["path"]), "--campaign", str(campaign_path),
               "--task-kind", task["task_kind"], "--dataset-index", str(task["dataset_index"]),
               "--output", str(candidate), "--threads", "4"]
        receipt["command"] = cmd
        with log.open("x") as handle:
            CHILD = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                     env=worker_environment(man, owner), start_new_session=True)
            while CHILD.poll() is None:
                rss = process_rss(CHILD.pid)
                receipt["peak_rss"] = max(receipt["peak_rss"], rss)
                elapsed = time.monotonic() - started
                if rss > man["per_task_rss_limit_bytes"] or elapsed >= man["per_task_timeout_seconds"]:
                    receipt["reason"] = "rss_budget_exceeded" if rss > man["per_task_rss_limit_bytes"] else "task_timeout"
                    stop_child(CHILD)
                    break
                heartbeat = dict(receipt, pid=CHILD.pid, rss=rss, heartbeat_epoch=time.time())
                atomic(man, root / "workers" / str(owner["job"]) / f"rank-{owner['rank']}.json", heartbeat, immutable=False)
                try:
                    CHILD.wait(timeout=min(10, max(.01, man["per_task_timeout_seconds"] - elapsed)))
                except subprocess.TimeoutExpired:
                    pass
            receipt["exit_code"] = CHILD.returncode
        if receipt["exit_code"] != 0:
            receipt["reason"] = receipt["reason"] or "worker_nonzero_exit"
        elif receipt["reason"] is None:
            result = valid_result(candidate, man, task)
            require(result.get("physical_gpu", {}).get("uuid") == owner["uuid"]
                    and result.get("physical_gpu", {}).get("pci_bus_id") == owner["pci"],
                    "Worker result GPU differs from validated rank binding")
            atomic(man, output, result)
    except BaseException as exc:
        stop_child(CHILD)
        receipt["reason"] = receipt["reason"] or f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
        if CHILD is not None:
            receipt["exit_code"] = CHILD.returncode
        if not isinstance(exc, Exception):
            fatal = exc
    finally:
        CHILD = None
    receipt.update(finished_epoch=time.time(), success=receipt["reason"] is None)
    if not receipt["success"] and not output.exists():
        atomic(man, output, dict(receipt, complete=False, status="error"))
    atomic(man, root / ("attempts" if receipt["success"] else "errors") / (identity + ".json"), receipt)
    atomic(man, root / "workers" / str(owner["job"]) / f"rank-{owner['rank']}.json",
           dict(receipt, state="idle", heartbeat_epoch=time.time()), immutable=False)
    print(json.dumps(receipt), flush=True)
    if fatal is not None:
        raise fatal
    return receipt["success"]


def smoke(man, campaign_path, tasks):
    owner = binding(man)
    task = smoke_tasks(man, tasks)[owner["rank"]]
    require(claim(man, task, owner, smoke=True), "Smoke already claimed; do not retry implicitly")
    require(launch(man, campaign_path, task, owner, smoke=True), "Smoke worker failed; bulk run forbidden")


def check_smoke(man, tasks, publish=False):
    records = check_preflight(man)
    selected = smoke_tasks(man, tasks)
    for rank, task in enumerate(selected):
        owner = read(task_path(man, "smoke_claims", task))
        require(owner["job"] == records[rank]["job"] and owner["rank"] == rank
                and owner["manifest_id"] == man["manifest_id"], "Smoke must run in the current allocation")
        valid_result(task_path(man, "smoke", task), man, task)
    value = {"manifest_id": man["manifest_id"], "job": records[0]["job"],
             "tasks": [{k: task[k] for k in ("task_kind", "dataset_index", "dataset")} for task in selected]}
    gate = Path(man["output_root"]) / "gates" / f"{records[0]['job']}.smoke-passed.json"
    if publish:
        atomic(man, gate, value)
    else:
        require(read(gate) == value, "Missing/current smoke gate mismatch")
    return value


def dispatch(man, campaign_path, tasks):
    owner = binding(man)
    check_smoke(man, tasks)
    attempted = Counter()
    for task in sorted(tasks, key=lambda t: (work_size(t), t["task_kind"], t["dataset_index"])):
        output = task_path(man, "results", task)
        if output.exists():
            # Error receipts deliberately remain incomplete; claims prevent retries.
            if read(output).get("complete") is True:
                valid_result(output, man, task)
            require(task_path(man, "claims", task).exists(), "Unclaimed output cannot be adopted")
            continue
        if not claim(man, task, owner):
            continue
        success = launch(man, campaign_path, task, owner)
        attempted["success" if success else "failed"] += 1
    value = dict(owner, finished_epoch=time.time(), attempts=dict(attempted),
                 reason="no unclaimed work; this rank finishing is not campaign completion")
    atomic(man, Path(man["output_root"]) / "worker_done" / owner["job"] / f"rank-{owner['rank']}.json", value)
    return value


def status(man, tasks):
    counts = {kind: Counter(total=count) for kind, count in COUNTS.items()}
    invalid = []
    for task in tasks:
        count = counts[task["task_kind"]]
        output = task_path(man, "results", task)
        claim_path = task_path(man, "claims", task)
        if claim_path.exists():
            count["claimed"] += 1
        if not output.exists():
            count["claimed_without_result" if claim_path.exists() else "unclaimed"] += 1
            continue
        try:
            valid_result(output, man, task)
            count["complete"] += 1
        except Exception as exc:
            count["failed_or_invalid"] += 1
            invalid.append({"task_kind": task["task_kind"], "dataset_index": task["dataset_index"], "error": str(exc)})
    return {"manifest_id": man["manifest_id"], "complete": all(c["complete"] == c["total"] for c in counts.values()),
            "counts": {kind: dict(value) for kind, value in counts.items()}, "failed_or_invalid": invalid}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "check", "smoke", "check-smoke", "run", "status"))
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args(argv)
    man, tasks = load_campaign(args.campaign)
    if args.mode in ("preflight", "smoke", "run"):
        env = normalize_visibility(os.environ)
        env["TABFM_ORIGINAL_VISIBILITY"] = env.pop("CLASS32_ORIGINAL_VISIBILITY")
        os.environ.clear()
        os.environ.update(env)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.mode == "preflight":
        value = preflight(man)
    elif args.mode == "check":
        value = check_preflight(man)
    elif args.mode == "smoke":
        value = smoke(man, args.campaign.resolve(), tasks)
    elif args.mode == "check-smoke":
        value = check_smoke(man, tasks, publish=True)
    elif args.mode == "run":
        value = dispatch(man, args.campaign.resolve(), tasks)
    else:
        value = status(man, tasks)
    print(json.dumps(value, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
