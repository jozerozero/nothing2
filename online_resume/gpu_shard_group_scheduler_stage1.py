#!/usr/bin/env python3
"""Coordinate one member of a four-GPU online checkpoint-evaluation group."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
import time
from pathlib import Path


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def strict_panel(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        names = [row["dataset"] for row in rows]
        values = [float(row["accuracy"]) for row in rows]
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return len(rows) == 178 and len(set(names)) == 178 and all(math.isfinite(v) and 0 <= v <= 1 for v in values)


def checkpoint_is_stable(path: Path, stable_sec: float) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    return stat.st_size > 100_000_000 and time.time() - stat.st_mtime >= stable_sec


def checkpoint_for_step(args: argparse.Namespace, step: int) -> tuple[Path, int]:
    if step < 8850 or step > 25000 or step % 50:
        raise ValueError(f"outside authorized online range: {step}")
    root, training = (args.checkpoint_root, 178786) if step <= 14500 else (args.resume_checkpoint_root, 181407)
    return root / f"step-{step}.ckpt", training


def completed_step(args: argparse.Namespace, step: int) -> bool:
    panel = args.output_root / f"step-{step}" / "talent_detailed.txt"
    receipt_path = panel.parent / "gpu_shard_receipt.json"
    summary = panel.parent / "talent_summary.txt"
    # The merger atomically writes the panel BEFORE its 12-second stability
    # check, receipt and summary. A peer must not mistake that window for done.
    try:
        if min(time.time() - p.stat().st_mtime for p in (panel, receipt_path, summary)) < 12:
            return False
        receipt = json.loads(receipt_path.read_text())
    except FileNotFoundError:
        return False
    if not strict_panel(panel):
        return False
    claim_path = args.claims_root / f"step-{step}.json"
    claim = json.loads(claim_path.read_text())
    checkpoint, training = checkpoint_for_step(args, step)
    assert claim["checkpoint"] == str(checkpoint.resolve())
    assert claim["training_job"] == training and claim["loop_passes"] == 3
    producer = str(claim["job_id"])
    if producer != os.environ["SLURM_JOB_ID"]:
        assert producer == '181580' and step in args.retained_steps, (step, producer)
    assert receipt["dataset_count"] == receipt["unique_dataset_count"] == 178
    assert receipt["explicit_fp32"] is True and receipt["clf_use_amp"] is False and receipt["clf_use_fa3"] is False
    assert receipt["shard_count"] == 4 and receipt["model_tag"] == f"step-{step}"
    assert receipt["stable_scan_sec"] >= 12
    assert receipt['n_estimators'] == 32 and receipt['outer_batch'] == 8
    assert receipt['n_jobs'] == 1 and receipt['kv_cache'] is False
    if hasattr(args, 'expected_dataset_names'):
        with panel.open(newline='') as handle:
            assert {r['dataset'] for r in csv.DictReader(handle, delimiter='\t')} == args.expected_dataset_names
    return True


def claim_checkpoint(args: argparse.Namespace, group_index: int, epoch: int) -> dict[str, object] | None:
    args.claims_root.mkdir(parents=True, exist_ok=True)
    args.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        strict_count = 0
        for step in args.steps:
            if completed_step(args, step):
                strict_count += 1
        if strict_count == len(args.steps):
            return {"done": True, "strict_count": strict_count, "epoch": epoch}

        for step in args.steps:
            if completed_step(args, step):
                continue
            checkpoint, training = checkpoint_for_step(args, step)
            if not checkpoint_is_stable(checkpoint, args.checkpoint_stable_sec):
                continue
            claim_path = args.claims_root / f"step-{step}.json"
            if claim_path.exists():
                continue
            payload: dict[str, object] = {
                "done": False,
                "step": step,
                "checkpoint": str(checkpoint.resolve()),
                "training_job": training,
                "loop_passes": 3,
                "group_index": group_index,
                "epoch": epoch,
                "claimed_at_unix": time.time(),
                "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
            }
            atomic_json(claim_path, payload)
            return payload
    return None


def wait_json(path: Path, poll_sec: float) -> dict[str, object]:
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(poll_sec)


def run_checked(command: list[str]) -> None:
    print("command=" + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--group-count", type=int, default=4)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--claims-root", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--gpu-shard-worker", type=Path, required=True)
    parser.add_argument("--merge-script", type=Path, required=True)
    parser.add_argument("--shard-policy", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--inner-batch-wrapper", type=Path, required=True)
    parser.add_argument("--inner-batch-policy", type=Path, required=True)
    parser.add_argument("--checkpoint-stable-sec", type=float, default=60.0)
    parser.add_argument("--poll-sec", type=float, default=6.0)
    parser.add_argument("--cpu-threads", type=int, default=12)
    args = parser.parse_args()

    if not 0 <= args.task_index < args.group_count * 4:
        raise SystemExit("task index outside configured four-GPU groups")
    args.steps = list(range(8850, 25001, 50))
    from recovery import runtime_contract, install_reusable_shard
    registration = runtime_contract()
    args.retained_steps = registration['retained_steps']
    policy = json.loads(args.shard_policy.read_text())
    args.expected_dataset_names = {name for shard in policy['shards'] for name in shard}
    assert len(args.expected_dataset_names) == 178
    assert args.checkpoint_root == Path("/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/checkpoints/e4_g5_support_condition_alpha_loops_20260907_v1/g36-g5scalpha-loop3-histe4-25k-v1/e4g5sc3lr1-178786")
    assert args.resume_checkpoint_root == Path("/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/checkpoints/e4_g5_support_condition_alpha_loops_20260907_v1/g36-g5scalpha-loop3-histe4-25k-v1/e4g5sc3lr2-181407")
    assert args.output_root == Path("/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/evaluation/e4_g5sc_loop3_resume181407_fp32_online_24gpu_gt_step50_20260910_v1/E4_G5SC_LOOP3/lineage-178786-181407")
    assert args.group_count == 4
    assert args.claims_root == args.output_root / ".claims-v1"
    group_index = args.task_index // 4
    shard_index = args.task_index % 4
    group_root = args.job_root / f"group-{group_index}"
    task_root = args.job_root / f"task-{args.task_index}"
    group_root.mkdir(parents=True, exist_ok=True)
    task_root.mkdir(parents=True, exist_ok=True)
    epoch = 0

    while True:
        assignment_path = group_root / f"assignment-{epoch:03d}.json"
        completion_path = group_root / f"complete-{epoch:03d}.json"
        if shard_index == 0:
            assignment = None
            while assignment is None:
                assignment = claim_checkpoint(args, group_index, epoch)
                if assignment is None:
                    print(
                        f"group={group_index} epoch={epoch} no stable unclaimed checkpoint; waiting",
                        flush=True,
                    )
                    time.sleep(args.poll_sec)
            atomic_json(assignment_path, assignment)
        assignment = wait_json(assignment_path, args.poll_sec)
        if assignment.get("done") is True:
            break

        step = int(assignment["step"])
        checkpoint = Path(str(assignment["checkpoint"]))
        step_root = args.job_root / "work" / f"step-{step}"
        shard_root = step_root / f"shard-{shard_index}"
        install_reusable_shard(registration, step, shard_index, shard_root, checkpoint, policy)
        shard_result = shard_root / "shard_result.json"
        if not shard_result.is_file():
            run_checked(
                [
                    "python",
                    str(args.gpu_shard_worker),
                    "--checkpoint",
                    str(checkpoint),
                    "--data-root",
                    str(args.data_root),
                    "--cache-root",
                    str(args.cache_root),
                    "--evaluator-dir",
                    str(args.evaluator_dir),
                    "--inner-batch-wrapper",
                    str(args.inner_batch_wrapper),
                    "--inner-batch-policy",
                    str(args.inner_batch_policy),
                    "--shard-policy",
                    str(args.shard_policy),
                    "--shard-index",
                    str(shard_index),
                    "--output-root",
                    str(shard_root),
                    "--cpu-threads",
                    str(args.cpu_threads),
                ]
            )

        if shard_index == 0:
            shard_results = [step_root / f"shard-{index}" / "shard_result.json" for index in range(4)]
            while not all(path.is_file() for path in shard_results):
                time.sleep(args.poll_sec)
            run_checked(
                [
                    "python3",
                    str(args.merge_script),
                    "--step-root",
                    str(step_root),
                    "--policy",
                    str(args.shard_policy),
                    "--model-tag",
                    f"step-{step}",
                    "--output-root",
                    str(args.output_root),
                    "--stable-scan-sec",
                    "12",
                ]
            )
            atomic_json(
                completion_path,
                {
                    "step": step,
                    "group_index": group_index,
                    "epoch": epoch,
                    "strict_exact178": True,
                    "explicit_fp32": True,
                    "completed_at_unix": time.time(),
                },
            )
        wait_json(completion_path, args.poll_sec)
        epoch += 1

    atomic_text(
        task_root / "task.complete",
        f"task={args.task_index} group={group_index} shard={shard_index} complete=1\n",
    )


if __name__ == "__main__":
    main()
