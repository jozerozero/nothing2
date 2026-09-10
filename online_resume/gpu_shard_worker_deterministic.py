#!/usr/bin/env python3
"""Evaluate one Exact178 shard with shape-keyed deterministic batch plans."""

from __future__ import annotations

import argparse
import json
import logging

BATCH_SAFETY_FACTOR = 0.20
import os
import sys
import time
from pathlib import Path


POLICIES = {
    "56000:14000:11": {
        "tf_col": (133, 108, 108, 108),
        "tf_row": (50000, 50000, 50000, 50000),
        "tf_icl": (32, 32, 32, 32),
    },
    "7315:1829:220": {
        "tf_col": (956, 1064, 1068, 1073),
        "tf_row": (1826, 4403, 10840, 10652),
        "tf_icl": (230, 209, 181, 182),
    },
    "78401:19652:20": {
        "tf_col": (61, 66, 71, 71),
        "tf_row": (50000, 50000, 50000, 50000),
        "tf_icl": (17, 17, 17, 17),
    },
    "119465:29867:4": {
        "tf_col": (42, 50, 50, 50),
        "tf_row": (37500, 37500, 37500, 37500),
    },
}


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def install_low_row_batch_retry(torch_module) -> None:
    from tabicl._model.inference import InferenceManager

    current_estimate = InferenceManager.estimate_safe_batch_size

    def estimate(self, seq_len, include_inputs=True, in_dim=None, max_bs=50000):
        gpu_mb, estimated = current_estimate(self, seq_len, include_inputs, in_dim, max_bs)
        if str(getattr(self, "enc_name", "")) != "tf_row" or int(estimated) >= 512:
            return gpu_mb, estimated
        torch_module.cuda.empty_cache()
        retry_gpu_mb, retry_estimated = current_estimate(
            self, seq_len, include_inputs, in_dim, max_bs
        )
        if int(retry_estimated) > int(estimated):
            logging.warning(
                "inner_batch_low_row_retry initial=%d retry=%d initial_gpu_mb=%.1f "
                "retry_gpu_mb=%.1f",
                int(estimated), int(retry_estimated), float(gpu_mb), float(retry_gpu_mb),
            )
            return retry_gpu_mb, retry_estimated
        return gpu_mb, estimated

    InferenceManager.estimate_safe_batch_size = estimate


def install_deterministic_shape_batches(inner_batch_module) -> None:
    """Replace four known-sensitive shapes with checkpoint-independent plans."""
    from tabicl._model.inference import InferenceManager

    adaptive_estimate = InferenceManager.estimate_safe_batch_size
    state: dict[str, object] = {"shape": None, "calls": {}}

    def estimate(self, seq_len, include_inputs=True, in_dim=None, max_bs=50000):
        gpu_mb, adaptive = adaptive_estimate(self, seq_len, include_inputs, in_dim, max_bs)
        shape = str(getattr(inner_batch_module, "_CURRENT_SHAPE", "unknown"))
        policy = POLICIES.get(shape)
        if policy is None:
            state["shape"] = None
            state["calls"] = {}
            minimum = int(getattr(self, "min_batch_size", 1))
            conservative = max(minimum, min(int(max_bs), int(int(adaptive) * BATCH_SAFETY_FACTOR)))
            logging.info(
                "rep6_v4_conservative_inner_batch shape=%s encoder=%s adaptive=%d effective=%d safety=%.2f",
                shape, str(getattr(self, "enc_name", "unknown")), int(adaptive), conservative,
                BATCH_SAFETY_FACTOR,
            )
            return gpu_mb, conservative

        if state["shape"] != shape:
            state["shape"] = shape
            state["calls"] = {}

        encoder = str(getattr(self, "enc_name", "unknown"))
        if encoder not in policy:
            raise RuntimeError(f"unexpected encoder={encoder} for deterministic shape={shape}")
        calls = state["calls"]
        assert isinstance(calls, dict)
        index = int(calls.get(encoder, 0))
        sequence = policy[encoder]
        calls[encoder] = index + 1
        sequence_index = min(index, len(sequence) - 1)
        overflow_clamped = index >= len(sequence)
        minimum = int(getattr(self, "min_batch_size", 1))
        effective = max(
            minimum,
            min(int(max_bs), int(int(sequence[sequence_index]) * BATCH_SAFETY_FACTOR)),
        )

        logging.info(
            "deterministic_inner_batch shape=%s encoder=%s call=%d "
            "adaptive=%d effective=%d gpu_mb=%.1f overflow_clamped=%s",
            shape, encoder, index + 1, int(adaptive), int(effective), float(gpu_mb), overflow_clamped,
        )
        return gpu_mb, effective

    InferenceManager.estimate_safe_batch_size = estimate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--inner-batch-wrapper", type=Path, required=True)
    parser.add_argument("--inner-batch-policy", type=Path, required=True)
    parser.add_argument("--shard-policy", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=12)
    args = parser.parse_args()

    os.environ["TABICL_EVAL_DISABLE_LOCAL_SRC"] = "1"
    sys.path.insert(0, str(args.evaluator_dir.resolve()))
    sys.path.insert(0, str(args.inner_batch_wrapper.resolve().parent))
    import torch
    import talent_eval_ckpt_per_gpu as production
    import talent_eval_ckpt_per_gpu_inner_batch as inner_batch

    policy = json.loads(args.shard_policy.read_text(encoding="utf-8"))
    shards = policy["shards"]
    if policy.get("dataset_count") != 178 or len(shards) != policy.get("shard_count"):
        raise SystemExit("invalid Exact178 shard policy")
    if not 0 <= args.shard_index < len(shards):
        raise SystemExit("shard index out of range")
    flattened = [str(name) for shard in shards for name in shard]
    if len(flattened) != 178 or len(set(flattened)) != 178:
        raise SystemExit("shard policy does not exactly partition 178 datasets")
    names = [str(name) for name in shards[args.shard_index]]
    data_dirs = [args.data_root.resolve() / name for name in names]
    missing = [str(path) for path in data_dirs if not path.is_dir()]
    if missing:
        raise SystemExit(f"missing dataset directories: {missing}")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    production._set_cpu_thread_limits(args.cpu_threads)
    torch.cuda.set_device(0)
    torch.set_num_threads(max(1, args.cpu_threads))
    inner_batch._install_inner_batch_patch()
    inner_batch._install_shape_tracking(args.inner_batch_policy.resolve(), 1.0)
    install_deterministic_shape_batches(inner_batch)
    install_low_row_batch_retry(torch)

    entries = production.preload_dataset_entries(data_dirs, args.cache_root.resolve())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model_tag, total, avg_acc, total_t, avg_t, avg_tr = production.evaluate_checkpoint_serial(
        ckpt=args.checkpoint.resolve(),
        outdir_root=output_root,
        dataset_entries=entries,
        cache_root=args.cache_root.resolve(),
        gpu_device="0",
        clf_n_estimators=32,
        clf_batch_size=8,
        clf_n_jobs=1,
        clf_norm_methods=None,
        clf_checkpoint_version=None,
        allow_auto_download=False,
        kv_cache=False,
        clf_use_amp=False,
        clf_use_fa3=False,
        use_pseudo_ssmax_thinking=False,
        pseudo_thinking_fraction=0.25,
        pseudo_thinking_conf_threshold=0.0,
        pseudo_thinking_max_passes=2,
        pseudo_thinking_consistency_threshold=0.995,
        use_torch_compile=False,
        torch_compile_mode=None,
        torch_compile_backend=None,
        torch_compile_fullgraph=False,
        torch_compile_dynamic=None,
        stderr_bad_dataset_threshold=0.70,
        stderr_bad_dataset_top_k=8,
        dataset_log_every=5,
    )
    wall_s = time.perf_counter() - started
    if total != len(names):
        raise RuntimeError(f"shard dataset count mismatch: expected={len(names)} actual={total}")
    atomic_json(
        output_root / "shard_result.json",
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "model_tag": model_tag,
            "shard_index": args.shard_index,
            "dataset_count": total,
            "average_accuracy": avg_acc,
            "aggregate_infer_s": total_t,
            "average_infer_s": avg_t,
            "average_train_ratio": avg_tr,
            "wall_s": wall_s,
            "explicit_fp32": True,
            "clf_use_amp": False,
            "clf_use_fa3": False,
            "n_estimators": 32,
            "outer_batch": 8,
            "n_jobs": 1,
            "kv_cache": False,
            "deterministic_policies": {
                shape: {encoder: list(values) for encoder, values in policy.items()}
                for shape, policy in POLICIES.items()
            },
        },
    )


if __name__ == "__main__":
    main()

