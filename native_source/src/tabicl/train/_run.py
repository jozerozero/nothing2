from __future__ import annotations

import atexit
import hashlib
import json
import os
import random
import re
import sys
import time
import timeit
import warnings
import functools
import inspect
import threading
from collections import deque
from contextlib import contextmanager, nullcontext
from datetime import timedelta

import math
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp

from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from tabicl._model.tabicl import TabICL
from tabicl._sklearn.preprocessing import PreprocessingPipeline
from tabicl.prior._dataset import PriorDataset
from tabicl.prior._genload import LoadPriorDataset
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.train._optim import Muon, get_scheduler
from tabicl.train._train_config import build_parser
from tabicl.train._t25_regression_adapter import (
    build_t25_prior, run_regression_micro_batch, validate_regression_task,
)

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


_SWIGLU_FFN_MATRIX_SUFFIXES = (
    ".linear1.weight",
    ".linear2.weight",
    ".swiglu_value_proj.weight",
)


def is_swiglu_down_matrix(config, name: str) -> bool:
    """Return whether *name* is an ICL SwiGLU down-projection matrix."""
    return bool(getattr(config, "swiglu_enabled", False)) and (
        "icl_predictor.tf_icl.blocks." in name and name.endswith(".linear2.weight")
    )


def resolve_muon_hparams(config, name: str, param: torch.Tensor) -> tuple[float, float]:
    """Resolve Muon LR/WD while preserving its shape-invariant update RMS.

    After zero-power orthogonalization an ``m x n`` update has Frobenius norm
    approximately ``sqrt(min(m, n))``.  Multiplying the learning rate by
    ``0.2 * sqrt(max(m, n))`` therefore makes the *per-element* update RMS
    approximately ``0.2 * global_lr`` for every matrix shape.  A narrower,
    parameter-matched SwiGLU must keep this native rule; forcing its LR to the
    E4 width would over-scale every element by ``sqrt(1024 / width)``.
    ``swiglu_muon_lr_multiplier`` is retained only as an explicit ablation.
    """
    lr = float(config.lr)
    if param.ndim >= 2:
        rows = int(param.shape[0])
        cols = int(param.numel() // max(rows, 1))
        lr *= 0.2 * math.sqrt(max(rows, cols))
    is_swiglu_ffn_matrix = (
        "icl_predictor.tf_icl.blocks." in name
        and name.endswith(_SWIGLU_FFN_MATRIX_SUFFIXES)
    )
    if bool(getattr(config, "swiglu_enabled", False)) and is_swiglu_ffn_matrix:
        lr *= float(getattr(config, "swiglu_muon_lr_multiplier", 1.0))
    weight_decay = float(config.weight_decay)
    value_override = float(getattr(config, "swiglu_value_proj_weight_decay", -1.0))
    if value_override >= 0.0 and ".swiglu_value_proj." in name:
        weight_decay = value_override
    return lr, weight_decay


def validate_training_stage_manifest(config) -> tuple[str, str]:
    """Validate the immutable stage manifest named by the launcher.

    A non-empty environment variable is not evidence of immutability.  Formal
    paired runs require both an exact lowercase digest and an on-disk manifest
    whose bytes produce that digest.
    """
    manifest_sha = str(os.environ.get("TRAINING_STAGE_MANIFEST_SHA256", "")).strip()
    manifest_path = str(os.environ.get("TRAINING_STAGE_MANIFEST_PATH", "")).strip()
    if not bool(getattr(config, "strict_training_stage_manifest", False)):
        return manifest_sha, manifest_path
    if re.fullmatch(r"[0-9a-f]{64}", manifest_sha) is None:
        raise ValueError(
            "--strict_training_stage_manifest requires a lowercase 64-hex "
            "TRAINING_STAGE_MANIFEST_SHA256."
        )
    if not manifest_path or not os.path.isfile(manifest_path):
        raise ValueError(
            "--strict_training_stage_manifest requires an existing "
            "TRAINING_STAGE_MANIFEST_PATH."
        )
    with open(manifest_path, "rb") as manifest_handle:
        actual_manifest_sha = hashlib.sha256(manifest_handle.read()).hexdigest()
    if actual_manifest_sha != manifest_sha:
        raise ValueError(
            "training stage manifest digest mismatch: "
            f"expected={manifest_sha} actual={actual_manifest_sha} path={manifest_path}"
        )
    return manifest_sha, manifest_path


class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False  # Don't suppress exceptions


class NonFiniteMicroBatchError(RuntimeError):
    """Raised when a micro-batch would poison the optimizer state."""


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution.

    Ensures that destroy_process_group() is called if DDP is enabled,
    even if an exception occurs during method execution.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if getattr(self, "prior_cache", None) is not None:
                self.prior_cache.stop()
            if self.ddp:
                destroy_process_group()

    return wrapper


class AsyncPriorCache:
    """Rank-local in-memory cache that asynchronously prefetches prior batches."""

    def __init__(
        self,
        max_batches: int,
        max_bytes: int,
        get_timeout_s: float,
        put_timeout_s: float,
    ):
        if max_batches <= 0:
            raise ValueError("prior_cache_max_batches must be positive.")
        if max_bytes <= 0:
            raise ValueError("prior_cache_max_gb must be positive.")

        self.max_batches = max_batches
        self.max_bytes = max_bytes
        self.get_timeout_s = get_timeout_s
        self.put_timeout_s = put_timeout_s

        self._queue = deque()
        self._current_bytes = 0
        self._producer_done = False
        self._stop_requested = False
        self._producer_exception = None
        self._source_iter = None
        self._producer_thread = None
        self._cv = threading.Condition()

        self.put_wait_count = 0
        self.put_wait_time_s = 0.0
        self.get_wait_count = 0
        self.get_wait_time_s = 0.0
        self.total_batches_produced = 0
        self.total_batches_consumed = 0
        self.total_bytes_produced = 0

    @staticmethod
    def _estimate_obj_bytes(obj) -> int:
        if torch.is_tensor(obj):
            if obj.is_nested:
                return sum(AsyncPriorCache._estimate_obj_bytes(t) for t in obj.unbind())
            return obj.element_size() * obj.numel()
        if isinstance(obj, (list, tuple)):
            return sum(AsyncPriorCache._estimate_obj_bytes(item) for item in obj)
        if isinstance(obj, dict):
            return sum(AsyncPriorCache._estimate_obj_bytes(item) for item in obj.values())
        return 0

    def start(self, source_iter):
        self._source_iter = source_iter
        self._producer_thread = threading.Thread(target=self._producer_loop, name="prior-cache-producer", daemon=True)
        self._producer_thread.start()

    def _producer_loop(self):
        try:
            while True:
                with self._cv:
                    if self._stop_requested:
                        break

                batch = next(self._source_iter)
                batch_bytes = self._estimate_obj_bytes(batch)
                if batch_bytes > self.max_bytes:
                    raise RuntimeError(
                        f"A single batch requires {batch_bytes / (1024 ** 3):.2f} GB, "
                        f"which exceeds prior_cache_max_gb={self.max_bytes / (1024 ** 3):.2f}."
                    )

                wait_started = None
                with self._cv:
                    while not self._stop_requested and (
                        len(self._queue) >= self.max_batches or self._current_bytes + batch_bytes > self.max_bytes
                    ):
                        if wait_started is None:
                            wait_started = time.monotonic()
                            self.put_wait_count += 1
                        notified = self._cv.wait(timeout=self.put_timeout_s)
                        if not notified:
                            raise TimeoutError(
                                "Timed out waiting for free space in prior cache "
                                f"(max_batches={self.max_batches}, max_bytes={self.max_bytes})."
                            )
                    if self._stop_requested:
                        break
                    if wait_started is not None:
                        self.put_wait_time_s += time.monotonic() - wait_started

                    self._queue.append((batch, batch_bytes))
                    self._current_bytes += batch_bytes
                    self.total_batches_produced += 1
                    self.total_bytes_produced += batch_bytes
                    self._cv.notify_all()
        except StopIteration:
            pass
        except Exception as exc:
            with self._cv:
                self._producer_exception = exc
                self._cv.notify_all()
        finally:
            with self._cv:
                self._producer_done = True
                self._cv.notify_all()

    def wait_until_prefilled(self, target_batches: int, target_bytes: int = 0):
        target_batches = min(target_batches, self.max_batches)
        target_bytes = min(max(target_bytes, 0), self.max_bytes)
        if target_batches <= 0 and target_bytes <= 0:
            return

        wait_started = None
        with self._cv:
            while len(self._queue) < target_batches or self._current_bytes < target_bytes:
                if self._producer_exception is not None:
                    raise RuntimeError("Prior cache producer failed during prefill.") from self._producer_exception
                if self._producer_done:
                    raise RuntimeError("Prior cache producer stopped before reaching the configured prefill target.")
                if wait_started is None:
                    wait_started = time.monotonic()
                    self.get_wait_count += 1
                notified = self._cv.wait(timeout=self.get_timeout_s)
                if not notified:
                    raise TimeoutError("Timed out waiting for prior cache prefill to reach the configured target.")
            if wait_started is not None:
                self.get_wait_time_s += time.monotonic() - wait_started

    def get_next_batch(self):
        wait_started = None
        with self._cv:
            while not self._queue:
                if self._producer_exception is not None:
                    raise RuntimeError("Prior cache producer failed.") from self._producer_exception
                if self._producer_done:
                    raise RuntimeError("Prior cache is empty and the producer has already stopped.")
                if wait_started is None:
                    wait_started = time.monotonic()
                    self.get_wait_count += 1
                notified = self._cv.wait(timeout=self.get_timeout_s)
                if not notified:
                    producer_alive = self._producer_thread.is_alive() if self._producer_thread is not None else False
                    raise TimeoutError(
                        "Timed out waiting for data from prior cache "
                        f"(queue={len(self._queue)}, current_gb={self._current_bytes / (1024 ** 3):.3f}, "
                        f"produced={self.total_batches_produced}, consumed={self.total_batches_consumed}, "
                        f"producer_done={self._producer_done}, producer_alive={producer_alive})."
                    )

            if wait_started is not None:
                self.get_wait_time_s += time.monotonic() - wait_started

            batch, batch_bytes = self._queue.popleft()
            self._current_bytes -= batch_bytes
            self.total_batches_consumed += 1
            self._cv.notify_all()
            return batch

    def get_stats(self):
        with self._cv:
            avg_batch_bytes = self.total_bytes_produced / self.total_batches_produced if self.total_batches_produced else 0.0
            return {
                "prior_cache_batches": len(self._queue),
                "prior_cache_gb": self._current_bytes / (1024 ** 3),
                "prior_cache_avg_batch_mb": avg_batch_bytes / (1024 ** 2),
                "prior_cache_put_wait_s": self.put_wait_time_s,
                "prior_cache_get_wait_s": self.get_wait_time_s,
                "prior_cache_put_waits": self.put_wait_count,
                "prior_cache_get_waits": self.get_wait_count,
            }

    def stop(self):
        with self._cv:
            self._stop_requested = True
            self._queue.clear()
            self._current_bytes = 0
            self._cv.notify_all()

        shutdown_workers = getattr(self._source_iter, "_shutdown_workers", None)
        if callable(shutdown_workers):
            try:
                shutdown_workers()
            except Exception:
                pass

        if self._producer_thread is not None:
            self._producer_thread.join(timeout=5)


class Trainer:
    """This class handles the complete training lifecycle for TabICL, including:

    - Environment setup and distributed training configuration
    - Model building and initialization
    - Optimizer, scheduler, and dataloader configuration
    - Checkpoint management and recovery
    - Training loop execution with gradient accumulation
    - Metrics tracking and logging using wandb

    Parameters
    ----------
    config : argparse.Namespace
        Training configuration parameters containing all settings for model,
        optimizer, distributed training, and data generation.
    """

    def __init__(self, config):
        self.config = config
        validate_regression_task(config)
        self.prior_cache = None
        self.stage3_teacher_model = None
        self.plasticity_reference_model = None
        self.plasticity_reference_max_classes = None
        self.continual_bp_hooks = []
        self.continual_bp_layers = []
        self.continual_bp_modules = {}
        self.continual_bp_utility = {}
        self.continual_bp_age = {}
        self.continual_bp_credit = {}
        self.continual_bp_activation_sum = {}
        self.continual_bp_activation_count = {}
        self._late_icl_freeze_params = None
        self._late_icl_freeze_param_count = 0
        self.stage3_anchor_params = {}
        self.stage3_anchor_numel = 0
        self.stage3_ema_state = None
        self.stage3_ema_ready = False
        self.firewall_kd_data = None
        self._bad_flag_tensor = None
        self.validate_experimental_switches()
        self.configure_ddp()
        self.configure_bad_batch_logging()
        self.configure_batch_source_logging()
        self.configure_paired_batch_audit()
        self.configure_speed_trace()
        self.configure_wandb()
        self.build_model()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()
        # Do not start/prefill stochastic DataLoader workers until the model,
        # optimizer and any resume contract have been resolved.
        self.configure_prior()
        self.configure_plasticity_reference()
        self.configure_continual_bp()
        self.configure_stage3_anchor()
        self.configure_stage3_teacher()
        self.configure_firewall_kd()
        self.configure_stage3_ema()

    def debug_log(self, message: str):
        """Print early-step timing checkpoints when debug_timing is enabled."""
        if not getattr(self.config, "debug_timing", False):
            return
        if self.curr_step >= getattr(self.config, "debug_timing_steps", 0):
            return
        print(
            f"[debug rank={self.ddp_rank} local={self.ddp_local_rank} step={self.curr_step}] {message}",
            flush=True,
        )

    def warning_log(self, message: str):
        print(f"[warn rank={self.ddp_rank} local={self.ddp_local_rank} step={self.curr_step}] {message}", flush=True)

    def _profile_timing_enabled(self) -> bool:
        if not (self.master_process and bool(getattr(self.config, "profile_timing", False))):
            return False
        every = int(getattr(self.config, "profile_timing_every", 1) or 0)
        if every <= 0 or self.curr_step % every != 0:
            return False
        until_step = int(getattr(self.config, "profile_timing_until_step", 200) or 0)
        return until_step < 0 or self.curr_step <= until_step

    def _profile_timing_sync(self):
        if (
            bool(getattr(self.config, "profile_timing_sync_cuda", True))
            and "cuda" in str(self.config.device)
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize()

    @contextmanager
    def _timed_phase(self, timings: dict[str, float] | None, name: str):
        if timings is None:
            yield
            return
        self._profile_timing_sync()
        start = timeit.default_timer()
        try:
            yield
        finally:
            self._profile_timing_sync()
            timings[name] = timings.get(name, 0.0) + (timeit.default_timer() - start)

    def configure_bad_batch_logging(self):
        self.bad_batch_records_written = 0
        self.bad_batch_matches_seen = 0
        self.bad_batch_log_path = None
        if not getattr(self.config, "bad_batch_log_enabled", False):
            return
        if not self.master_process:
            return

        log_path = getattr(self.config, "bad_batch_log_path", None)
        if not log_path:
            base_dir = self.config.checkpoint_dir or os.getcwd()
            log_path = os.path.join(base_dir, "bad_batches_rank0.jsonl")
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.bad_batch_log_path = log_path
        print(
            "Bad-batch diagnostics enabled: "
            f"path={self.bad_batch_log_path} "
            f"start_step={self.config.bad_batch_start_step} "
            f"accuracy_threshold={self.config.bad_batch_accuracy_threshold} "
            f"ce_threshold={self.config.bad_batch_ce_threshold}",
            file=sys.stderr,
            flush=True,
        )

    def configure_batch_source_logging(self):
        self.batch_source_records_written = 0
        self.batch_source_buffer = []
        self.batch_source_log_path = None
        self.batch_source_flush_every = 1
        if not getattr(self.config, "batch_source_log_enabled", False):
            return
        if not self.master_process:
            return

        log_path = getattr(self.config, "batch_source_log_path", None)
        if not log_path:
            base_dir = self.config.checkpoint_dir or os.getcwd()
            log_path = os.path.join(base_dir, "batch_sources_rank0.jsonl")
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.batch_source_log_path = log_path
        self.batch_source_flush_every = max(1, int(getattr(self.config, "batch_source_flush_every", 100) or 100))
        atexit.register(self._flush_batch_source_log)
        print(
            "Batch-source diagnostics enabled: "
            f"path={self.batch_source_log_path} "
            f"log_every={self.config.batch_source_log_every} "
            f"flush_every={self.batch_source_flush_every}",
            file=sys.stderr,
            flush=True,
        )

    def configure_paired_batch_audit(self):
        """Configure all-rank, fresh-run data fingerprints for paired arms.

        Each rank owns one file, so no distributed collective or shared-file
        append is introduced into the training path.  The audit reads already
        materialized batch tensors and therefore consumes no RNG.
        """
        self.paired_batch_audit_steps = int(
            getattr(self.config, "paired_batch_audit_steps", 0) or 0
        )
        self.paired_batch_audit_path = None
        if self.paired_batch_audit_steps <= 0:
            return

        audit_dir = getattr(self.config, "paired_batch_audit_dir", None)
        if not audit_dir:
            base_dir = self.config.checkpoint_dir or os.getcwd()
            audit_dir = os.path.join(base_dir, "paired_batch_audit")
        os.makedirs(audit_dir, exist_ok=True)
        self.paired_batch_audit_path = os.path.join(
            audit_dir, f"rank-{int(self.ddp_rank):05d}.jsonl"
        )
        # Formal runs are fresh-only.  Refuse an ambiguous append or a stale
        # artifact rather than silently accepting a mixed data trace.
        if os.path.exists(self.paired_batch_audit_path):
            raise FileExistsError(
                f"paired-batch audit file already exists: {self.paired_batch_audit_path}"
            )
        print(
            "Paired-batch all-rank audit enabled: "
            f"path={self.paired_batch_audit_path} "
            f"steps={self.paired_batch_audit_steps} "
            f"sample_values={int(self.config.paired_batch_audit_sample_values)}",
            file=sys.stderr,
            flush=True,
        )

    def configure_speed_trace(self):
        self.speed_trace_path = None
        if not self.master_process:
            return
        every = int(getattr(self.config, "speed_trace_every", 0) or 0)
        if every <= 0:
            return
        log_path = getattr(self.config, "speed_trace_path", None)
        if not log_path:
            base_dir = self.config.checkpoint_dir or os.getcwd()
            log_path = os.path.join(base_dir, "speed_trace_rank0.jsonl")
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.speed_trace_path = log_path
        print(
            "Speed trace enabled: "
            f"path={self.speed_trace_path} "
            f"every={every} "
            f"until_step={getattr(self.config, 'speed_trace_until_step', -1)}",
            file=sys.stderr,
            flush=True,
        )

    def _write_speed_trace(self, step: int, results: dict):
        if self.speed_trace_path is None:
            return
        every = int(getattr(self.config, "speed_trace_every", 0) or 0)
        if every <= 0 or step % every != 0:
            return
        until_step = int(getattr(self.config, "speed_trace_until_step", -1) or -1)
        if until_step >= 0 and step > until_step:
            return

        numeric_results = {}
        for key, value in results.items():
            if isinstance(value, bool):
                numeric_results[key] = bool(value)
            elif isinstance(value, (int, float)):
                numeric_results[key] = float(value)
        record = {
            "step": int(step),
            "completed_step": int(step + 1),
            "time": time.time(),
            "rank": int(self.ddp_rank),
            "world_size": int(self.ddp_world_size),
            "strategy_id": os.environ.get("STRATEGY_ID", ""),
            "metrics": numeric_results,
        }
        with open(self.speed_trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _flush_batch_source_log(self):
        if self.batch_source_log_path is None or not getattr(self, "batch_source_buffer", None):
            return
        records = self.batch_source_buffer
        self.batch_source_buffer = []
        with open(self.batch_source_log_path, "a", encoding="utf-8") as f:
            f.writelines(records)

    def _sync_bad_flag(self, local_bad: bool) -> bool:
        if not self.ddp:
            return bool(local_bad)
        device = torch.device(self.config.device)
        flag = self._bad_flag_tensor
        if flag is None or flag.device != device:
            flag = torch.empty(1, device=device, dtype=torch.int32)
            self._bad_flag_tensor = flag
        flag.fill_(1 if local_bad else 0)
        all_reduce(flag, op=ReduceOp.MAX)
        return bool(flag.item())

    def _nonfinite_checks_enabled(self) -> bool:
        every = int(getattr(self.config, "nonfinite_check_every", 1) or 0)
        if every <= 0:
            return False
        until_step = int(getattr(self.config, "nonfinite_check_until_step", -1) or -1)
        if until_step >= 0 and self.curr_step > until_step:
            return False
        return self.curr_step % every == 0

    def _sync_nonfinite_issue(self, issue: str | None) -> bool:
        if not self._nonfinite_checks_enabled():
            return False
        return self._sync_bad_flag(issue is not None)

    def _nonfinite_tensor_reason(self, name: str, tensor: torch.Tensor) -> str | None:
        if not self._nonfinite_checks_enabled():
            return None
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            return None
        finite = torch.isfinite(tensor)
        if bool(finite.all().item()):
            return None
        nonfinite = int((~finite).sum().item())
        nan = int(torch.isnan(tensor).sum().item())
        inf = int(torch.isinf(tensor).sum().item())
        return f"{name}: nonfinite={nonfinite} nan={nan} inf={inf} shape={tuple(tensor.shape)}"

    def _local_micro_batch_issue(self, micro_X: torch.Tensor, micro_y: torch.Tensor) -> str | None:
        if not self._nonfinite_checks_enabled():
            return None
        reasons = []
        for name, tensor in (("micro_X", micro_X), ("micro_y", micro_y)):
            reason = self._nonfinite_tensor_reason(name, tensor)
            if reason is not None:
                reasons.append(reason)
        return "; ".join(reasons) if reasons else None

    def _local_forward_issue(self, pred: torch.Tensor, true: torch.Tensor, loss: torch.Tensor | None = None) -> str | None:
        if not self._nonfinite_checks_enabled():
            return None
        reasons = []
        reason = self._nonfinite_tensor_reason("logits", pred)
        if reason is not None:
            reasons.append(reason)
        if true.numel() == 0:
            reasons.append("target: empty after train/test split")
        else:
            min_target = int(true.min().item())
            max_target = int(true.max().item())
            if min_target < 0 or max_target >= pred.shape[-1]:
                reasons.append(f"target: out_of_range min={min_target} max={max_target} classes={pred.shape[-1]}")
        if loss is not None:
            reason = self._nonfinite_tensor_reason("loss", loss)
            if reason is not None:
                reasons.append(reason)
        return "; ".join(reasons) if reasons else None

    def _local_nonfinite_grad_reason(self) -> str | None:
        if not self._nonfinite_checks_enabled():
            return None
        for name, param in self.raw_model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            reason = self._nonfinite_tensor_reason(f"grad[{name}]", grad)
            if reason is not None:
                return reason
        return None

    def _should_log_grad_norm(self) -> bool:
        every = int(getattr(self.config, "log_grad_norm_every", 0) or 0)
        return self.master_process and every > 0 and self.curr_step % every == 0

    def _train_metrics_enabled(self) -> bool:
        every = int(getattr(self.config, "train_metrics_every", 1) or 0)
        if every <= 0:
            return False
        until_step = int(getattr(self.config, "train_metrics_until_step", -1) or -1)
        if until_step >= 0 and self.curr_step > until_step:
            return False
        return self.curr_step % every == 0

    def _model_structure_metrics(self) -> dict[str, float]:
        out = {}
        for attr_name in ("schema_expert_metrics", "function_token_metrics", "layer_gate_metrics"):
            metrics = getattr(self.raw_model, attr_name, {}) or {}
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        continue
                    out[key] = float(value.detach().float().item())
                elif isinstance(value, (int, float)):
                    out[key] = float(value)
        return out

    def _gradient_clip_foreach(self) -> bool | None:
        mode = str(getattr(self.config, "gradient_clip_foreach", "auto") or "auto").lower()
        if mode == "auto":
            return None
        return mode == "true"

    def _gradient_clip_error_sync_enabled(self) -> bool:
        every = int(getattr(self.config, "gradient_clip_error_sync_every", 1) or 0)
        if every <= 0:
            return False
        until_step = int(getattr(self.config, "gradient_clip_error_sync_until_step", -1) or -1)
        if until_step >= 0 and self.curr_step > until_step:
            return False
        return self.curr_step % every == 0

    def _module_grad_norms(self) -> dict[str, float]:
        norms = {}
        total_sq = 0.0
        for module_name, module in self.raw_model.named_modules():
            if module_name == "":
                continue
            module_sq = 0.0
            has_grad = False
            for param in module.parameters(recurse=False):
                if param.grad is None:
                    continue
                has_grad = True
                grad_norm = float(param.grad.detach().float().norm(2).item())
                module_sq += grad_norm * grad_norm
            if has_grad:
                norms[module_name] = math.sqrt(module_sq) if math.isfinite(module_sq) else module_sq
                total_sq += module_sq
        norms["total"] = math.sqrt(total_sq) if math.isfinite(total_sq) else total_sq
        return norms

    @staticmethod
    def _format_norm(value: float) -> str:
        return f"{value:.6g}" if math.isfinite(value) else str(value)

    def _log_module_grad_norms(self, norms: dict[str, float], total_norm: float):
        if not self._should_log_grad_norm():
            return
        parts = [f"total={self._format_norm(total_norm)}"]
        for name in sorted(k for k in norms if k != "total"):
            parts.append(f"{name}={self._format_norm(norms[name])}")
        print(f"[grad_norm step={self.curr_step} preclip] " + " ".join(parts), flush=True)

    @staticmethod
    def _label_summary(labels: torch.Tensor) -> dict:
        labels = labels.detach().long()
        labels = labels[labels >= 0]
        if labels.numel() == 0:
            return {
                "num_labels": 0,
                "num_classes": 0,
                "class_ids": [],
                "counts": [],
                "max_class_frac": 0.0,
                "min_class_count": 0,
                "entropy_norm": 0.0,
            }

        max_label = int(labels.max().item())
        counts = torch.bincount(labels, minlength=max_label + 1).detach().cpu()
        class_ids = torch.nonzero(counts, as_tuple=False).flatten()
        nonzero_counts = counts[class_ids].float()
        total = float(nonzero_counts.sum().item())
        probs = nonzero_counts / max(total, 1.0)
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        entropy_norm = entropy / math.log(len(nonzero_counts)) if len(nonzero_counts) > 1 else 0.0
        return {
            "num_labels": int(total),
            "num_classes": int(len(nonzero_counts)),
            "class_ids": [int(v) for v in class_ids.tolist()],
            "counts": [int(v) for v in nonzero_counts.tolist()],
            "max_class_frac": float(nonzero_counts.max().item() / max(total, 1.0)),
            "min_class_count": int(nonzero_counts.min().item()),
            "entropy_norm": float(entropy_norm),
        }

    @staticmethod
    def _feature_summary(features: torch.Tensor) -> dict:
        features = features.detach().float()
        if features.numel() == 0 or features.shape[-1] == 0:
            return {
                "active_features": int(features.shape[-1]) if features.ndim > 0 else 0,
                "feature_abs_mean": 0.0,
                "feature_std_mean": 0.0,
                "feature_std_max": 0.0,
                "constant_feature_frac": 0.0,
                "zero_frac": 0.0,
            }

        col_std = features.std(dim=0, unbiased=False)
        return {
            "active_features": int(features.shape[-1]),
            "feature_abs_mean": float(features.abs().mean().item()),
            "feature_std_mean": float(col_std.mean().item()),
            "feature_std_max": float(col_std.max().item()),
            "constant_feature_frac": float((col_std < 1e-6).float().mean().item()),
            "zero_frac": float((features.abs() < 1e-8).float().mean().item()),
        }

    @staticmethod
    def _metadata_at(source_metadata, index: int):
        if source_metadata is None:
            return None
        try:
            return source_metadata[index]
        except (IndexError, KeyError, TypeError):
            return None

    @staticmethod
    def _split_metadata(source_metadata, chunk_size: int):
        if source_metadata is None:
            return None
        if isinstance(source_metadata, tuple):
            source_metadata = list(source_metadata)
        if not isinstance(source_metadata, list):
            return None
        return [source_metadata[start : start + chunk_size] for start in range(0, len(source_metadata), chunk_size)]

    @staticmethod
    def _format_top_counts(counts, limit: int = 3) -> str:
        if not isinstance(counts, dict) or not counts:
            return "none"
        items = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
        return ",".join(f"{key}:{value}" for key, value in items)

    def _source_preview(self, source) -> str:
        if not isinstance(source, dict):
            return "source=n/a"
        return (
            f"nodes={source.get('num_graph_nodes', 'n/a')} "
            f"edges={source.get('num_graph_edges', 'n/a')} "
            f"cat={source.get('input_categorical_features', 'n/a')}/"
            f"{source.get('requested_num_features', 'n/a')} "
            f"classes={source.get('final_num_classes', source.get('sampled_num_classes', 'n/a'))} "
            f"funcs={self._format_top_counts(source.get('random_function_kinds'))} "
            f"converters={self._format_top_counts(source.get('input_converter_kinds'))}"
        )

    def _maybe_log_batch_source(self, batch_tensors, source_metadata):
        if self.batch_source_log_path is None:
            return
        max_records = int(getattr(self.config, "batch_source_max_records", 0) or 0)
        if max_records > 0 and self.batch_source_records_written >= max_records:
            return
        log_every = int(getattr(self.config, "batch_source_log_every", 1) or 1)
        if self.curr_step % log_every != 0:
            return

        _, _, d_batch, seq_lens, train_sizes = batch_tensors
        datasets = []
        batch_size = int(seq_lens.shape[0])
        for i in range(batch_size):
            seq_len_i = int(seq_lens[i].item())
            train_size_i = int(train_sizes[i].item())
            source = self._metadata_at(source_metadata, i)
            datasets.append(
                {
                    "idx": int(i),
                    "seq_len": seq_len_i,
                    "train_size": train_size_i,
                    "train_ratio": float(train_size_i / max(seq_len_i, 1)),
                    "d": int(d_batch[i].item()),
                    "source": source,
                }
            )

        record = {
            "step": int(self.curr_step),
            "rank": int(self.ddp_rank),
            "local_rank": int(self.ddp_local_rank),
            "prior_type": self.config.prior_type,
            "batch_fingerprint": self._batch_fingerprint(batch_tensors),
            "datasets": datasets,
        }
        self.batch_source_buffer.append(json.dumps(record, sort_keys=True) + "\n")
        self.batch_source_records_written += 1
        if len(self.batch_source_buffer) >= self.batch_source_flush_every:
            self._flush_batch_source_log()

        if getattr(self.config, "batch_source_print_to_stderr", True) and datasets:
            print(
                "[batch_source] "
                f"step={record['step']} rank={record['rank']} "
                f"{self._source_preview(datasets[0].get('source'))} "
                f"jsonl={self.batch_source_log_path}",
                file=sys.stderr,
                flush=True,
            )

    @staticmethod
    def _batch_fingerprint(batch_tensors, sample_values: int = 128) -> str:
        """Cheap architecture-independent fingerprint for paired-data audits."""
        digest = hashlib.sha256()
        for tensor in batch_tensors:
            if not torch.is_tensor(tensor):
                digest.update(repr(type(tensor)).encode("utf-8"))
                continue
            value = tensor.detach()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            flat = value.reshape(-1)
            if flat.numel() == 0:
                continue
            count = min(int(sample_values), int(flat.numel()))
            indices = torch.linspace(0, int(flat.numel()) - 1, count, device=flat.device).long()
            sample = flat.index_select(0, indices).contiguous().cpu()
            digest.update(sample.numpy().tobytes())
        return digest.hexdigest()

    def _maybe_write_paired_batch_audit(self, batch_tensors) -> None:
        """Write one fail-closed data-fingerprint record from this rank."""
        if self.paired_batch_audit_path is None:
            return
        step = int(self.curr_step)
        if step < 0 or step >= self.paired_batch_audit_steps:
            return
        sample_values = int(self.config.paired_batch_audit_sample_values)
        record = {
            "schema_version": 1,
            "step": step,
            "rank": int(self.ddp_rank),
            "local_rank": int(self.ddp_local_rank),
            "world_size": int(self.ddp_world_size),
            "prior_loader_seed": int(self.config.prior_loader_seed),
            "sample_values_per_tensor": sample_values,
            "batch_fingerprint_sha256": self._batch_fingerprint(
                batch_tensors, sample_values=sample_values
            ),
            "tensor_shapes": [
                list(tensor.shape) if torch.is_tensor(tensor) else None
                for tensor in batch_tensors
            ],
            "tensor_dtypes": [
                str(tensor.dtype) if torch.is_tensor(tensor) else str(type(tensor))
                for tensor in batch_tensors
            ],
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        # Exclusive creation on the first record rejects stale runs.  Later
        # records append only to this rank-owned file and are fsynced because
        # these first few records are release-gate evidence.
        mode = "x" if step == 0 else "a"
        with open(self.paired_batch_audit_path, mode, encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _maybe_log_bad_micro_batch(
        self,
        micro_batch_idx: int,
        num_micro_batches: int,
        micro_X: torch.Tensor,
        micro_y: torch.Tensor,
        micro_d: torch.Tensor,
        micro_seq_len: torch.Tensor,
        micro_train_size: torch.Tensor,
        ce: float,
        accuracy: float,
        source_metadata=None,
    ):
        if self.bad_batch_log_path is None:
            return
        if self.curr_step < int(getattr(self.config, "bad_batch_start_step", 0)):
            return
        max_records = int(getattr(self.config, "bad_batch_max_records", 0) or 0)
        if max_records > 0 and self.bad_batch_records_written >= max_records:
            return

        acc_threshold = float(getattr(self.config, "bad_batch_accuracy_threshold", -1.0))
        ce_threshold = float(getattr(self.config, "bad_batch_ce_threshold", float("inf")))
        if accuracy > acc_threshold and ce < ce_threshold:
            return
        self.bad_batch_matches_seen += 1
        log_every = int(getattr(self.config, "bad_batch_log_every", 1) or 1)
        if log_every > 1 and self.bad_batch_matches_seen % log_every != 0:
            return

        datasets = []
        batch_size = int(micro_y.shape[0])
        for i in range(batch_size):
            seq_len_i = int(micro_seq_len[i].item())
            train_size_i = int(micro_train_size[i].item())
            d_i = int(micro_d[i].item())
            y_i = micro_y[i, :seq_len_i]
            train_y = y_i[:train_size_i]
            test_y = y_i[train_size_i:]
            train_summary = self._label_summary(train_y)
            test_summary = self._label_summary(test_y)
            train_classes = set(train_summary["class_ids"])
            test_classes = set(test_summary["class_ids"])
            datasets.append(
                {
                    "idx": i,
                    "seq_len": seq_len_i,
                    "train_size": train_size_i,
                    "train_ratio": float(train_size_i / max(seq_len_i, 1)),
                    "d": d_i,
                    "num_classes_total": int(y_i.max().item() + 1) if y_i.numel() else 0,
                    "train_labels": train_summary,
                    "test_labels": test_summary,
                    "test_classes_missing_from_train": sorted(int(c) for c in (test_classes - train_classes)),
                    "features": self._feature_summary(micro_X[i, :seq_len_i, :d_i]),
                    "source": self._metadata_at(source_metadata, i),
                }
            )

        record = {
            "step": int(self.curr_step),
            "rank": int(self.ddp_rank),
            "local_rank": int(self.ddp_local_rank),
            "micro_batch_idx": int(micro_batch_idx),
            "num_micro_batches": int(num_micro_batches),
            "prior_type": self.config.prior_type,
            "accuracy": float(accuracy),
            "ce": float(ce),
            "datasets": datasets,
        }
        with open(self.bad_batch_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self.bad_batch_records_written += 1

        if getattr(self.config, "bad_batch_print_to_stderr", True) and datasets:
            first = datasets[0]
            train_labels = first["train_labels"]
            test_labels = first["test_labels"]
            features = first["features"]
            print(
                "[bad_batch] "
                f"step={record['step']} rank={record['rank']} "
                f"acc={accuracy:.4f} ce={ce:.4f} "
                f"seq_len={first['seq_len']} train_ratio={first['train_ratio']:.3f} "
                f"d={first['d']} classes={first['num_classes_total']} "
                f"train_classes={train_labels['num_classes']} test_classes={test_labels['num_classes']} "
                f"train_max_frac={train_labels['max_class_frac']:.3f} "
                f"test_max_frac={test_labels['max_class_frac']:.3f} "
                f"constant_frac={features['constant_feature_frac']:.3f} "
                f"missing_test_classes={len(first['test_classes_missing_from_train'])} "
                f"{self._source_preview(first.get('source'))} "
                f"jsonl={self.bad_batch_log_path}",
                file=sys.stderr,
                flush=True,
            )

    def configure_ddp(self):
        """Set up distributed training and system configuration.

        This method:
        1. Configures distributed data parallel (DDP) if enabled
        2. Sets up device and process information
        3. Adjusts batch size for multi-GPU training
        4. Sets random seeds for reproducibility
        """
        # Setup distributed training
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            ddp_timeout_s = int(os.environ.get("TABICL_DDP_TIMEOUT_S", "600"))
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            ddp_device = torch.device(f"cuda:{self.ddp_local_rank}")
            self.config.device = str(ddp_device)
            torch.cuda.set_device(ddp_device)
            init_process_group(
                backend="nccl",
                timeout=timedelta(seconds=ddp_timeout_s),
                device_id=ddp_device,
            )

            # Adjust batch size for distributed training
            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(original_batch_size / self.ddp_world_size)

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.batch_size}")
                else:
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by world size ({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.batch_size}.\n"
                        f"Effective batch size is {self.config.batch_size * self.ddp_world_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("No DDP training")

        self.curr_step = 0  # Initialize current step for training

        # Set random seeds
        seed_offset = self.ddp_rank if self.ddp else 0
        self.seed_offset = seed_offset
        python_seed = self.config.np_seed + seed_offset
        random.seed(python_seed)
        np.random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        allow_tf32 = bool(getattr(self.config, "allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if self.master_process:
            print(f"torch allow_tf32: {allow_tf32}")
        float32_matmul_precision = str(getattr(self.config, "float32_matmul_precision", "") or "").strip()
        if float32_matmul_precision:
            torch.set_float32_matmul_precision(float32_matmul_precision)
            if self.master_process:
                print(f"torch float32 matmul precision: {torch.get_float32_matmul_precision()}")
        self.configure_sdpa_backends()

    def configure_sdpa_backends(self):
        requested = str(getattr(self.config, "sdpa_backends", "") or "").strip().lower()
        if not requested:
            return

        aliases = {
            "flash": "flash",
            "flash_attention": "flash",
            "mem": "mem_efficient",
            "mem_efficient": "mem_efficient",
            "efficient": "mem_efficient",
            "efficient_attention": "mem_efficient",
            "math": "math",
            "cudnn": "cudnn",
            "cudnn_attention": "cudnn",
            "all": "all",
        }
        parsed = []
        for item in re.split(r"[,+\s]+", requested):
            if not item:
                continue
            if item not in aliases:
                raise ValueError(
                    f"Unknown --sdpa_backends entry {item!r}; supported={sorted(aliases)}"
                )
            parsed.append(aliases[item])
        if not parsed:
            return

        enabled = {"flash", "mem_efficient", "math", "cudnn"} if "all" in parsed else set(parsed)
        backend_specs = [
            ("flash", "enable_flash_sdp", "flash_sdp_enabled"),
            ("mem_efficient", "enable_mem_efficient_sdp", "mem_efficient_sdp_enabled"),
            ("math", "enable_math_sdp", "math_sdp_enabled"),
            ("cudnn", "enable_cudnn_sdp", "cudnn_sdp_enabled"),
        ]
        statuses = {}
        for name, enable_name, status_name in backend_specs:
            enable_fn = getattr(torch.backends.cuda, enable_name, None)
            if enable_fn is None:
                if name in enabled:
                    raise RuntimeError(f"PyTorch does not expose torch.backends.cuda.{enable_name}.")
                continue
            enable_fn(name in enabled)
            status_fn = getattr(torch.backends.cuda, status_name, None)
            statuses[name] = bool(status_fn()) if status_fn is not None else (name in enabled)

        if self.master_process:
            status_text = ", ".join(f"{name}={value}" for name, value in statuses.items())
            print(f"torch SDPA backends: requested={requested} {status_text}")

    @staticmethod
    def _parse_first_int(value):
        if value is None:
            return None
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else None

    def resolve_prior_num_workers(self):
        if self.config.prior_num_workers >= 0:
            return self.config.prior_num_workers

        local_world_size = (
            self._parse_first_int(os.environ.get("LOCAL_WORLD_SIZE"))
            or self._parse_first_int(os.environ.get("SLURM_GPUS_ON_NODE"))
            or 1
        )
        cpu_budget = self._parse_first_int(os.environ.get("SLURM_CPUS_PER_TASK")) or (os.cpu_count() or 1)
        cpus_per_rank = max(1, cpu_budget // max(1, local_world_size))

        if self.config.prior_n_jobs > 1:
            nested_parallelism = self.config.prior_n_jobs * max(1, self.config.prior_num_threads_per_generate)
            return max(1, min(8, cpus_per_rank // nested_parallelism))

        return max(1, min(8, cpus_per_rank - 2))

    def configure_wandb(self):
        """Set up Weights & Biases logging."""

        if self.config.wandb_log and self.master_process:
            if wandb is None:
                raise ImportError("wandb is required when --wandb_log True. Install tabicl[pretrain] or wandb.")
            id_path = os.path.join(self.config.checkpoint_dir, "wand_id.txt")
            if self.config.wandb_id is None:
                if os.path.exists(id_path):
                    with open(id_path, "r") as f:
                        self.config.wandb_id = f.read().strip()

            self.wandb_run = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                id=self.config.wandb_id,
                config=self.config,
                resume="allow",
                mode=self.config.wandb_mode,
            )

            with open(id_path, "w") as f:
                f.write(self.wandb_run.id)
        else:
            self.wandb_run = None

    def build_model(self):
        """Build and initialize the TabICL model."""

        same_model_seed = self.ddp and not self.config.ddp_init_sync
        if same_model_seed:
            np_rng_state = np.random.get_state()
            torch_rng_state = torch.random.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            np.random.seed(self.config.np_seed)
            torch.manual_seed(self.config.torch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.torch_seed)

        recompute = bool(self.config.recompute)
        if self.config.recompute_seq_len_threshold > 0:
            max_seq_len = self.config.max_seq_len or 0
            recompute = recompute or max_seq_len > self.config.recompute_seq_len_threshold

        self.model_config = {
            "max_classes": 0,
            "num_quantiles": self.config.num_quantiles,
            "embed_dim": self.config.embed_dim,
            "col_num_blocks": self.config.col_num_blocks,
            "col_nhead": self.config.col_nhead,
            "col_num_inds": self.config.col_num_inds,
            "col_feature_group": self.config.col_feature_group,
            "col_feature_group_size": self.config.col_feature_group_size,
            "col_ssmax": self.config.col_ssmax,
            "col_output_layer_norm": self.config.col_output_layer_norm,
            "row_num_blocks": self.config.row_num_blocks,
            "row_nhead": self.config.row_nhead,
            "row_num_cls": self.config.row_num_cls,
            "row_rope_base": self.config.row_rope_base,
            "row_use_rope": self.config.row_use_rope,
            "icl_num_blocks": self.config.icl_num_blocks,
            "icl_nhead": self.config.icl_nhead,
            "icl_ssmax": self.config.icl_ssmax,
            "icl_qassmax_cap_enabled": self.config.qassmax_cap_enabled,
            "icl_qassmax_cap_layers": self.config.qassmax_cap_layers,
            "icl_qassmax_cap_base_scale": self.config.qassmax_cap_base_scale,
            "icl_qassmax_cap_scale": self.config.qassmax_cap_scale,
            "icl_qassmax_cap_query_logit": self.config.qassmax_cap_query_logit,
            "layer_gate_enabled": self.config.layer_gate_enabled,
            "layer_gate_layers": self.config.layer_gate_layers,
            "layer_gate_hidden_dim": self.config.layer_gate_hidden_dim,
            "layer_gate_temperature": self.config.layer_gate_temperature,
            "layer_gate_low_layer_floor": self.config.layer_gate_low_layer_floor,
            "layer_gate_low_layer_max": self.config.layer_gate_low_layer_max,
            "layer_gate_use_confidence_features": self.config.layer_gate_use_confidence_features,
            "layer_gate_repr_source": self.config.layer_gate_repr_source,
            "layer_gate_confidence_source": self.config.layer_gate_confidence_source,
            "layer_gate_max_weight_target": self.config.layer_gate_max_weight_target,
            "attention_gate_enabled": self.config.attention_gate_enabled,
            "attention_gate_shape": self.config.attention_gate_shape,
            "attention_gate_layers": self.config.attention_gate_layers,
            "attention_gate_hidden_dim": self.config.attention_gate_hidden_dim,
            "attention_gate_rho": self.config.attention_gate_rho,
            "swiglu_enabled": self.config.swiglu_enabled,
            "swiglu_conditioned": self.config.swiglu_conditioned,
            "swiglu_hidden_dim": self.config.swiglu_hidden_dim,
            "swiglu_context_hidden_dim": self.config.swiglu_context_hidden_dim,
            "swiglu_rho": self.config.swiglu_rho,
            "swiglu_output_scale": self.config.swiglu_output_scale,
            "swiglu_product_tanh_rms_multiple": self.config.swiglu_product_tanh_rms_multiple,
            "swiglu_product_tanh_last_n_layers": self.config.swiglu_product_tanh_last_n_layers,
            "swiglu_init_seed_base": self.config.swiglu_init_seed_base,
            "cr2_shared_refinement_enabled": self.config.cr2_shared_refinement_enabled,
            "qk_pds_attention_enabled": self.config.qk_pds_attention_enabled,
            "cls8_pooled_enabled": self.config.cls8_pooled_enabled,
            "shared_depth_icl_enabled": self.config.shared_depth_icl_enabled,
            "shared_depth_icl_rho": self.config.shared_depth_icl_rho,
            "shared_depth_icl_dataset_conditioned": self.config.shared_depth_icl_dataset_conditioned,
            "shared_depth_icl_num_passes": self.config.shared_depth_icl_num_passes,
            "cls8_width_enabled": self.config.cls8_width_enabled,
            "schema_expert_enabled": self.config.schema_expert_enabled,
            "schema_expert_num_experts": self.config.schema_expert_num_experts,
            "schema_expert_top_k": self.config.schema_expert_top_k,
            "schema_expert_bottleneck": self.config.schema_expert_bottleneck,
            "schema_expert_hidden_dim": self.config.schema_expert_hidden_dim,
            "schema_expert_router_temperature": self.config.schema_expert_router_temperature,
            "schema_expert_adapter_scale": self.config.schema_expert_adapter_scale,
            "schema_film_enabled": self.config.schema_film_enabled,
            "schema_film_scale": self.config.schema_film_scale,
            "schema_expert_dropout": self.config.schema_expert_dropout,
            "schema_context_source": self.config.schema_context_source,
            "function_tokens_enabled": self.config.function_tokens_enabled,
            "function_token_use_dataset_token": self.config.function_token_use_dataset_token,
            "function_token_use_query_token": self.config.function_token_use_query_token,
            "function_token_use_latent_tokens": self.config.function_token_use_latent_tokens,
            "function_token_dataset_count": self.config.function_token_dataset_count,
            "function_token_query_count": self.config.function_token_query_count,
            "function_token_latent_count": self.config.function_token_latent_count,
            "function_token_num_heads": self.config.function_token_num_heads,
            "function_token_num_layers": self.config.function_token_num_layers,
            "function_token_hidden_dim": self.config.function_token_hidden_dim,
            "function_token_scale": self.config.function_token_scale,
            "function_token_dropout": self.config.function_token_dropout,
            "function_token_context_source": self.config.function_token_context_source,
            "ff_factor": self.config.ff_factor,
            "dropout": self.config.dropout,
            "activation": self.config.activation,
            "norm_first": self.config.norm_first,
            "bias_free_ln": self.config.bias_free_ln,
            "recompute": recompute,
        }
        model = TabICL(**self.model_config)
        if same_model_seed:
            np.random.set_state(np_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        model.to(device=self.config.device)
        from tabicl.train._g5sc_runtime_probe import probe
        probe(model, self.config.checkpoint_dir)

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {num_params} parameters.")
            if same_model_seed:
                print("DDP init_sync is disabled; model weights are initialized from the same seed on every rank.")

        # Freeze model components if requested
        if self.config.freeze_col:
            model.col_embedder.eval()
            for param in model.col_embedder.parameters():
                param.requires_grad = False

        if self.config.freeze_row:
            model.row_interactor.eval()
            for param in model.row_interactor.parameters():
                param.requires_grad = False

        if self.config.freeze_icl:
            model.icl_predictor.eval()
            for param in model.icl_predictor.parameters():
                param.requires_grad = False

        self.apply_qassmax_caps(model)

        compile_mode = getattr(self.config, "model_compile_mode", None) or None

        # Compile model if requested
        if self.config.model_compile:
            model = torch.compile(model, dynamic=True, mode=compile_mode)
            if self.master_process:
                print(f"Model compiled successfully. mode={compile_mode or 'default'}")
        else:
            compile_parts = [
                part.strip().lower()
                for part in re.split(r"[,+:;]", str(getattr(self.config, "model_compile_parts", "") or ""))
                if part.strip()
            ]
            valid_compile_parts = {
                "col": "col_embedder",
                "row": "row_interactor",
                "icl": "icl_predictor",
            }
            unknown_parts = [part for part in compile_parts if part not in valid_compile_parts]
            if unknown_parts:
                raise ValueError(
                    f"Unknown --model_compile_parts values {unknown_parts}; "
                    f"supported={sorted(valid_compile_parts)}"
                )
            if compile_parts and not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is unavailable in this PyTorch build.")
            for part in compile_parts:
                attr_name = valid_compile_parts[part]
                setattr(model, attr_name, torch.compile(getattr(model, attr_name), dynamic=True, mode=compile_mode))
            if compile_parts and self.master_process:
                print(
                    "Model submodules compiled successfully: "
                    f"parts={','.join(compile_parts)} mode={compile_mode or 'default'}"
                )

        # Wrap model into DDP container if using distributed training
        if self.ddp:
            ddp_signature = inspect.signature(DDP).parameters
            ddp_kwargs = {
                "device_ids": [self.ddp_local_rank],
                "broadcast_buffers": False,
                "find_unused_parameters": self.config.ddp_find_unused_parameters,
            }
            bucket_cap_mb = float(getattr(self.config, "ddp_bucket_cap_mb", 0.0) or 0.0)
            if bucket_cap_mb > 0.0 and "bucket_cap_mb" in ddp_signature:
                ddp_kwargs["bucket_cap_mb"] = bucket_cap_mb
            if bool(getattr(self.config, "ddp_gradient_as_bucket_view", False)):
                if "gradient_as_bucket_view" in ddp_signature:
                    ddp_kwargs["gradient_as_bucket_view"] = True
                elif self.master_process:
                    print("Warning: this PyTorch version does not support DDP gradient_as_bucket_view.")
            if bool(getattr(self.config, "ddp_static_graph", False)):
                if "static_graph" in ddp_signature:
                    ddp_kwargs["static_graph"] = True
                elif self.master_process:
                    print("Warning: this PyTorch version does not support DDP static_graph.")
            if "init_sync" in ddp_signature:
                ddp_kwargs["init_sync"] = self.config.ddp_init_sync
            elif not self.config.ddp_init_sync and self.master_process:
                print("Warning: this PyTorch version does not support DDP init_sync=False; using default DDP sync.")
            if self.master_process:
                print(f"DDP kwargs: {ddp_kwargs}")

            self.model = DDP(model, **ddp_kwargs)
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

    def configure_prior(self):
        """Set up a tabular dataset generator for synthetic data during training."""

        if self.config.prior_dir is None:
            # Generate prior data on the fly
            cross_table_enabled = os.environ.get("CROSS_TABLE_ENABLED", "").lower() == "true"
            return_prior_metadata = cross_table_enabled or (
                self.master_process
                and self.config.prior_type in {"tabiclv2_cls", "hybrid178"}
                and (
                    getattr(self.config, "batch_source_log_enabled", False)
                    or getattr(self.config, "bad_batch_log_enabled", False)
                )
            )
            dataset = build_t25_prior(self.config)
        else:
            # Load pre-generated prior data from disk
            dataset = LoadPriorDataset(
                data_dir=self.config.prior_dir,
                batch_size=self.config.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                start_from=self.config.load_prior_start,
                delete_after_load=self.config.delete_after_load,
                device=self.config.prior_device,
            )

        if self.master_process:
            print(dataset)

        prior_num_workers = self.resolve_prior_num_workers()
        if self.master_process:
            print(
                "Prior loader config: "
                f"num_workers={prior_num_workers}, "
                f"prefetch_factor={self.config.prior_prefetch_factor if prior_num_workers > 0 else 'n/a'}, "
                f"persistent_workers={self.config.prior_persistent_workers if prior_num_workers > 0 else False}, "
                f"prior_n_jobs={self.config.prior_n_jobs}, "
                f"prior_num_threads_per_generate={self.config.prior_num_threads_per_generate}, "
                f"pin_memory={self.config.prior_pin_memory and self.config.prior_device == 'cpu'}, "
                f"prior_cache_enabled={self.config.prior_cache_enabled}, "
                f"talent_mix_ratio={self.config.talent_mix_ratio}, "
                f"talent_mix_source={self.config.talent_mix_source}, "
                f"protected_batch_mix_ratio={self.config.protected_batch_mix_ratio}, "
                f"protected_batch_mix_source={self.config.protected_batch_mix_source}, "
                f"tabiclv2_graph_nodes={self.config.tabiclv2_graph_nodes_min}-{self.config.tabiclv2_graph_nodes_max}, "
                f"tabiclv2_node_extra_dim={self.config.tabiclv2_node_extra_dim_min}-{self.config.tabiclv2_node_extra_dim_max}, "
                f"tabiclv2_latent_needed_nodes="
                f"{self.config.tabiclv2_latent_needed_nodes_min}-{self.config.tabiclv2_latent_needed_nodes_max}"
            )

        # Create dataloader for efficient loading and prefetching
        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=None,
            shuffle=False,
            num_workers=prior_num_workers,
            pin_memory=self.config.prior_pin_memory and self.config.prior_device == "cpu",
        )
        # Historical E4 (job 142620) intentionally relied on PyTorch's global
        # RNG state when DataLoader created worker base seeds.  Do not inject
        # the later Phase-G independent generator: it changes every synthetic
        # batch even when the model constructor preserves the global RNG.
        if prior_num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.config.prior_prefetch_factor
            dataloader_kwargs["persistent_workers"] = self.config.prior_persistent_workers

        self.dataloader = DataLoader(**dataloader_kwargs)

        if self.config.prior_cache_enabled:
            if self.config.prior_device != "cpu":
                raise ValueError("prior_cache_enabled currently requires prior_device=cpu.")

            source_iter = iter(self.dataloader)
            self.prior_cache = AsyncPriorCache(
                max_batches=self.config.prior_cache_max_batches,
                max_bytes=int(self.config.prior_cache_max_gb * (1024 ** 3)),
                get_timeout_s=self.config.prior_cache_get_timeout_s,
                put_timeout_s=self.config.prior_cache_put_timeout_s,
            )
            self.prior_cache.start(source_iter)
            self.prior_cache.wait_until_prefilled(
                self.config.prior_cache_prefill_batches,
                target_bytes=int(self.config.prior_cache_prefill_gb * (1024 ** 3)),
            )
            if self.master_process:
                stats = self.prior_cache.get_stats()
                print(
                    "Prior cache ready: "
                    f"{stats['prior_cache_batches']} batches, "
                    f"{stats['prior_cache_gb']:.2f} GB resident."
                )

    def validate_experimental_switches(self):
        """Reject requested experimental losses that are not implemented yet."""
        if "SWIGLU_MATCH_MUON_LR" in os.environ:
            raise ValueError(
                "Legacy SWIGLU_MATCH_MUON_LR is forbidden; use --swiglu_muon_lr_multiplier explicitly."
            )
        validate_training_stage_manifest(self.config)
        if int(getattr(self.config, "prior_loader_seed", 42)) < 0:
            raise ValueError("--prior_loader_seed must be non-negative.")
        paired_audit_steps = int(getattr(self.config, "paired_batch_audit_steps", 0) or 0)
        paired_audit_samples = int(
            getattr(self.config, "paired_batch_audit_sample_values", 4096) or 0
        )
        if paired_audit_steps < 0 or paired_audit_steps > 10:
            raise ValueError("--paired_batch_audit_steps must be within [0, 10].")
        if paired_audit_steps > 0 and not 128 <= paired_audit_samples <= 65536:
            raise ValueError(
                "--paired_batch_audit_sample_values must be within [128, 65536] "
                "when paired-batch auditing is enabled."
            )
        if bool(getattr(self.config, "swiglu_enabled", False)):
            hidden_dim = int(getattr(self.config, "swiglu_hidden_dim", 0))
            output_scale = float(getattr(self.config, "swiglu_output_scale", 1.0))
            product_tanh_multiple = float(
                getattr(self.config, "swiglu_product_tanh_rms_multiple", 0.0)
            )
            product_tanh_last_n = int(
                getattr(self.config, "swiglu_product_tanh_last_n_layers", 0)
            )
            muon_multiplier = float(getattr(self.config, "swiglu_muon_lr_multiplier", 1.0))
            value_wd = float(getattr(self.config, "swiglu_value_proj_weight_decay", -1.0))
            down_bootstrap_multiplier = float(
                getattr(self.config, "swiglu_down_lr_bootstrap_multiplier", 1.0)
            )
            down_bootstrap_steps = int(
                getattr(self.config, "swiglu_down_lr_bootstrap_steps", 0)
            )
            if hidden_dim <= 0:
                raise ValueError("--swiglu_hidden_dim must be positive.")
            if not math.isfinite(output_scale) or output_scale <= 0.0:
                raise ValueError("--swiglu_output_scale must be finite and positive.")
            if not math.isfinite(product_tanh_multiple) or product_tanh_multiple < 0.0:
                raise ValueError(
                    "--swiglu_product_tanh_rms_multiple must be finite and non-negative."
                )
            if not 0 <= product_tanh_last_n <= int(self.config.icl_num_blocks):
                raise ValueError(
                    "--swiglu_product_tanh_last_n_layers must be within "
                    f"[0, {self.config.icl_num_blocks}]."
                )
            if (product_tanh_multiple > 0.0) != (product_tanh_last_n > 0):
                raise ValueError(
                    "SwiGLU product smoothing multiple and last-N depth must be enabled together."
                )
            if not math.isfinite(muon_multiplier) or muon_multiplier <= 0.0:
                raise ValueError("--swiglu_muon_lr_multiplier must be finite and positive.")
            if not math.isfinite(value_wd) or (value_wd < 0.0 and value_wd != -1.0):
                raise ValueError("--swiglu_value_proj_weight_decay must be -1 (inherit) or non-negative.")
            if not math.isfinite(down_bootstrap_multiplier) or not (
                1.0 <= down_bootstrap_multiplier <= 1.5
            ):
                raise ValueError("--swiglu_down_lr_bootstrap_multiplier must be finite and in [1, 1.5].")
            if down_bootstrap_steps < 0 or down_bootstrap_steps > int(self.config.max_steps):
                raise ValueError("--swiglu_down_lr_bootstrap_steps must be in [0, max_steps].")
            if (down_bootstrap_multiplier == 1.0) != (down_bootstrap_steps == 0):
                raise ValueError(
                    "SwiGLU down bootstrap requires multiplier > 1 and steps > 0 together, "
                    "or multiplier=1 and steps=0 together."
                )
            if str(getattr(self.config, "optimizer", "")).lower() != "muon" and (
                muon_multiplier != 1.0 or value_wd >= 0.0 or down_bootstrap_steps > 0
            ):
                raise ValueError("SwiGLU Muon LR/WD overrides require --optimizer muon.")
        else:
            if float(getattr(self.config, "swiglu_muon_lr_multiplier", 1.0)) != 1.0:
                raise ValueError("--swiglu_muon_lr_multiplier must be 1 when SwiGLU is disabled.")
            if float(getattr(self.config, "swiglu_value_proj_weight_decay", -1.0)) != -1.0:
                raise ValueError("--swiglu_value_proj_weight_decay must be -1 when SwiGLU is disabled.")
            if float(getattr(self.config, "swiglu_down_lr_bootstrap_multiplier", 1.0)) != 1.0:
                raise ValueError("--swiglu_down_lr_bootstrap_multiplier must be 1 when SwiGLU is disabled.")
            if int(getattr(self.config, "swiglu_down_lr_bootstrap_steps", 0)) != 0:
                raise ValueError("--swiglu_down_lr_bootstrap_steps must be 0 when SwiGLU is disabled.")
            if float(getattr(self.config, "swiglu_product_tanh_rms_multiple", 0.0)) != 0.0:
                raise ValueError(
                    "--swiglu_product_tanh_rms_multiple must be 0 when SwiGLU is disabled."
                )
            if int(getattr(self.config, "swiglu_product_tanh_last_n_layers", 0)) != 0:
                raise ValueError(
                    "--swiglu_product_tanh_last_n_layers must be 0 when SwiGLU is disabled."
                )
        stage3_kd_weight = float(getattr(self.config, "stage3_kd_weight", 0.0) or 0.0)
        stage3_teacher_checkpoint_path = getattr(self.config, "stage3_teacher_checkpoint_path", None)
        if stage3_kd_weight < 0.0:
            raise ValueError("--stage3_kd_weight must be non-negative.")
        if stage3_kd_weight > 0.0 and not stage3_teacher_checkpoint_path:
            raise ValueError("--stage3_teacher_checkpoint_path is required when --stage3_kd_weight > 0.")
        if float(getattr(self.config, "stage3_kd_temperature", 2.0) or 0.0) <= 0.0:
            raise ValueError("--stage3_kd_temperature must be positive.")
        stage3_l2sp_weight = float(getattr(self.config, "stage3_l2sp_weight", 0.0) or 0.0)
        stage3_anchor_pullback = float(getattr(self.config, "stage3_anchor_pullback", 0.0) or 0.0)
        if stage3_l2sp_weight < 0.0:
            raise ValueError("--stage3_l2sp_weight must be non-negative.")
        if stage3_anchor_pullback < 0.0 or stage3_anchor_pullback >= 1.0:
            raise ValueError("--stage3_anchor_pullback must be in [0, 1).")
        stage3_ema_decay = float(getattr(self.config, "stage3_ema_decay", 0.0) or 0.0)
        if stage3_ema_decay < 0.0 or stage3_ema_decay >= 1.0:
            raise ValueError("--stage3_ema_decay must be in [0, 1).")
        if int(getattr(self.config, "stage3_ema_start_step", 0) or 0) < 0:
            raise ValueError("--stage3_ema_start_step must be non-negative.")
        if (stage3_l2sp_weight > 0.0 or stage3_anchor_pullback > 0.0) and not getattr(
            self.config, "checkpoint_path", None
        ):
            raise ValueError("Stage-3 anchoring requires --checkpoint_path so the Stage-2 anchor is well defined.")
        label_smoothing = float(getattr(self.config, "label_smoothing", 0.0) or 0.0)
        if label_smoothing < 0.0 or label_smoothing >= 1.0:
            raise ValueError("--label_smoothing must be in [0, 1).")
        if bool(getattr(self.config, "layer_gate_enabled", False)):
            layer_gate_temperature = float(getattr(self.config, "layer_gate_temperature", 1.0) or 0.0)
            layer_gate_floor = float(getattr(self.config, "layer_gate_low_layer_floor", 0.0) or 0.0)
            if layer_gate_temperature <= 0.0:
                raise ValueError("--layer_gate_temperature must be positive.")
            if layer_gate_floor < 0.0 or layer_gate_floor >= 1.0:
                raise ValueError("--layer_gate_low_layer_floor must be in [0, 1).")
            layer_gate_repr_source = str(getattr(self.config, "layer_gate_repr_source", "deepest") or "deepest")
            if layer_gate_repr_source not in {"deepest", "multi_mean", "multi_delta"}:
                raise ValueError("--layer_gate_repr_source must be deepest, multi_mean, or multi_delta.")
            layer_gate_conf_source = str(
                getattr(self.config, "layer_gate_confidence_source", "deepest") or "deepest"
            )
            if layer_gate_conf_source not in {"none", "deepest", "all_layers"}:
                raise ValueError("--layer_gate_confidence_source must be none, deepest, or all_layers.")
            layer_gate_max_weight_target = float(getattr(self.config, "layer_gate_max_weight_target", 0.85) or 0.0)
            if layer_gate_max_weight_target <= 0.0 or layer_gate_max_weight_target > 1.0:
                raise ValueError("--layer_gate_max_weight_target must be in (0, 1].")
            for attr in (
                "layer_gate_entropy_reg_weight",
                "layer_gate_max_weight_reg_weight",
                "layer_gate_aux_ce_weight",
            ):
                if float(getattr(self.config, attr, 0.0) or 0.0) < 0.0:
                    raise ValueError(f"--{attr} must be non-negative.")
        balanced_ce_weight = float(getattr(self.config, "balanced_ce_weight", 0.0) or 0.0)
        balanced_ce_max_weight = float(getattr(self.config, "balanced_ce_max_weight", 3.0) or 0.0)
        if balanced_ce_weight < 0.0:
            raise ValueError("--balanced_ce_weight must be non-negative.")
        if balanced_ce_weight > 0.0 and balanced_ce_max_weight <= 0.0:
            raise ValueError("--balanced_ce_max_weight must be positive when --balanced_ce_weight > 0.")
        support_balanced_ce_weight = float(getattr(self.config, "support_balanced_ce_weight", 0.0) or 0.0)
        support_balanced_ce_max_weight = float(getattr(self.config, "support_balanced_ce_max_weight", 4.0) or 0.0)
        support_balanced_ce_threshold = float(
            getattr(self.config, "support_balanced_ce_minority_threshold", 0.35) or 0.0
        )
        if support_balanced_ce_weight < 0.0:
            raise ValueError("--support_balanced_ce_weight must be non-negative.")
        if support_balanced_ce_weight > 0.0 and support_balanced_ce_max_weight <= 0.0:
            raise ValueError("--support_balanced_ce_max_weight must be positive when enabled.")
        if support_balanced_ce_threshold < 0.0 or support_balanced_ce_threshold > 1.0:
            raise ValueError("--support_balanced_ce_minority_threshold must be in [0, 1].")
        if float(getattr(self.config, "support_balanced_ce_gamma", 0.5) or 0.0) < 0.0:
            raise ValueError("--support_balanced_ce_gamma must be non-negative.")
        if int(getattr(self.config, "support_balanced_ce_min_count", 2) or 0) < 1:
            raise ValueError("--support_balanced_ce_min_count must be positive.")
        support_prior_kl_weight = float(getattr(self.config, "support_prior_kl_weight", 0.0) or 0.0)
        if support_prior_kl_weight < 0.0:
            raise ValueError("--support_prior_kl_weight must be non-negative.")
        if int(getattr(self.config, "support_prior_kl_min_count", 2) or 0) < 1:
            raise ValueError("--support_prior_kl_min_count must be positive.")
        support_prior_kl_eps = float(getattr(self.config, "support_prior_kl_eps", 1e-6) or 0.0)
        if support_prior_kl_eps <= 0.0:
            raise ValueError("--support_prior_kl_eps must be positive.")
        support_prior_floor_weight = float(getattr(self.config, "support_prior_floor_weight", 0.0) or 0.0)
        if support_prior_floor_weight < 0.0:
            raise ValueError("--support_prior_floor_weight must be non-negative.")
        support_prior_floor_ratio = float(getattr(self.config, "support_prior_floor_ratio", 0.6) or 0.0)
        if support_prior_floor_ratio <= 0.0:
            raise ValueError("--support_prior_floor_ratio must be positive.")
        if int(getattr(self.config, "support_prior_floor_min_count", 2) or 0) < 1:
            raise ValueError("--support_prior_floor_min_count must be positive.")
        support_prior_floor_min_prior = float(
            getattr(self.config, "support_prior_floor_min_prior", 0.005) or 0.0
        )
        support_prior_floor_max_prior = float(
            getattr(self.config, "support_prior_floor_max_prior", 0.5) or 0.0
        )
        if support_prior_floor_min_prior < 0.0 or support_prior_floor_max_prior > 1.0:
            raise ValueError("--support_prior_floor_min_prior/max_prior must be in [0, 1].")
        if support_prior_floor_min_prior > support_prior_floor_max_prior:
            raise ValueError("--support_prior_floor_min_prior must be <= --support_prior_floor_max_prior.")
        support_margin_weight = float(getattr(self.config, "support_margin_weight", 0.0) or 0.0)
        if support_margin_weight < 0.0:
            raise ValueError("--support_margin_weight must be non-negative.")
        if float(getattr(self.config, "support_margin_value", 0.5) or 0.0) < 0.0:
            raise ValueError("--support_margin_value must be non-negative.")
        if float(getattr(self.config, "support_margin_gamma", 0.5) or 0.0) < 0.0:
            raise ValueError("--support_margin_gamma must be non-negative.")
        if float(getattr(self.config, "support_margin_max_weight", 6.0) or 0.0) <= 0.0:
            raise ValueError("--support_margin_max_weight must be positive.")
        if int(getattr(self.config, "support_margin_min_count", 2) or 0) < 1:
            raise ValueError("--support_margin_min_count must be positive.")
        support_margin_threshold = float(getattr(self.config, "support_margin_threshold", 0.5) or 0.0)
        if support_margin_threshold < 0.0 or support_margin_threshold > 1.0:
            raise ValueError("--support_margin_threshold must be in [0, 1].")
        if bool(getattr(self.config, "continual_bp_enabled", False)):
            if str(getattr(self.config, "continual_bp_target", "ffn") or "ffn").lower() != "ffn":
                raise ValueError("--continual_bp_target currently only supports 'ffn'.")
            if int(getattr(self.config, "continual_bp_start_step", 12000) or 0) < 0:
                raise ValueError("--continual_bp_start_step must be non-negative.")
            if int(getattr(self.config, "continual_bp_maturity_steps", 5000) or 0) < 0:
                raise ValueError("--continual_bp_maturity_steps must be non-negative.")
            if int(getattr(self.config, "continual_bp_replace_every", 200) or 0) <= 0:
                raise ValueError("--continual_bp_replace_every must be positive.")
            if float(getattr(self.config, "continual_bp_replacement_rate", 1e-6) or 0.0) < 0.0:
                raise ValueError("--continual_bp_replacement_rate must be non-negative.")
            if int(getattr(self.config, "continual_bp_max_replace_per_event", 1) or 0) < 1:
                raise ValueError("--continual_bp_max_replace_per_event must be positive.")
            utility_decay = float(getattr(self.config, "continual_bp_utility_decay", 0.99) or 0.0)
            if utility_decay < 0.0 or utility_decay >= 1.0:
                raise ValueError("--continual_bp_utility_decay must be in [0, 1).")
        if bool(getattr(self.config, "plasticity_proto_enabled", False)):
            if int(getattr(self.config, "plasticity_proto_start_step", 12000) or 0) < 0:
                raise ValueError("--plasticity_proto_start_step must be non-negative.")
            if int(getattr(self.config, "plasticity_proto_ramp_steps", 2000) or 0) < 0:
                raise ValueError("--plasticity_proto_ramp_steps must be non-negative.")
            if float(getattr(self.config, "plasticity_proto_tau", 0.07) or 0.0) <= 0.0:
                raise ValueError("--plasticity_proto_tau must be positive.")
            entropy_min = float(getattr(self.config, "plasticity_proto_entropy_min", 0.88) or 0.0)
            if entropy_min < 0.0 or entropy_min > 1.0:
                raise ValueError("--plasticity_proto_entropy_min must be in [0, 1].")
            top1_limit = float(getattr(self.config, "plasticity_proto_top1_limit", 0.08) or 0.0)
            if top1_limit <= 0.0 or top1_limit > 1.0:
                raise ValueError("--plasticity_proto_top1_limit must be in (0, 1].")
            for attr_name in (
                "plasticity_proto_entropy_weight",
                "plasticity_proto_top1_weight",
                "plasticity_proto_usage_weight",
            ):
                if float(getattr(self.config, attr_name, 0.0) or 0.0) < 0.0:
                    raise ValueError(f"--{attr_name} must be non-negative.")
            if int(getattr(self.config, "plasticity_proto_query_sample_rows", 128) or 0) < 0:
                raise ValueError("--plasticity_proto_query_sample_rows must be non-negative.")
            if int(getattr(self.config, "plasticity_proto_support_sample_rows", 512) or 0) < 0:
                raise ValueError("--plasticity_proto_support_sample_rows must be non-negative.")
        if bool(getattr(self.config, "plasticity_attn_enabled", False)):
            if int(getattr(self.config, "plasticity_attn_start_step", 12000) or 0) < 0:
                raise ValueError("--plasticity_attn_start_step must be non-negative.")
            if int(getattr(self.config, "plasticity_attn_ramp_steps", 2000) or 0) < 0:
                raise ValueError("--plasticity_attn_ramp_steps must be non-negative.")
            entropy_min = float(getattr(self.config, "plasticity_attn_entropy_min", 0.60) or 0.0)
            if entropy_min < 0.0 or entropy_min > 1.0:
                raise ValueError("--plasticity_attn_entropy_min must be in [0, 1].")
            top1_limit = float(getattr(self.config, "plasticity_attn_top1_limit", 0.22) or 0.0)
            if top1_limit <= 0.0 or top1_limit > 1.0:
                raise ValueError("--plasticity_attn_top1_limit must be in (0, 1].")
            for attr_name in (
                "plasticity_attn_entropy_weight",
                "plasticity_attn_top1_weight",
                "plasticity_attn_usage_weight",
            ):
                if float(getattr(self.config, attr_name, 0.0) or 0.0) < 0.0:
                    raise ValueError(f"--{attr_name} must be non-negative.")
            if int(getattr(self.config, "plasticity_attn_query_sample_rows", 128) or 0) < 0:
                raise ValueError("--plasticity_attn_query_sample_rows must be non-negative.")
            if int(getattr(self.config, "plasticity_attn_support_sample_rows", 512) or 0) < 0:
                raise ValueError("--plasticity_attn_support_sample_rows must be non-negative.")
        if bool(getattr(self.config, "qassmax_cap_enabled", False)):
            for attr_name in ("qassmax_cap_base_scale", "qassmax_cap_scale", "qassmax_cap_query_logit"):
                if float(getattr(self.config, attr_name, 0.0) or 0.0) <= 0.0:
                    raise ValueError(f"--{attr_name} must be positive when --qassmax_cap_enabled=True.")
        if bool(getattr(self.config, "late_icl_freeze_enabled", False)):
            if int(getattr(self.config, "late_icl_freeze_start_step", 18000) or 0) < 0:
                raise ValueError("--late_icl_freeze_start_step must be non-negative.")
        minority_kd_weight = float(getattr(self.config, "stage3_minority_kd_weight", 0.0) or 0.0)
        if minority_kd_weight < 0.0:
            raise ValueError("--stage3_minority_kd_weight must be non-negative.")
        if minority_kd_weight > 0.0 and not stage3_teacher_checkpoint_path:
            raise ValueError("--stage3_teacher_checkpoint_path is required when --stage3_minority_kd_weight > 0.")
        if float(getattr(self.config, "stage3_minority_kd_gamma", 0.5) or 0.0) < 0.0:
            raise ValueError("--stage3_minority_kd_gamma must be non-negative.")
        if float(getattr(self.config, "stage3_minority_kd_max_weight", 4.0) or 0.0) <= 1.0:
            raise ValueError("--stage3_minority_kd_max_weight must be greater than 1.")
        minority_kd_conf = float(getattr(self.config, "stage3_minority_kd_min_confidence", 0.35) or 0.0)
        if minority_kd_conf < 0.0 or minority_kd_conf > 1.0:
            raise ValueError("--stage3_minority_kd_min_confidence must be in [0, 1].")
        if int(getattr(self.config, "stage3_minority_kd_min_count", 2) or 0) < 1:
            raise ValueError("--stage3_minority_kd_min_count must be positive.")
        minority_kd_threshold = float(getattr(self.config, "stage3_minority_kd_threshold", 0.35) or 0.0)
        if minority_kd_threshold < 0.0 or minority_kd_threshold > 1.0:
            raise ValueError("--stage3_minority_kd_threshold must be in [0, 1].")
        support_classwise_kd_weight = float(
            getattr(self.config, "stage3_support_classwise_kd_weight", 0.0) or 0.0
        )
        if support_classwise_kd_weight < 0.0:
            raise ValueError("--stage3_support_classwise_kd_weight must be non-negative.")
        if support_classwise_kd_weight > 0.0 and not stage3_teacher_checkpoint_path:
            raise ValueError(
                "--stage3_teacher_checkpoint_path is required when --stage3_support_classwise_kd_weight > 0."
            )
        if float(getattr(self.config, "stage3_support_classwise_kd_gamma", 0.5) or 0.0) < 0.0:
            raise ValueError("--stage3_support_classwise_kd_gamma must be non-negative.")
        if float(getattr(self.config, "stage3_support_classwise_kd_max_weight", 6.0) or 0.0) <= 1.0:
            raise ValueError("--stage3_support_classwise_kd_max_weight must be greater than 1.")
        if int(getattr(self.config, "stage3_support_classwise_kd_min_count", 2) or 0) < 1:
            raise ValueError("--stage3_support_classwise_kd_min_count must be positive.")
        support_classwise_min_prior = float(
            getattr(self.config, "stage3_support_classwise_kd_min_prior", 0.005) or 0.0
        )
        support_classwise_max_prior = float(
            getattr(self.config, "stage3_support_classwise_kd_max_prior", 0.5) or 0.0
        )
        if support_classwise_min_prior < 0.0 or support_classwise_max_prior > 1.0:
            raise ValueError("--stage3_support_classwise_kd_min_prior/max_prior must be in [0, 1].")
        if support_classwise_min_prior > support_classwise_max_prior:
            raise ValueError(
                "--stage3_support_classwise_kd_min_prior must be <= --stage3_support_classwise_kd_max_prior."
            )
        firewall_kd_weight = float(getattr(self.config, "firewall_kd_weight", 0.0) or 0.0)
        firewall_kd_ce_weight = float(getattr(self.config, "firewall_kd_ce_weight", 0.0) or 0.0)
        if firewall_kd_weight < 0.0:
            raise ValueError("--firewall_kd_weight must be non-negative.")
        if firewall_kd_ce_weight < 0.0:
            raise ValueError("--firewall_kd_ce_weight must be non-negative.")
        if firewall_kd_weight > 0.0 and not stage3_teacher_checkpoint_path:
            raise ValueError("--stage3_teacher_checkpoint_path is required when --firewall_kd_weight > 0.")
        if (firewall_kd_weight > 0.0 or firewall_kd_ce_weight > 0.0) and not getattr(
            self.config, "firewall_kd_dataset_dir", None
        ):
            raise ValueError("--firewall_kd_dataset_dir is required when firewall KD/CE is enabled.")
        if float(getattr(self.config, "firewall_kd_temperature", 2.0) or 0.0) <= 0.0:
            raise ValueError("--firewall_kd_temperature must be positive.")
        if int(getattr(self.config, "firewall_kd_interval", 1) or 0) <= 0:
            raise ValueError("--firewall_kd_interval must be positive.")
        if int(getattr(self.config, "firewall_kd_support_rows", 0) or 0) <= 0:
            raise ValueError("--firewall_kd_support_rows must be positive.")
        if int(getattr(self.config, "firewall_kd_query_rows", 0) or 0) <= 0:
            raise ValueError("--firewall_kd_query_rows must be positive.")
        firewall_kd_prob = float(getattr(self.config, "firewall_kd_prob", 1.0) or 0.0)
        if firewall_kd_prob < 0.0 or firewall_kd_prob > 1.0:
            raise ValueError("--firewall_kd_prob must be in [0, 1].")

    def stage3_kd_enabled(self) -> bool:
        return (
            float(getattr(self.config, "stage3_kd_weight", 0.0) or 0.0) > 0.0
            and getattr(self.config, "stage3_teacher_checkpoint_path", None) is not None
        )

    def firewall_kd_enabled(self) -> bool:
        return bool(
            getattr(self.config, "firewall_kd_dataset_dir", None)
            and (
                float(getattr(self.config, "firewall_kd_weight", 0.0) or 0.0) > 0.0
                or float(getattr(self.config, "firewall_kd_ce_weight", 0.0) or 0.0) > 0.0
            )
        )

    def stage3_teacher_required(self) -> bool:
        return (
            self.stage3_kd_enabled()
            or float(getattr(self.config, "stage3_minority_kd_weight", 0.0) or 0.0) > 0.0
            or float(getattr(self.config, "stage3_support_classwise_kd_weight", 0.0) or 0.0) > 0.0
            or (
                self.firewall_kd_enabled()
                and float(getattr(self.config, "firewall_kd_weight", 0.0) or 0.0) > 0.0
            )
        )

    def stage3_anchor_enabled(self) -> bool:
        return (
            float(getattr(self.config, "stage3_l2sp_weight", 0.0) or 0.0) > 0.0
            or float(getattr(self.config, "stage3_anchor_pullback", 0.0) or 0.0) > 0.0
        )

    def stage3_ema_enabled(self) -> bool:
        return float(getattr(self.config, "stage3_ema_decay", 0.0) or 0.0) > 0.0

    def configure_optimizer(self):
        """Configure optimizer and scheduler."""

        named_trainable = [
            (name, param) for name, param in self.raw_model.named_parameters() if param.requires_grad
        ]
        trainable_params = [param for _, param in named_trainable]
        self._optimizer_param_names = {id(param): name for name, param in named_trainable}
        if self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(params=trainable_params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "muon":
            param_groups = []
            grouped_by_lr = {}
            for name, param in named_trainable:
                lr, weight_decay = resolve_muon_hparams(self.config, name, param)
                down_bootstrap = is_swiglu_down_matrix(self.config, name)
                if bool(getattr(self.config, "muon_group_by_lr", False)):
                    # Keep down matrices isolated even when LR grouping is on;
                    # the temporary bootstrap must never affect unrelated
                    # matrices that happen to have the same base LR and WD.
                    bucket = grouped_by_lr.setdefault(
                        (lr, weight_decay, down_bootstrap),
                        {"params": [], "param_names": []},
                    )
                    bucket["params"].append(param)
                    bucket["param_names"].append(name)
                else:
                    param_groups.append(
                        {
                            "params": [param], "lr": lr, "weight_decay": weight_decay,
                            "param_names": [name], "swiglu_down_bootstrap": down_bootstrap,
                        }
                    )
            if grouped_by_lr:
                param_groups = [
                    {
                        "params": bucket["params"],
                        "param_names": bucket["param_names"],
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "swiglu_down_bootstrap": down_bootstrap,
                    }
                    for (lr, weight_decay, down_bootstrap), bucket in grouped_by_lr.items()
                ]
                if self.master_process:
                    print(
                        "Muon grouped parameters by learning rate and weight decay: "
                        f"{len(trainable_params)} parameters into {len(param_groups)} optimizer groups."
                    )
            self.optimizer = Muon(
                param_groups,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                momentum=self.config.muon_momentum,
                ns_steps=self.config.muon_ns_steps,
                cautious_weight_decay=self.config.cautious_weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)

    def reapply_explicit_optimizer_overrides(self) -> None:
        """Restore explicit WD and bootstrap metadata after optimizer restore."""
        value_override = float(getattr(self.config, "swiglu_value_proj_weight_decay", -1.0))
        if self.config.optimizer != "muon":
            return
        names_by_id = getattr(self, "_optimizer_param_names", {})
        updated_values = 0
        updated_down = 0
        for group in self.optimizer.param_groups:
            names = [names_by_id.get(id(param), "") for param in group["params"]]
            expected = {
                resolve_muon_hparams(self.config, name, param)[1]
                for name, param in zip(names, group["params"])
            }
            if len(expected) != 1:
                raise RuntimeError(
                    "optimizer group mixes parameters requiring different weight decay; "
                    "disable incompatible grouping or rebuild the optimizer"
                )
            group["weight_decay"] = expected.pop()
            group["param_names"] = names
            down_flags = {is_swiglu_down_matrix(self.config, name) for name in names}
            if len(down_flags) != 1:
                raise RuntimeError(
                    "optimizer group mixes SwiGLU down matrices with unrelated parameters; "
                    "rebuild the optimizer with bootstrap-aware grouping"
                )
            group["swiglu_down_bootstrap"] = down_flags.pop()
            updated_values += sum(".swiglu_value_proj." in name for name in names)
            updated_down += sum(is_swiglu_down_matrix(self.config, name) for name in names)
        expected_values = sum(
            ".swiglu_value_proj." in name for name in self._optimizer_param_names.values()
        )
        if value_override >= 0.0 and updated_values != expected_values:
            raise RuntimeError(
                f"value-projection WD override coverage mismatch: {updated_values}/{expected_values}"
            )
        expected_down = sum(
            is_swiglu_down_matrix(self.config, name)
            for name in self._optimizer_param_names.values()
        )
        if updated_down != expected_down:
            raise RuntimeError(f"down-bootstrap group coverage mismatch: {updated_down}/{expected_down}")

    def activate_swiglu_down_bootstrap(self) -> list[tuple[dict, float, float]]:
        """Temporarily boost only SwiGLU down LR for the current optimizer step.

        The scheduler always owns the nominal LR.  This method is invoked
        immediately before ``optimizer.step`` and returns exact values for a
        ``finally`` restoration, so scheduler base LRs and checkpoint state do
        not inherit the multiplier.  Dividing WD by the multiplier preserves
        the nominal ``lr * weight_decay`` shrink coefficient.
        """
        multiplier = float(getattr(self.config, "swiglu_down_lr_bootstrap_multiplier", 1.0))
        steps = int(getattr(self.config, "swiglu_down_lr_bootstrap_steps", 0))
        if multiplier == 1.0 or steps == 0 or int(self.curr_step) >= steps:
            return []
        restore = []
        for group in self.optimizer.param_groups:
            if not bool(group.get("swiglu_down_bootstrap", False)):
                continue
            nominal_lr = float(group["lr"])
            nominal_wd = float(group["weight_decay"])
            restore.append((group, nominal_lr, nominal_wd))
            group["lr"] = nominal_lr * multiplier
            group["weight_decay"] = nominal_wd / multiplier
        expected_down = sum(
            is_swiglu_down_matrix(self.config, name)
            for name in getattr(self, "_optimizer_param_names", {}).values()
        )
        covered_down = sum(len(group["params"]) for group, _, _ in restore)
        if covered_down != expected_down:
            for group, nominal_lr, nominal_wd in restore:
                group["lr"] = nominal_lr
                group["weight_decay"] = nominal_wd
            raise RuntimeError(f"active down-bootstrap coverage mismatch: {covered_down}/{expected_down}")
        return restore

    @staticmethod
    def restore_swiglu_down_bootstrap(restore: list[tuple[dict, float, float]]) -> None:
        for group, nominal_lr, nominal_wd in restore:
            group["lr"] = nominal_lr
            group["weight_decay"] = nominal_wd

    def paired_training_contract(self) -> dict:
        """Fields that must not silently drift across an optimizer resume."""
        model_config_payload = json.dumps(self.model_config, sort_keys=True, default=str).encode("utf-8")
        prior_config = {
            key: value
            for key, value in vars(self.config).items()
            if key == "prior_type"
            or key.startswith(("tabiclv2_", "hybrid178_", "talent_", "protected_batch_"))
            or key
            in {
                "batch_size",
                "batch_size_per_gp",
                "micro_batch_size",
                "min_features",
                "max_features",
                "max_classes",
                "min_seq_len",
                "max_seq_len",
                "min_train_size",
                "max_train_size",
                "replay_small",
                "log_seq_len",
                "log_n_features",
                "seq_len_per_gp",
                "prior_n_jobs",
                "prior_num_threads_per_generate",
            }
        }
        prior_config["t25_graph_config"] = vars(PriorConfig.from_args(self.config))
        prior_config["regression_method"] = self.config.regression_method
        prior_config["num_quantiles"] = self.config.num_quantiles
        prior_config["safe_tail"] = True
        prior_config["effective_generator_n_jobs"] = 1
        with open(self.config.regression_target_profile, "rb") as profile_handle:
            prior_config["target_profile_sha256"] = hashlib.sha256(profile_handle.read()).hexdigest()
        prior_config_payload = json.dumps(prior_config, sort_keys=True, default=str).encode("utf-8")
        optimizer_topology = []
        for name, parameter in self.raw_model.named_parameters():
            if not parameter.requires_grad:
                continue
            expected_lr, expected_wd = resolve_muon_hparams(self.config, name, parameter) if (
                str(self.config.optimizer) == "muon"
            ) else (float(self.config.lr), float(self.config.weight_decay))
            optimizer_topology.append(
                [name, list(parameter.shape), str(parameter.dtype), expected_lr, expected_wd]
            )
        optimizer_payload = json.dumps(optimizer_topology, separators=(",", ":")).encode("utf-8")
        runtime_contract = {
            key: os.environ.get(key, "UNSET")
            for key in (
                "SYNTHETIC96_RW_DGP_ENABLED",
                "SYNTHETIC96_RW_DGP_ARM",
                "SYNTHETIC96_RW_DGP_EXPERIMENT_SEED",
                # These names must match rw_dgp_series_runtime_patch.py
                # exactly.  Recording invented aliases would let the actual
                # frozen RW-Sample artifacts drift across a resume.
                "SYNTHETIC96_RWSAMPLE178_ARTIFACT",
                "SYNTHETIC96_RWSAMPLE178_SHA256",
                "SYNTHETIC96_RWSAMPLEMETA_ARTIFACT",
                "SYNTHETIC96_RWSAMPLEMETA_SHA256",
                "CROSS_TABLE_ENABLED",
                "CROSS_TABLE_ARM",
                "CROSS_TABLE_MAX_NUM_DOMAINS",
                "CROSS_TABLE_EXPERIMENT_SEED",
            )
        }
        return {
            "optimizer": str(self.config.optimizer),
            "global_lr": float(self.config.lr),
            "weight_decay": float(self.config.weight_decay),
            "muon_momentum": float(getattr(self.config, "muon_momentum", 0.95)),
            "muon_ns_steps": int(getattr(self.config, "muon_ns_steps", 5)),
            "muon_group_by_lr": bool(getattr(self.config, "muon_group_by_lr", False)),
            "cautious_weight_decay": bool(getattr(self.config, "cautious_weight_decay", False)),
            "scheduler": str(self.config.scheduler),
            "warmup_steps": int(self.config.warmup_steps),
            "warmup_proportion": float(self.config.warmup_proportion),
            "max_steps": int(self.config.max_steps),
            "lr_floor": float(getattr(self.config, "lr_floor", 0.0)),
            "fast_cosine_scheduler": bool(getattr(self.config, "fast_cosine_scheduler", False)),
            "cosine_num_cycles": int(getattr(self.config, "cosine_num_cycles", 1)),
            "cosine_amplitude_decay": float(getattr(self.config, "cosine_amplitude_decay", 1.0)),
            "cosine_lr_end": float(getattr(self.config, "cosine_lr_end", 0.0)),
            "gradient_clipping": float(getattr(self.config, "gradient_clipping", 0.0)),
            "skip_nonfinite_batches": bool(getattr(self.config, "skip_nonfinite_batches", True)),
            "abort_on_nonfinite_batch": bool(getattr(self.config, "abort_on_nonfinite_batch", False)),
            "error_if_nonfinite_grad": bool(getattr(self.config, "error_if_nonfinite_grad", False)),
            "label_smoothing": float(getattr(self.config, "label_smoothing", 0.0)),
            "amp": bool(getattr(self.config, "amp", False)),
            "grad_scaler": bool(getattr(self.config, "grad_scaler", True)),
            "dtype": str(getattr(self.config, "dtype", "float32")),
            "world_size": int(getattr(self, "ddp_world_size", 1)),
            "rank_batch_size": int(self.config.batch_size),
            "micro_batch_size": int(getattr(self.config, "micro_batch_size", self.config.batch_size)),
            "prior_num_workers_requested": int(getattr(self.config, "prior_num_workers", 0)),
            "prior_num_workers_resolved": int(self.resolve_prior_num_workers()),
            "prior_cache_enabled": bool(getattr(self.config, "prior_cache_enabled", False)),
            "np_seed": int(self.config.np_seed),
            "torch_seed": int(self.config.torch_seed),
            "prior_loader_seed": int(getattr(self.config, "prior_loader_seed", 42)),
            "swiglu_enabled": bool(getattr(self.config, "swiglu_enabled", False)),
            "swiglu_muon_lr_multiplier": float(getattr(self.config, "swiglu_muon_lr_multiplier", 1.0)),
            "swiglu_value_proj_weight_decay": float(
                getattr(self.config, "swiglu_value_proj_weight_decay", -1.0)
            ),
            "swiglu_down_lr_bootstrap_multiplier": float(
                getattr(self.config, "swiglu_down_lr_bootstrap_multiplier", 1.0)
            ),
            "swiglu_down_lr_bootstrap_steps": int(
                getattr(self.config, "swiglu_down_lr_bootstrap_steps", 0)
            ),
            "swiglu_output_scale": float(getattr(self.config, "swiglu_output_scale", 1.0)),
            "swiglu_product_tanh_rms_multiple": float(
                getattr(self.config, "swiglu_product_tanh_rms_multiple", 0.0)
            ),
            "swiglu_product_tanh_last_n_layers": int(
                getattr(self.config, "swiglu_product_tanh_last_n_layers", 0)
            ),
            "swiglu_hidden_dim": int(getattr(self.config, "swiglu_hidden_dim", 1024)),
            "swiglu_init_seed_base": int(getattr(self.config, "swiglu_init_seed_base", 2026082601)),
            "cr2_shared_refinement_enabled": bool(
                getattr(self.config, "cr2_shared_refinement_enabled", False)
            ),
            "qk_pds_attention_enabled": bool(
                getattr(self.config, "qk_pds_attention_enabled", False)
            ),
            "cls8_pooled_enabled": bool(getattr(self.config, "cls8_pooled_enabled", False)),
            "shared_depth_icl_enabled": bool(
                getattr(self.config, "shared_depth_icl_enabled", False)
            ),
            "shared_depth_icl_rho": float(
                getattr(self.config, "shared_depth_icl_rho", 1.0)
            ),
            "shared_depth_icl_dataset_conditioned": bool(
                getattr(self.config, "shared_depth_icl_dataset_conditioned", False)
            ),
            "shared_depth_icl_num_passes": int(
                getattr(self.config, "shared_depth_icl_num_passes", 2)
            ),
            "cls8_width_enabled": bool(getattr(self.config, "cls8_width_enabled", False)),
            "model_config_sha256": hashlib.sha256(model_config_payload).hexdigest(),
            "prior_config_sha256": hashlib.sha256(prior_config_payload).hexdigest(),
            "optimizer_topology_sha256": hashlib.sha256(optimizer_payload).hexdigest(),
            "runtime_contract": runtime_contract,
            "resume_data_policy": (
                "restart_not_exact"
                if bool(getattr(self.config, "allow_nonexact_prior_resume", False))
                else "fresh_start_required"
            ),
            "training_stage_manifest_sha256": str(
                os.environ.get("TRAINING_STAGE_MANIFEST_SHA256", "UNSET")
            ),
            "training_stage_manifest_path": str(
                os.environ.get("TRAINING_STAGE_MANIFEST_PATH", "UNSET")
            ),
        }

    def validate_resume_training_contract(self, checkpoint: dict) -> None:
        if bool(getattr(self.config, "only_load_model", False)):
            return
        saved = checkpoint.get("training_contract")
        current = self.paired_training_contract()
        if saved is None:
            if bool(getattr(self.config, "strict_resume_training_contract", True)):
                raise RuntimeError(
                    "optimizer resume checkpoint has no training_contract; use --only_load_model True "
                    "for a deliberate weights-only fork or disable strict validation explicitly"
                )
            return
        mismatches = {
            key: {"checkpoint": saved.get(key), "current": value}
            for key, value in current.items()
            if saved.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"resume training contract mismatch: {json.dumps(mismatches, sort_keys=True)}")

    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""

        self.amp = self.config.amp and "cuda" in self.config.device
        grad_scaler_enabled = bool(getattr(self.config, "grad_scaler", True))
        self.scaler = torch.GradScaler("cuda", enabled=self.amp and grad_scaler_enabled)
        if self.amp:
            if self.master_process:
                print(f"Automatic Mixed Precision is enabled. GradScaler enabled={self.scaler.is_enabled()}.")
            self.amp_ctx = torch.autocast(
                device_type="cuda", dtype=torch.float16 if self.config.dtype == "float16" else torch.float32
            )
        else:
            self.amp_ctx = nullcontext()

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`

        Only considers files with the .ckpt extension (PyTorch checkpoint files).
        """
        ckpt_dir = self.config.checkpoint_dir

        if not os.path.isdir(ckpt_dir):
            return None

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]

        if not checkpoints:
            return None

        # Sort the checkpoint files by step number and get the latest
        try:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))[-1]
            checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
            return checkpoint_path
        except Exception as e:
            print(f"Error parsing checkpoint filenames: {e}")
            return None

    def load_checkpoint(self):
        """Load model and training state from checkpoint.

        First checks if `checkpoint_path` is directly specified. If not, attempts to find
        the latest checkpoint in the checkpoint directory.
        """

        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)

        # Load model state
        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model state")

        self.raw_model.load_state_dict(checkpoint["state_dict"])

        # Optionally load optimizer and scheduler state
        if bool(getattr(self.config, "only_load_model", False)):
            print("Only loading model weights")
        else:
            self.validate_resume_training_contract(checkpoint)
            if self.config.prior_dir is None and not bool(
                getattr(self.config, "allow_nonexact_prior_resume", False)
            ):
                raise RuntimeError(
                    "Exact optimizer resume is unavailable for the online stochastic prior because worker RNG, "
                    "runtime counters and async-cache contents are not checkpointed. Start a fresh paired run, "
                    "use --only_load_model True for a deliberate weights-only fork, or explicitly set "
                    "--allow_nonexact_prior_resume True."
                )
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.reapply_explicit_optimizer_overrides()
            scaler_state = checkpoint.get("scaler_state")
            if scaler_state is not None:
                self.scaler.load_state_dict(scaler_state)
            elif self.scaler.is_enabled() and bool(getattr(self.config, "strict_resume_training_contract", True)):
                raise RuntimeError("AMP optimizer resume checkpoint has no scaler_state")
            self.curr_step = checkpoint["curr_step"]
            print(f"Resuming training at step {self.curr_step}")

    def configure_plasticity_reference(self):
        """Load a frozen reference model for representation CKA anchoring."""

        cka_weight = float(getattr(self.config, "plasticity_cka_weight", 0.0) or 0.0)
        if cka_weight <= 0.0:
            self.plasticity_reference_model = None
            return

        checkpoint_path = str(getattr(self.config, "plasticity_cka_reference_checkpoint", "") or "")
        if not checkpoint_path:
            raise ValueError("--plasticity_cka_reference_checkpoint is required when --plasticity_cka_weight > 0.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Plasticity CKA reference checkpoint does not exist: {checkpoint_path}")

        if self.master_process:
            print(f"Loading frozen plasticity CKA reference from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)
        if "state_dict" not in checkpoint:
            raise ValueError("Plasticity CKA reference checkpoint does not contain model state")
        reference_config = dict(checkpoint.get("config") or self.model_config)
        tabicl_params = inspect.signature(TabICL).parameters
        reference_config = {key: value for key, value in reference_config.items() if key in tabicl_params}
        reference_model = TabICL(**reference_config)
        reference_model.to(device=self.config.device)
        reference_model.load_state_dict(checkpoint["state_dict"])
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        self.plasticity_reference_model = reference_model
        self.plasticity_reference_max_classes = int(getattr(reference_model, "max_classes", 0) or 0)
        if self.master_process and self.plasticity_reference_max_classes != int(self.config.max_classes):
            print(
                "Plasticity CKA reference max_classes="
                f"{self.plasticity_reference_max_classes}; current max_classes={self.config.max_classes}. "
                "CKA anchor will be skipped for micro-batches outside the reference label range."
            )

    @staticmethod
    def _parse_layer_indices(value: str | int | None) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        layers = []
        for item in re.split(r"[,;:+\s]+", str(value)):
            if not item:
                continue
            layers.append(int(item))
        return layers

    def late_icl_freeze_enabled(self) -> bool:
        return bool(getattr(self.config, "late_icl_freeze_enabled", False))

    def late_icl_freeze_active(self) -> bool:
        if not self.late_icl_freeze_enabled():
            return False
        start_step = int(getattr(self.config, "late_icl_freeze_start_step", 18000) or 0)
        return int(self.curr_step) >= start_step

    def late_icl_freeze_layer_indices(self) -> list[int]:
        return sorted(set(self._parse_layer_indices(getattr(self.config, "late_icl_freeze_layers", "7,8,9,10"))))

    def apply_qassmax_caps(self, model: nn.Module) -> None:
        if not bool(getattr(self.config, "qassmax_cap_enabled", False)):
            return
        blocks = getattr(getattr(model, "icl_predictor", None), "tf_icl", None)
        blocks = getattr(blocks, "blocks", None)
        if blocks is None:
            raise ValueError("Could not locate model.icl_predictor.tf_icl.blocks for QASSMax cap.")

        layers = sorted(set(self._parse_layer_indices(getattr(self.config, "qassmax_cap_layers", "7,8,9,10"))))
        base_scale = float(getattr(self.config, "qassmax_cap_base_scale", 16.0) or 16.0)
        scale = float(getattr(self.config, "qassmax_cap_scale", 16.0) or 16.0)
        query_logit = float(getattr(self.config, "qassmax_cap_query_logit", 8.0) or 8.0)
        capped_layers: list[int] = []
        for layer_idx in layers:
            if layer_idx < 0 or layer_idx >= len(blocks):
                continue
            ssmax_layer = getattr(getattr(blocks[layer_idx], "attn", None), "ssmax_layer", None)
            if ssmax_layer is None:
                continue
            if not all(
                hasattr(ssmax_layer, attr)
                for attr in ("max_abs_base_scale", "max_abs_scale", "max_abs_query_logit")
            ):
                continue
            ssmax_layer.max_abs_base_scale = base_scale
            ssmax_layer.max_abs_scale = scale
            ssmax_layer.max_abs_query_logit = query_logit
            capped_layers.append(layer_idx)
        if self.master_process:
            print(
                "QASSMax cap applied: "
                f"layers={capped_layers} "
                f"base_scale={base_scale} scale={scale} query_logit={query_logit}"
            )

    def late_icl_freeze_parameters(self) -> list[Tensor]:
        if self._late_icl_freeze_params is not None:
            return self._late_icl_freeze_params
        if not self.late_icl_freeze_enabled():
            self._late_icl_freeze_params = []
            self._late_icl_freeze_param_count = 0
            return self._late_icl_freeze_params

        blocks = getattr(getattr(self.raw_model, "icl_predictor", None), "tf_icl", None)
        blocks = getattr(blocks, "blocks", None)
        if blocks is None:
            raise ValueError("Could not locate raw_model.icl_predictor.tf_icl.blocks for late ICL freeze.")

        params: list[Tensor] = []
        for layer_idx in self.late_icl_freeze_layer_indices():
            if layer_idx < 0 or layer_idx >= len(blocks):
                raise ValueError(f"late ICL freeze layer {layer_idx} is outside available ICL blocks [0, {len(blocks)-1}].")
            params.extend(list(blocks[layer_idx].parameters()))

        self._late_icl_freeze_params = params
        self._late_icl_freeze_param_count = sum(int(param.numel()) for param in params)
        if self.master_process:
            print(
                "Late ICL update-freeze configured: "
                f"layers={self.late_icl_freeze_layer_indices()} "
                f"start={int(getattr(self.config, 'late_icl_freeze_start_step', 18000) or 0)} "
                f"params={self._late_icl_freeze_param_count}"
            )
        return self._late_icl_freeze_params

    @torch.no_grad()
    def apply_late_icl_freeze_to_grads(self) -> dict[str, float]:
        if not self.late_icl_freeze_active():
            return {}
        params = self.late_icl_freeze_parameters()
        grad_norm_sq = 0.0
        grad_tensors = 0
        for param in params:
            if param.grad is None:
                continue
            grad_norm_sq += float(param.grad.detach().float().pow(2).sum().item())
            grad_tensors += 1
            param.grad = None
        return {
            "late_icl_freeze_active": 1.0,
            "late_icl_freeze_layers": float(len(self.late_icl_freeze_layer_indices())),
            "late_icl_freeze_params": float(self._late_icl_freeze_param_count),
            "late_icl_freeze_grad_tensors": float(grad_tensors),
            "late_icl_freeze_grad_norm": math.sqrt(max(0.0, grad_norm_sq)),
        }

    def continual_bp_enabled(self) -> bool:
        return bool(getattr(self.config, "continual_bp_enabled", False))

    def configure_continual_bp(self):
        """Set up conservative continual BP hooks for selected ICL FFN channels."""

        if not self.continual_bp_enabled():
            return
        if bool(getattr(self.config, "model_compile", False)) or str(
            getattr(self.config, "model_compile_parts", "") or ""
        ).strip():
            raise ValueError("continual BP is not supported together with torch.compile/model_compile_parts.")

        layers = sorted(set(self._parse_layer_indices(getattr(self.config, "continual_bp_layers", "8,9,10"))))
        if not layers:
            raise ValueError("--continual_bp_layers must contain at least one layer when continual BP is enabled.")
        blocks = getattr(getattr(self.raw_model, "icl_predictor", None), "tf_icl", None)
        blocks = getattr(blocks, "blocks", None)
        if blocks is None:
            raise ValueError("Could not locate raw_model.icl_predictor.tf_icl.blocks for continual BP.")

        self.continual_bp_layers = []
        for layer_idx in layers:
            if layer_idx < 0 or layer_idx >= len(blocks):
                raise ValueError(f"continual BP layer {layer_idx} is outside available ICL blocks [0, {len(blocks)-1}].")
            block = blocks[layer_idx]
            if not hasattr(block, "linear1") or not hasattr(block, "linear2"):
                raise ValueError(f"ICL block {layer_idx} does not expose linear1/linear2 FFN modules.")
            hidden_dim = int(block.linear1.weight.shape[0])
            device = block.linear1.weight.device
            self.continual_bp_layers.append(layer_idx)
            self.continual_bp_modules[layer_idx] = block
            self.continual_bp_utility[layer_idx] = torch.zeros(hidden_dim, device=device, dtype=torch.float32)
            self.continual_bp_age[layer_idx] = torch.full(
                (hidden_dim,),
                int(max(0, self.curr_step)),
                device=device,
                dtype=torch.long,
            )
            self.continual_bp_credit[layer_idx] = 0.0
            self.continual_bp_activation_sum[layer_idx] = torch.zeros(hidden_dim, device=device, dtype=torch.float32)
            self.continual_bp_activation_count[layer_idx] = torch.zeros((), device=device, dtype=torch.float32)
            self.continual_bp_hooks.append(
                block.linear1.register_forward_hook(self._make_continual_bp_activation_hook(layer_idx))
            )

        if self.master_process:
            print(
                "Continual BP enabled: "
                f"layers={self.continual_bp_layers} target=ffn "
                f"start={int(getattr(self.config, 'continual_bp_start_step', 12000) or 0)} "
                f"maturity={int(getattr(self.config, 'continual_bp_maturity_steps', 5000) or 0)} "
                f"replace_every={int(getattr(self.config, 'continual_bp_replace_every', 200) or 200)} "
                f"rate={float(getattr(self.config, 'continual_bp_replacement_rate', 1e-6) or 0.0)}"
            )

    def _make_continual_bp_activation_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if not self.continual_bp_enabled():
                return
            if not torch.is_tensor(output):
                return
            with torch.no_grad():
                flat = output.detach().reshape(-1, output.shape[-1]).float()
                if flat.numel() == 0:
                    return
                self.continual_bp_activation_sum[layer_idx].add_(flat.abs().mean(dim=0))
                self.continual_bp_activation_count[layer_idx].add_(1.0)

        return hook

    def _zero_optimizer_state_slice(self, param: Tensor, index: int, dim: int | None = None):
        state = self.optimizer.state.get(param, None)
        if not state:
            return
        for value in state.values():
            if not torch.is_tensor(value) or value.shape != param.shape:
                continue
            if dim is None:
                value[index].zero_()
            elif dim == 0:
                value[index, :].zero_()
            elif dim == 1:
                value[:, index].zero_()

    def _continual_bp_generator(self, device: torch.device, layer_idx: int, channel_idx: int) -> torch.Generator:
        try:
            generator = torch.Generator(device=device)
        except TypeError:
            generator = torch.Generator()
        seed = (
            int(getattr(self.config, "torch_seed", 42))
            + 1000003 * int(self.curr_step + 1)
            + 1009 * int(layer_idx + 1)
            + 9176 * int(channel_idx + 1)
        ) % (2**63 - 1)
        generator.manual_seed(seed)
        return generator

    @torch.no_grad()
    def _reset_continual_bp_channel(self, layer_idx: int, channel_idx: int):
        block = self.continual_bp_modules[layer_idx]
        linear1 = block.linear1
        linear2 = block.linear2
        device = linear1.weight.device
        generator = self._continual_bp_generator(device, layer_idx, channel_idx)
        fan_in = int(linear1.weight.shape[1])
        bound = 1.0 / math.sqrt(max(1, fan_in))

        new_row = torch.empty_like(linear1.weight[channel_idx])
        new_row.uniform_(-bound, bound, generator=generator)
        linear1.weight[channel_idx].copy_(new_row)
        self._zero_optimizer_state_slice(linear1.weight, channel_idx, dim=0)

        if linear1.bias is not None:
            new_bias = torch.empty_like(linear1.bias[channel_idx])
            new_bias.uniform_(-bound, bound, generator=generator)
            linear1.bias[channel_idx].copy_(new_bias)
            self._zero_optimizer_state_slice(linear1.bias, channel_idx, dim=None)

        if bool(getattr(self.config, "continual_bp_reset_outgoing_zero", True)):
            linear2.weight[:, channel_idx].zero_()
            self._zero_optimizer_state_slice(linear2.weight, channel_idx, dim=1)

        self.continual_bp_utility[layer_idx][channel_idx] = 0.0
        self.continual_bp_age[layer_idx][channel_idx] = 0

    @torch.no_grad()
    def continual_bp_step(self) -> dict[str, float]:
        if not self.continual_bp_enabled() or not self.continual_bp_layers:
            return {}

        decay = float(getattr(self.config, "continual_bp_utility_decay", 0.99) or 0.99)
        start_step = int(getattr(self.config, "continual_bp_start_step", 12000) or 0)
        maturity_steps = int(getattr(self.config, "continual_bp_maturity_steps", 5000) or 0)
        replace_every = int(getattr(self.config, "continual_bp_replace_every", 200) or 200)
        replacement_rate = float(getattr(self.config, "continual_bp_replacement_rate", 1e-6) or 0.0)
        max_replace_per_event = int(getattr(self.config, "continual_bp_max_replace_per_event", 1) or 1)
        event_step = self.curr_step >= start_step and (self.curr_step - start_step) % replace_every == 0

        total_replaced = 0
        total_eligible = 0
        utility_means = []
        for layer_idx in self.continual_bp_layers:
            block = self.continual_bp_modules[layer_idx]
            stat_sum = self.continual_bp_activation_sum[layer_idx]
            stat_count = self.continual_bp_activation_count[layer_idx]
            if self.ddp:
                all_reduce(stat_sum, op=ReduceOp.SUM)
                all_reduce(stat_count, op=ReduceOp.SUM)
            if float(stat_count.item()) > 0.0:
                activation_mean = stat_sum / stat_count.clamp_min(1.0)
                outgoing_norm = block.linear2.weight.detach().float().norm(dim=0)
                utility_now = activation_mean * outgoing_norm
                self.continual_bp_utility[layer_idx].mul_(decay).add_(utility_now, alpha=1.0 - decay)
            self.continual_bp_age[layer_idx].add_(1)
            utility_means.append(float(self.continual_bp_utility[layer_idx].detach().mean().item()))

            stat_sum.zero_()
            stat_count.zero_()

            eligible = self.continual_bp_age[layer_idx] >= maturity_steps
            total_eligible += int(eligible.sum().item())
            if not event_step or replacement_rate <= 0.0 or not bool(eligible.any().item()):
                continue

            hidden_dim = int(self.continual_bp_utility[layer_idx].numel())
            self.continual_bp_credit[layer_idx] += replacement_rate * hidden_dim * replace_every
            replace_count = min(max_replace_per_event, int(self.continual_bp_credit[layer_idx]))
            replace_count = min(replace_count, int(eligible.sum().item()))
            if replace_count <= 0:
                continue

            scores = self.continual_bp_utility[layer_idx].detach().clone()
            scores[~eligible] = torch.inf
            channel_indices = torch.topk(scores, k=replace_count, largest=False).indices
            for channel_idx in channel_indices.tolist():
                self._reset_continual_bp_channel(layer_idx, int(channel_idx))
            self.continual_bp_credit[layer_idx] = max(0.0, self.continual_bp_credit[layer_idx] - replace_count)
            total_replaced += int(replace_count)

        metrics = {
            "continual_bp_active": 1.0,
            "continual_bp_replaced": float(total_replaced),
            "continual_bp_eligible_channels": float(total_eligible),
        }
        if utility_means:
            metrics["continual_bp_utility_mean"] = float(sum(utility_means) / len(utility_means))
        return metrics

    def configure_stage3_anchor(self):
        """Capture the loaded Stage-2 parameters for L2-SP and pullback anchoring."""

        if not self.stage3_anchor_enabled():
            self.stage3_anchor_params = {}
            self.stage3_anchor_numel = 0
            return

        anchors = {}
        numel = 0
        for name, param in self.raw_model.named_parameters():
            if not param.requires_grad or not torch.is_floating_point(param):
                continue
            anchors[name] = param.detach().clone()
            numel += int(param.numel())
        if not anchors:
            raise ValueError("Stage-3 anchoring was requested, but no trainable floating-point parameters were found.")

        self.stage3_anchor_params = anchors
        self.stage3_anchor_numel = numel
        if self.master_process:
            print(
                "Stage-3 parameter anchor enabled: "
                f"params={len(anchors)} numel={numel} "
                f"l2sp_weight={float(getattr(self.config, 'stage3_l2sp_weight', 0.0) or 0.0)} "
                f"pullback={float(getattr(self.config, 'stage3_anchor_pullback', 0.0) or 0.0)}"
            )

    def stage3_l2sp_loss(self) -> Optional[Tensor]:
        if not self.stage3_anchor_params:
            return None
        weight = float(getattr(self.config, "stage3_l2sp_weight", 0.0) or 0.0)
        if weight <= 0.0:
            return None

        sq_sum = None
        numel = 0
        for name, param in self.raw_model.named_parameters():
            anchor = self.stage3_anchor_params.get(name)
            if anchor is None:
                continue
            diff = param.float() - anchor.to(device=param.device, dtype=torch.float32)
            term = diff.pow(2).sum()
            sq_sum = term if sq_sum is None else sq_sum + term
            numel += int(param.numel())
        if sq_sum is None or numel == 0:
            return None
        return sq_sum / float(numel)

    def apply_stage3_anchor_pullback(self) -> Optional[Tensor]:
        if not self.stage3_anchor_params:
            return None
        pullback = float(getattr(self.config, "stage3_anchor_pullback", 0.0) or 0.0)
        if pullback <= 0.0:
            return None

        sq_sum = None
        numel = 0
        with torch.no_grad():
            for name, param in self.raw_model.named_parameters():
                anchor = self.stage3_anchor_params.get(name)
                if anchor is None:
                    continue
                anchor = anchor.to(device=param.device, dtype=param.dtype)
                diff = param.float() - anchor.float()
                term = diff.pow(2).sum()
                sq_sum = term if sq_sum is None else sq_sum + term
                numel += int(param.numel())
                param.lerp_(anchor, pullback)
        if sq_sum is None or numel == 0:
            return None
        return torch.sqrt(sq_sum / float(numel))

    def configure_stage3_teacher(self):
        """Load an optional frozen Stage-2 teacher for ordinary Stage-3 KD."""

        if not self.stage3_teacher_required():
            self.stage3_teacher_model = None
            return

        checkpoint_path = self.config.stage3_teacher_checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Stage-3 teacher checkpoint does not exist: {checkpoint_path}")

        teacher_model = TabICL(**self.model_config)
        teacher_model.to(device=self.config.device)

        if self.master_process:
            print(f"Loading frozen Stage-3 teacher from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)
        if "state_dict" not in checkpoint:
            raise ValueError("Stage-3 teacher checkpoint does not contain model state")
        teacher_model.load_state_dict(checkpoint["state_dict"])
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        self.stage3_teacher_model = teacher_model
        if self.master_process:
            kd_weight = float(getattr(self.config, "stage3_kd_weight", 0.0) or 0.0)
            kd_temperature = float(getattr(self.config, "stage3_kd_temperature", 2.0) or 2.0)
            print(f"Stage-3 teacher KD enabled: weight={kd_weight} temperature={kd_temperature}")

    def configure_stage3_ema(self):
        """Initialize a master-rank EMA shadow for saved Stage-3 checkpoints."""

        if not self.stage3_ema_enabled() or not self.master_process:
            self.stage3_ema_state = None
            self.stage3_ema_ready = False
            return

        self.stage3_ema_state = {}
        self.stage3_ema_ready = False
        print(
            "Stage-3 EMA checkpoints enabled: "
            f"decay={float(getattr(self.config, 'stage3_ema_decay', 0.0) or 0.0)} "
            f"start_step={int(getattr(self.config, 'stage3_ema_start_step', 0) or 0)} "
            f"save_ema={bool(getattr(self.config, 'stage3_save_ema_checkpoints', False))}"
        )

    def update_stage3_ema(self) -> bool:
        if self.stage3_ema_state is None:
            return False
        start_step = int(getattr(self.config, "stage3_ema_start_step", 0) or 0)
        if int(self.curr_step) < start_step:
            return False

        decay = float(getattr(self.config, "stage3_ema_decay", 0.0) or 0.0)
        with torch.no_grad():
            current = self.raw_model.state_dict()
            if not self.stage3_ema_ready:
                self.stage3_ema_state.clear()
                self.stage3_ema_state.update(
                    {
                        name: tensor.detach().clone()
                        for name, tensor in current.items()
                    }
                )
                self.stage3_ema_ready = True
                return True
            for name, tensor in current.items():
                shadow = self.stage3_ema_state.get(name)
                if shadow is None:
                    self.stage3_ema_state[name] = tensor.detach().clone()
                    continue
                value = tensor.detach()
                if torch.is_floating_point(shadow):
                    shadow.mul_(decay).add_(value.to(device=shadow.device, dtype=shadow.dtype), alpha=1.0 - decay)
                else:
                    shadow.copy_(value.to(device=shadow.device, dtype=shadow.dtype))
        return True

    def configure_firewall_kd(self):
        """Load real internet_firewall train+val rows for a small heldout KD task."""

        if not self.firewall_kd_enabled():
            self.firewall_kd_data = None
            return

        dataset_dir = os.path.abspath(os.path.expanduser(str(self.config.firewall_kd_dataset_dir)))
        required = ["N_train.npy", "y_train.npy", "N_val.npy", "y_val.npy"]
        missing = [name for name in required if not os.path.exists(os.path.join(dataset_dir, name))]
        if missing:
            raise FileNotFoundError(f"Firewall KD dataset is missing files in {dataset_dir}: {missing}")

        x_train = np.load(os.path.join(dataset_dir, "N_train.npy")).astype(np.float32, copy=False)
        y_train = np.load(os.path.join(dataset_dir, "y_train.npy")).astype(np.int64, copy=False).reshape(-1)
        x_val = np.load(os.path.join(dataset_dir, "N_val.npy")).astype(np.float32, copy=False)
        y_val = np.load(os.path.join(dataset_dir, "y_val.npy")).astype(np.int64, copy=False).reshape(-1)
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError(f"Invalid firewall KD arrays: X={x.shape} y={y.shape}")

        class_indices = {
            int(label): np.flatnonzero(y == label).astype(np.int64, copy=False)
            for label in np.unique(y)
        }
        if 2 not in class_indices:
            raise ValueError("Firewall KD requires class 2 to protect internet_firewall class-2 recall.")

        self.firewall_kd_data = {
            "dataset_dir": dataset_dir,
            "X": x,
            "y": y,
            "class_indices": class_indices,
            "num_features": int(x.shape[1]),
        }
        if self.master_process:
            counts = {label: int(indices.size) for label, indices in sorted(class_indices.items())}
            print(
                "Firewall heldout KD enabled: "
                f"dir={dataset_dir} rows={x.shape[0]} features={x.shape[1]} counts={counts} "
                f"weight={float(getattr(self.config, 'firewall_kd_weight', 0.0) or 0.0)} "
                f"ce_weight={float(getattr(self.config, 'firewall_kd_ce_weight', 0.0) or 0.0)}"
            )

    def save_checkpoint(self, name: str):
        """Save model and training state to checkpoint file.

        Parameters
        ----------
        name : str
            Filename for the checkpoint
        """

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        save_ema = (
            bool(getattr(self.config, "stage3_save_ema_checkpoints", False))
            and self.stage3_ema_state is not None
            and self.stage3_ema_ready
        )
        state_dict = self.stage3_ema_state if save_ema else self.raw_model.state_dict()
        checkpoint = {
            "config": self.model_config,
            "state_dict": state_dict,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "curr_step": self.curr_step,
            "training_contract": self.paired_training_contract(),
        }
        if save_ema:
            checkpoint["stage3_ema_decay"] = float(getattr(self.config, "stage3_ema_decay", 0.0) or 0.0)
            checkpoint["stage3_ema_checkpoint"] = True
        torch.save(checkpoint, checkpoint_path)

    def manage_checkpoint(self):
        """Manage temporary checkpoints by deleting the oldest when limit is exceeded."""
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                # Consider a checkpoint temporary if its step is not divisible by save_perm_every
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except:
                continue  # Ignore files that don't match the format

        # Sort temporary checkpoints by step number (ascending)
        temp_checkpoints.sort(key=lambda x: x[0])

        # Remove oldest temporary checkpoints if limit is exceeded
        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    print(f"Error removing checkpoint {ckpt_path}: {e}")

    @ddp_cleanup
    def train(self):
        """Main training loop.

        Iterates through batches, processes them, updates model parameters,
        and handles checkpoint saving and metric logging.
        """
        if not bool(getattr(self.config, "set_train_every_step", True)):
            self.model.train()

        if self.master_process:
            progress_file = sys.stdout if getattr(self.config, "progress_to_stdout", False) else sys.stderr
            step_progress = tqdm(
                range(self.curr_step, self.config.max_steps),
                desc="Step",
                leave=True,
                file=progress_file,
                mininterval=float(getattr(self.config, "progress_min_interval", 0.1)),
                maxinterval=float(getattr(self.config, "progress_max_interval", 10.0)),
                miniters=int(getattr(self.config, "progress_min_iters", 1)),
                dynamic_ncols=bool(getattr(self.config, "progress_dynamic_ncols", True)),
            )
        else:
            step_progress = range(self.curr_step, self.config.max_steps)

        dataloader = None if self.prior_cache is not None else iter(self.dataloader)
        for step in step_progress:
            # Get the next batch
            self.debug_log(f"step {step} begin")
            with Timer() as prior_timer:
                if self.prior_cache is not None:
                    batch = self.prior_cache.get_next_batch()
                else:
                    batch = next(dataloader)
            prior_time = prior_timer.elapsed
            self.debug_log(f"step {step} prior_done {prior_time:.3f}s")

            # Train the model on the batch
            with Timer() as train_timer:
                results = self.run_batch(batch)
            train_time = train_timer.elapsed
            self.debug_log(f"step {step} train_done {train_time:.3f}s")

            self.curr_step = step + 1
            empty_cache_every = int(getattr(self.config, "empty_cache_every", 1) or 0)
            if empty_cache_every > 0 and self.curr_step % empty_cache_every == 0:
                torch.cuda.empty_cache()

            if self.master_process:
                # Add timing information to results
                results.update({"prior_time": prior_time, "train_time": train_time})
                prior_cache_stats_every = int(getattr(self.config, "prior_cache_stats_every", 1) or 0)
                if (
                    self.prior_cache is not None
                    and prior_cache_stats_every > 0
                    and self.curr_step % prior_cache_stats_every == 0
                ):
                    results.update(self.prior_cache.get_stats())

                # Save checkpoints
                is_temp_save = self.curr_step % self.config.save_temp_every == 0
                is_perm_save = self.curr_step % self.config.save_perm_every == 0
                checkpoint_time = 0.0

                if is_temp_save or is_perm_save:
                    ckpt_name = f"step-{self.curr_step}.ckpt"
                    with Timer() as checkpoint_timer:
                        self.save_checkpoint(name=ckpt_name)

                        # Manage checkpoint limit only for temporary checkpoints
                        if is_temp_save and not is_perm_save and self.config.max_checkpoints > 0:
                            self.manage_checkpoint()
                    checkpoint_time = checkpoint_timer.elapsed
                results["checkpoint_time"] = checkpoint_time
                self._write_speed_trace(step, results)

                # Update progress bar with rounded values for cleaner display
                postfix_every = max(1, int(getattr(self.config, "progress_postfix_every", 1) or 1))
                if self.curr_step % postfix_every == 0 or is_temp_save or is_perm_save:
                    step_progress.set_postfix(
                        **{k: round(v, 3) if isinstance(v, float) else v for k, v in results.items()},
                        refresh=bool(getattr(self.config, "progress_postfix_refresh", True)),
                    )

            # Logging to Weights & Biases
            if self.wandb_run is not None:
                # Add learning rate to results
                results["lr"] = self.scheduler.get_last_lr()[0]
                wandb.log(results, step=self.curr_step)

        self._flush_batch_source_log()

    def validate_micro_batch(self, micro_seq_len, micro_train_size):
        """Validate consistent sequence length and train size within a micro batch.

        Ensures all datasets in a micro batch share the same sequence length and
        train/test split position, required for efficient batch processing during
        gradient accumulation.

        Parameters
        ----------
        micro_seq_len : Tensor
            Sequence lengths for each dataset, shape ``(micro_batch_size,)``.

        micro_train_size : Tensor
            Training sizes (split positions) for each dataset, shape
            ``(micro_batch_size,)``.

        Returns
        -------
        seq_len : int
            The common sequence length for the micro batch.

        train_size : int
            The common train size for the micro batch.

        Raises
        ------
        ValueError
            If sequence lengths or train sizes are inconsistent.
        """
        if bool(getattr(self.config, "validate_micro_batch_shapes", True)):
            if len(torch.unique(micro_seq_len)) > 1:
                raise ValueError("All datasets in the micro batch must have the same sequence length.")

            if len(torch.unique(micro_train_size)) > 1:
                raise ValueError("All datasets in the micro batch must have the same training size.")

        seq_len = micro_seq_len[0].item()
        train_size = micro_train_size[0].item()

        return seq_len, train_size

    def align_micro_batch(self, micro_X, micro_y, micro_d, seq_len):
        """Truncate micro batch tensors to required dimensions.

        Truncates sequence length and feature dimensions to the validated `seq_len`
        and the maximum active features (``micro_d.max()``) respectively. This
        optimizes memory and computation by removing unused tensor elements.

        Parameters
        ----------
        micro_X : Tensor
            Input features per dataset of shape ``(B, T, H)``.

        micro_y : Tensor
            Target labels per dataset of shape ``(B, T)``.

        micro_d : Tensor
            Number of active features per dataset of shape ``(B,)``.

        seq_len : int
            Validated sequence length for this micro batch.

        Returns
        -------
        micro_X : Tensor
            Truncated features of shape ``(B, seq_len, micro_d.max())``.

        micro_y : Tensor
            Truncated labels of shape ``(B, seq_len)``.
        """
        # Truncate sequence length
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]

        if micro_y.shape[1] > seq_len:
            micro_y = micro_y[:, :seq_len]

        # Truncate feature dimension
        max_features = micro_d.max().item()
        if micro_X.shape[-1] > max_features:
            micro_X = micro_X[..., :max_features]

        return micro_X, micro_y

    @staticmethod
    def support_class_counts(y_train: Tensor, num_classes: int) -> Tensor:
        """Count support-set labels per dataset for class-prior-aware losses."""

        y = y_train.long()
        valid = (y >= 0) & (y < num_classes)
        indices = y.clamp(min=0, max=max(num_classes - 1, 0))
        counts = torch.zeros(y.shape[0], num_classes, dtype=torch.float32, device=y.device)
        counts.scatter_add_(1, indices, valid.float())
        return counts

    @staticmethod
    def support_minority_weights(
        labels: Tensor,
        support_counts: Tensor,
        gamma: float,
        max_weight: float,
        min_count: int,
        minority_threshold: float,
    ) -> tuple[Tensor, Tensor]:
        """Return per-row weights for labels whose support frequency is a minority."""

        num_classes = int(support_counts.shape[-1])
        labels = labels.long()
        valid = (labels >= 0) & (labels < num_classes)
        label_idx = labels.clamp(min=0, max=max(num_classes - 1, 0))
        cls_counts = support_counts.gather(1, label_idx)
        support_total = support_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        max_counts = support_counts.max(dim=1, keepdim=True).values.clamp_min(1.0)
        cls_frac = cls_counts / support_total
        eligible = valid & (cls_counts >= float(min_count)) & (cls_frac <= float(minority_threshold))
        ratio = (max_counts / cls_counts.clamp_min(1.0)).pow(float(gamma))
        weights = torch.where(eligible, ratio.clamp(min=1.0, max=float(max_weight)), torch.ones_like(ratio))
        return weights, eligible

    @staticmethod
    def support_prior_kl_loss(
        logits: Tensor,
        support_counts: Tensor,
        min_count: int,
        eps: float,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Penalize query prediction marginals that erase support-present classes."""

        if logits.ndim != 3:
            raise ValueError(f"support_prior_kl_loss expects 3D logits, got shape={tuple(logits.shape)}")
        num_classes = int(logits.shape[-1])
        if int(support_counts.shape[-1]) != num_classes:
            support_counts = support_counts[..., :num_classes]

        present = support_counts >= float(min_count)
        valid_task = present.sum(dim=-1) >= 2
        if not bool(valid_task.any().item()):
            return None, None, None

        support_prior = torch.where(present, support_counts, torch.zeros_like(support_counts))
        support_prior = support_prior / support_prior.sum(dim=-1, keepdim=True).clamp_min(float(eps))
        pred_prior = F.softmax(logits.float(), dim=-1).mean(dim=1)
        pred_prior = pred_prior.clamp_min(float(eps))
        pred_prior = pred_prior / pred_prior.sum(dim=-1, keepdim=True).clamp_min(float(eps))
        prior_kl = (
            support_prior
            * (support_prior.clamp_min(float(eps)).log() - pred_prior.clamp_min(float(eps)).log())
        ).sum(dim=-1)
        prior_kl = prior_kl[valid_task].mean()
        present_classes = present.sum(dim=-1).float()[valid_task].mean()
        support_entropy = -(
            support_prior.clamp_min(float(eps)) * support_prior.clamp_min(float(eps)).log()
        ).sum(dim=-1)
        support_entropy = support_entropy[valid_task].mean()
        return prior_kl, present_classes, support_entropy

    @staticmethod
    def support_prior_floor_loss(
        logits: Tensor,
        support_counts: Tensor,
        min_count: int,
        ratio: float,
        min_prior: float,
        max_prior: float,
        eps: float,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Asymmetric prior loss that discourages erasing support-present classes."""

        if logits.ndim != 3:
            raise ValueError(f"support_prior_floor_loss expects 3D logits, got shape={tuple(logits.shape)}")
        num_classes = int(logits.shape[-1])
        if int(support_counts.shape[-1]) != num_classes:
            support_counts = support_counts[..., :num_classes]

        present = support_counts >= float(min_count)
        support_prior = torch.where(present, support_counts, torch.zeros_like(support_counts))
        support_prior = support_prior / support_prior.sum(dim=-1, keepdim=True).clamp_min(float(eps))
        eligible = (
            present
            & (support_prior >= float(min_prior))
            & (support_prior <= float(max_prior))
            & (support_prior > 0.0)
        )
        valid_task = eligible.sum(dim=-1) >= 1
        if not bool(valid_task.any().item()):
            return None, None, None, None

        pred_prior = F.softmax(logits.float(), dim=-1).mean(dim=1)
        pred_prior = pred_prior.clamp_min(float(eps))
        pred_prior = pred_prior / pred_prior.sum(dim=-1, keepdim=True).clamp_min(float(eps))
        floor = (support_prior * float(ratio)).clamp_min(float(eps))
        under = F.relu(floor.log() - pred_prior.clamp_min(float(eps)).log())
        class_weights = torch.where(eligible, support_prior, torch.zeros_like(support_prior))
        loss_per_task = (under * class_weights).sum(dim=-1) / class_weights.sum(dim=-1).clamp_min(float(eps))
        loss = loss_per_task[valid_task].mean()

        under_mask = eligible & (pred_prior < floor)
        active_classes = eligible.sum(dim=-1).float()[valid_task].mean()
        under_rate = under_mask.float().sum(dim=-1)[valid_task] / eligible.float().sum(dim=-1)[valid_task].clamp_min(1.0)
        under_rate = under_rate.mean()
        pred_to_support = pred_prior / support_prior.clamp_min(float(eps))
        min_ratio = torch.where(eligible, pred_to_support, torch.full_like(pred_to_support, float("inf")))
        min_ratio = min_ratio.min(dim=-1).values[valid_task].mean()
        return loss, active_classes, under_rate, min_ratio

    @staticmethod
    def support_classwise_kd_loss(
        student_logits: Tensor,
        teacher_logits: Tensor,
        support_counts: Tensor,
        temperature: float,
        gamma: float,
        max_weight: float,
        min_count: int,
        min_prior: float,
        max_prior: float,
        eps: float,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Binary classwise KD for support-present non-majority classes.

        Ordinary KL preserves the whole teacher distribution uniformly. This
        auxiliary loss focuses on classes that appear in the support set but are
        not the dominant class, so Stage-3 synthetic CE has less room to erase
        mid-frequency class probabilities late in training.
        """

        if student_logits.ndim != 3 or teacher_logits.ndim != 3:
            raise ValueError(
                "support_classwise_kd_loss expects 3D logits, "
                f"got student={tuple(student_logits.shape)} teacher={tuple(teacher_logits.shape)}"
            )
        shared_classes = min(int(student_logits.shape[-1]), int(teacher_logits.shape[-1]))
        if shared_classes <= 0:
            return None, None, None, None
        student_logits = student_logits[..., :shared_classes]
        teacher_logits = teacher_logits[..., :shared_classes]
        if int(support_counts.shape[-1]) != shared_classes:
            support_counts = support_counts[..., :shared_classes]

        present = support_counts >= float(min_count)
        support_prior = torch.where(present, support_counts, torch.zeros_like(support_counts))
        support_prior = support_prior / support_prior.sum(dim=-1, keepdim=True).clamp_min(float(eps))
        eligible = (
            present
            & (support_prior >= float(min_prior))
            & (support_prior <= float(max_prior))
            & (support_prior > 0.0)
        )
        valid_task = eligible.sum(dim=-1) >= 1
        if not bool(valid_task.any().item()):
            return None, None, None, None

        dominant_prior = support_prior.max(dim=-1, keepdim=True).values.clamp_min(float(eps))
        ratio = (dominant_prior / support_prior.clamp_min(float(eps))).pow(float(gamma))
        class_weights = torch.where(
            eligible,
            ratio.clamp(min=1.0, max=float(max_weight)),
            torch.zeros_like(ratio),
        )

        temperature = float(temperature)
        teacher_prob = F.softmax(teacher_logits.float() / temperature, dim=-1).clamp(float(eps), 1.0 - float(eps))
        student_prob = F.softmax(student_logits.float() / temperature, dim=-1).clamp(float(eps), 1.0 - float(eps))
        binary_kl = teacher_prob * (teacher_prob.log() - student_prob.log())
        binary_kl = binary_kl + (1.0 - teacher_prob) * ((1.0 - teacher_prob).log() - (1.0 - student_prob).log())
        binary_kl = binary_kl * (temperature * temperature)

        weighted = binary_kl * class_weights.unsqueeze(1)
        denom = class_weights.sum(dim=-1).clamp_min(float(eps)) * float(student_logits.shape[1])
        loss_per_task = weighted.sum(dim=(1, 2)) / denom
        loss = loss_per_task[valid_task].mean()
        active_classes = eligible.sum(dim=-1).float()[valid_task].mean()
        max_class_weight = class_weights.max(dim=-1).values[valid_task].mean()
        eligible_prior = torch.where(eligible, support_prior, torch.zeros_like(support_prior))
        mean_eligible_prior = (
            eligible_prior.sum(dim=-1)[valid_task] / eligible.sum(dim=-1).float()[valid_task].clamp_min(1.0)
        ).mean()
        return loss, active_classes, max_class_weight, mean_eligible_prior

    def support_margin_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        support_counts: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Smooth true-vs-competing logit margin for support-present non-majority classes."""

        num_classes = int(logits.shape[-1])
        margin = float(getattr(self.config, "support_margin_value", 0.5) or 0.0)
        weights, eligible = self.support_minority_weights(
            labels,
            support_counts,
            gamma=float(getattr(self.config, "support_margin_gamma", 0.5) or 0.0),
            max_weight=float(getattr(self.config, "support_margin_max_weight", 6.0) or 6.0),
            min_count=int(getattr(self.config, "support_margin_min_count", 2) or 2),
            minority_threshold=float(getattr(self.config, "support_margin_threshold", 0.5) or 0.5),
        )
        valid = eligible & (labels.long() >= 0) & (labels.long() < num_classes)
        if not bool(valid.any().item()):
            return None, None, None

        label_idx = labels.long().clamp(min=0, max=max(num_classes - 1, 0))
        true_logit = logits.gather(-1, label_idx.unsqueeze(-1)).squeeze(-1)
        one_hot = F.one_hot(label_idx, num_classes=num_classes).bool()
        other_logit = logits.masked_fill(one_hot, torch.finfo(logits.dtype).min).max(dim=-1).values
        row_loss = F.softplus(margin + other_logit.float() - true_logit.float())
        row_weight = torch.where(valid, weights, torch.zeros_like(weights))
        weight_sum = row_weight.sum()
        if not bool(weight_sum.detach().item() > 0.0):
            return None, None, None
        loss = (row_loss * row_weight).sum() / weight_sum.clamp_min(1e-12)
        boost_rate = valid.float().mean()
        max_sample_weight = row_weight.max()
        return loss, boost_rate, max_sample_weight

    def plasticity_reg_scale(self) -> float:
        if not bool(getattr(self.config, "plasticity_reg_enabled", False)):
            return 0.0
        start_step = int(getattr(self.config, "plasticity_reg_start_step", 10000) or 0)
        ramp_steps = int(getattr(self.config, "plasticity_reg_ramp_steps", 2000) or 0)
        end_step = int(getattr(self.config, "plasticity_reg_end_step", -1) or -1)
        if end_step >= 0 and self.curr_step >= end_step:
            return 0.0
        if self.curr_step < start_step:
            return 0.0
        if ramp_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, (float(self.curr_step - start_step) + 1.0) / float(ramp_steps)))

    def plasticity_proto_scale(self) -> float:
        if not bool(getattr(self.config, "plasticity_proto_enabled", False)):
            return 0.0
        start_step = int(getattr(self.config, "plasticity_proto_start_step", 12000) or 0)
        ramp_steps = int(getattr(self.config, "plasticity_proto_ramp_steps", 2000) or 0)
        end_step = int(getattr(self.config, "plasticity_proto_end_step", -1) or -1)
        if end_step >= 0 and self.curr_step >= end_step:
            return 0.0
        if self.curr_step < start_step:
            return 0.0
        if ramp_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, (float(self.curr_step - start_step) + 1.0) / float(ramp_steps)))

    def plasticity_attn_scale(self) -> float:
        if not bool(getattr(self.config, "plasticity_attn_enabled", False)):
            return 0.0
        start_step = int(getattr(self.config, "plasticity_attn_start_step", 12000) or 0)
        ramp_steps = int(getattr(self.config, "plasticity_attn_ramp_steps", 2000) or 0)
        end_step = int(getattr(self.config, "plasticity_attn_end_step", 18000) or -1)
        if end_step >= 0 and self.curr_step >= end_step:
            return 0.0
        if self.curr_step < start_step:
            return 0.0
        if ramp_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, (float(self.curr_step - start_step) + 1.0) / float(ramp_steps)))

    def plasticity_rep_source(self) -> str:
        return str(getattr(self.config, "plasticity_reg_source", "row") or "row").lower()

    def plasticity_parse_layers(self, value: str | None, default: str = "") -> list[int]:
        value = str(value if value is not None else default)
        layers: list[int] = []
        for item in re.split(r"[,;]", value):
            item = item.strip()
            if not item:
                continue
            layers.append(int(item))
        return layers

    def plasticity_reg_layer_indices(self) -> list[int]:
        return self.plasticity_parse_layers(
            str(getattr(self.config, "plasticity_reg_layers", "8,9,10") or ""),
            default="8,9,10",
        )

    def plasticity_low_high_teacher_layers(self) -> list[int]:
        return self.plasticity_parse_layers(
            str(getattr(self.config, "plasticity_low_high_teacher_layers", "5,6") or ""),
            default="5,6",
        )

    def plasticity_low_high_student_layers(self) -> list[int]:
        return self.plasticity_parse_layers(
            str(getattr(self.config, "plasticity_low_high_student_layers", "8,9,10") or ""),
            default="8,9,10",
        )

    def plasticity_proto_layer_indices(self) -> list[int]:
        return self.plasticity_parse_layers(
            str(getattr(self.config, "plasticity_proto_layers", "8,9,10") or ""),
            default="8,9,10",
        )

    def plasticity_attn_layer_indices(self) -> list[int]:
        return self.plasticity_parse_layers(
            str(getattr(self.config, "plasticity_attn_layers", "8,9,10") or ""),
            default="8,9,10",
        )

    def plasticity_icl_layers(self) -> list[int]:
        layers = list(self.plasticity_reg_layer_indices())
        low_high_weight = float(getattr(self.config, "plasticity_low_high_cka_weight", 0.0) or 0.0)
        if low_high_weight > 0.0:
            layers.extend(self.plasticity_low_high_teacher_layers())
            layers.extend(self.plasticity_low_high_student_layers())
        if bool(getattr(self.config, "plasticity_proto_enabled", False)):
            proto_weights = (
                float(getattr(self.config, "plasticity_proto_entropy_weight", 0.0) or 0.0)
                + float(getattr(self.config, "plasticity_proto_top1_weight", 0.0) or 0.0)
                + float(getattr(self.config, "plasticity_proto_usage_weight", 0.0) or 0.0)
            )
            if proto_weights > 0.0:
                layers.extend(self.plasticity_proto_layer_indices())
        if bool(getattr(self.config, "plasticity_attn_enabled", False)):
            attn_weights = (
                float(getattr(self.config, "plasticity_attn_entropy_weight", 0.0) or 0.0)
                + float(getattr(self.config, "plasticity_attn_top1_weight", 0.0) or 0.0)
                + float(getattr(self.config, "plasticity_attn_usage_weight", 0.0) or 0.0)
            )
            if attn_weights > 0.0:
                # Attention for block L is recomputed from the representation entering
                # block L, which is the output of block L - 1 for L > 0.
                layers.extend(layer_idx - 1 for layer_idx in self.plasticity_attn_layer_indices() if layer_idx > 0)
        return sorted(set(layers))

    def plasticity_needs_representations(
        self,
        scale: float,
        proto_scale: float = 0.0,
        attn_scale: float = 0.0,
    ) -> tuple[bool, bool]:
        if scale <= 0.0 and proto_scale <= 0.0 and attn_scale <= 0.0:
            return False, False
        cos_weight = float(getattr(self.config, "plasticity_cos_weight", 0.0) or 0.0)
        var_weight = float(getattr(self.config, "plasticity_var_weight", 0.0) or 0.0)
        cka_weight = float(getattr(self.config, "plasticity_cka_weight", 0.0) or 0.0)
        low_high_weight = float(getattr(self.config, "plasticity_low_high_cka_weight", 0.0) or 0.0)
        proto_weight = (
            float(getattr(self.config, "plasticity_proto_entropy_weight", 0.0) or 0.0)
            + float(getattr(self.config, "plasticity_proto_top1_weight", 0.0) or 0.0)
            + float(getattr(self.config, "plasticity_proto_usage_weight", 0.0) or 0.0)
        )
        attn_weight = (
            float(getattr(self.config, "plasticity_attn_entropy_weight", 0.0) or 0.0)
            + float(getattr(self.config, "plasticity_attn_top1_weight", 0.0) or 0.0)
            + float(getattr(self.config, "plasticity_attn_usage_weight", 0.0) or 0.0)
        )
        needs_base_repr = cos_weight > 0.0 or var_weight > 0.0 or cka_weight > 0.0
        needs_proto_repr = proto_scale > 0.0 and proto_weight > 0.0
        needs_attn_repr = attn_scale > 0.0 and attn_weight > 0.0
        if not needs_base_repr and low_high_weight <= 0.0 and not needs_proto_repr and not needs_attn_repr:
            return False, False
        source = self.plasticity_rep_source()
        needs_row = scale > 0.0 and needs_base_repr and source == "row"
        needs_icl = (
            (scale > 0.0 and needs_base_repr and source == "icl")
            or low_high_weight > 0.0
            or needs_proto_repr
            or needs_attn_repr
        )
        return needs_row, needs_icl

    def plasticity_icl_representation_map(
        self,
        representations: Tensor | dict[int, Tensor] | None,
    ) -> dict[int, Tensor]:
        if representations is None or torch.is_tensor(representations):
            return {}
        icl_representations = representations.get("icl") if isinstance(representations, dict) else None
        if isinstance(icl_representations, dict):
            return {int(layer_idx): value for layer_idx, value in icl_representations.items() if torch.is_tensor(value)}
        return {int(layer_idx): value for layer_idx, value in representations.items() if torch.is_tensor(value)}

    def plasticity_table_representation(self, row_representations: Tensor, train_size: int) -> Tensor | None:
        """Pool selected rows into one representation per table.

        This intentionally regularizes table-level representations across
        different tables in a micro-batch, not row-row distances within a table.
        """

        span = str(getattr(self.config, "plasticity_reg_span", "query") or "query").lower()
        if span == "support":
            selected = row_representations[:, :train_size, :]
        elif span == "full":
            selected = row_representations
        else:
            selected = row_representations[:, train_size:, :]

        if selected.shape[1] <= 0:
            return None

        sample_rows = int(getattr(self.config, "plasticity_reg_table_sample_rows", 0) or 0)
        if sample_rows > 0 and selected.shape[1] > sample_rows:
            indices = torch.linspace(
                0,
                selected.shape[1] - 1,
                sample_rows,
                device=selected.device,
            ).round().long()
            selected = selected.index_select(1, indices)

        return selected.float().mean(dim=1)

    def plasticity_table_representations(
        self,
        representations: Tensor | dict[int, Tensor] | None,
        train_size: int,
    ) -> list[tuple[str, Tensor]]:
        if representations is None:
            return []
        if torch.is_tensor(representations):
            z_table = self.plasticity_table_representation(representations, train_size)
            return [] if z_table is None else [("row", z_table)]
        out: list[tuple[str, Tensor]] = []
        if "row" in representations or "icl" in representations:
            row_representations = representations.get("row")
            if torch.is_tensor(row_representations):
                z_table = self.plasticity_table_representation(row_representations, train_size)
                if z_table is not None:
                    out.append(("row", z_table))
            icl_representations = representations.get("icl")
            if isinstance(icl_representations, dict):
                for layer_idx in sorted(icl_representations):
                    z_table = self.plasticity_table_representation(icl_representations[layer_idx], train_size)
                    if z_table is not None:
                        out.append((f"icl{layer_idx}", z_table))
            return out
        for layer_idx in sorted(representations):
            z_table = self.plasticity_table_representation(representations[layer_idx], train_size)
            if z_table is not None:
                out.append((f"icl{layer_idx}", z_table))
        return out

    def plasticity_linear_cka_loss(
        self,
        current: Tensor,
        reference: Tensor,
        *,
        detach_reference: bool = True,
    ) -> tuple[Tensor, Tensor]:
        current = current.float()
        reference = reference.detach().float() if detach_reference else reference.float()
        current = current - current.mean(dim=0, keepdim=True)
        reference = reference - reference.mean(dim=0, keepdim=True)
        cross = current.transpose(0, 1) @ reference
        current_cov = current.transpose(0, 1) @ current
        reference_cov = reference.transpose(0, 1) @ reference
        numerator = cross.pow(2).sum()
        denominator = current_cov.pow(2).sum().sqrt() * reference_cov.pow(2).sum().sqrt()
        eps = float(getattr(self.config, "plasticity_cka_eps", 1e-8) or 1e-8)
        cka = numerator / denominator.clamp_min(eps)
        return 1.0 - cka, cka

    def plasticity_subsample_rows(self, tensor: Tensor, max_rows: int) -> Tensor:
        if max_rows <= 0 or tensor.shape[1] <= max_rows:
            return tensor
        indices = torch.linspace(0, tensor.shape[1] - 1, max_rows, device=tensor.device).round().long()
        return tensor.index_select(1, indices)

    def plasticity_support_prototype_loss(
        self,
        representations: Tensor | dict[int, Tensor] | None,
        train_size: int,
        scale: float,
    ) -> tuple[list[Tensor], dict[str, float]]:
        metrics: dict[str, float] = {"plasticity_proto_scale": scale}
        if scale <= 0.0 or not bool(getattr(self.config, "plasticity_proto_enabled", False)):
            return [], metrics

        entropy_weight = float(getattr(self.config, "plasticity_proto_entropy_weight", 0.0) or 0.0)
        top1_weight = float(getattr(self.config, "plasticity_proto_top1_weight", 0.0) or 0.0)
        usage_weight = float(getattr(self.config, "plasticity_proto_usage_weight", 0.0) or 0.0)
        if entropy_weight <= 0.0 and top1_weight <= 0.0 and usage_weight <= 0.0:
            return [], metrics

        icl_map = self.plasticity_icl_representation_map(representations)
        if not icl_map:
            return [], metrics

        tau = float(getattr(self.config, "plasticity_proto_tau", 0.07) or 0.07)
        entropy_min = float(getattr(self.config, "plasticity_proto_entropy_min", 0.88) or 0.0)
        top1_limit = float(getattr(self.config, "plasticity_proto_top1_limit", 0.08) or 1.0)
        query_sample_rows = int(getattr(self.config, "plasticity_proto_query_sample_rows", 128) or 0)
        support_sample_rows = int(getattr(self.config, "plasticity_proto_support_sample_rows", 512) or 0)
        proto_layers = self.plasticity_proto_layer_indices()

        entropy_losses: list[Tensor] = []
        top1_losses: list[Tensor] = []
        usage_losses: list[Tensor] = []
        entropy_means: list[float] = []
        top1_means: list[float] = []
        usage_concentrations: list[float] = []
        usage_effective_fracs: list[float] = []
        active_layers = 0

        for layer_idx in proto_layers:
            hidden = icl_map.get(layer_idx)
            if hidden is None:
                continue
            support = hidden[:, :train_size, :]
            query = hidden[:, train_size:, :]
            if support.shape[1] < 2 or query.shape[1] < 1:
                continue

            support = self.plasticity_subsample_rows(support, support_sample_rows).float()
            query = self.plasticity_subsample_rows(query, query_sample_rows).float()
            support = F.normalize(support, dim=-1, eps=1e-6)
            query = F.normalize(query, dim=-1, eps=1e-6)

            assignment = torch.softmax(torch.bmm(query, support.transpose(1, 2)) / tau, dim=-1)
            assignment = assignment.clamp_min(1e-12)
            support_count = int(assignment.shape[-1])
            log_support_count = math.log(max(2, support_count))
            entropy_norm = -(assignment * assignment.log()).sum(dim=-1) / log_support_count
            top1 = assignment.max(dim=-1).values
            usage = assignment.mean(dim=1)
            uniform_usage = 1.0 / float(max(1, support_count))
            usage_concentration = (usage.square().sum(dim=-1) * float(support_count) - 1.0).clamp_min(0.0)
            usage_entropy = -(usage * usage.clamp_min(1e-12).log()).sum(dim=-1)
            usage_effective_frac = usage_entropy.exp() / float(max(1, support_count))

            active_layers += 1
            entropy_mean = float(entropy_norm.detach().mean().item())
            top1_mean = float(top1.detach().mean().item())
            usage_concentration_mean = float(usage_concentration.detach().mean().item())
            entropy_means.append(entropy_mean)
            top1_means.append(top1_mean)
            usage_concentrations.append(usage_concentration_mean)
            usage_effective_fracs.append(float(usage_effective_frac.detach().mean().item()))
            metrics[f"plasticity_proto_icl{layer_idx}_entropy_norm"] = entropy_mean
            metrics[f"plasticity_proto_icl{layer_idx}_top1"] = top1_mean
            metrics[f"plasticity_proto_icl{layer_idx}_usage_concentration"] = usage_concentration_mean
            metrics[f"plasticity_proto_icl{layer_idx}_usage_effective_frac"] = usage_effective_fracs[-1]
            metrics[f"plasticity_proto_icl{layer_idx}_uniform_usage"] = uniform_usage

            if entropy_weight > 0.0:
                entropy_loss = F.relu(entropy_min - entropy_norm).pow(2).mean()
                entropy_losses.append(entropy_loss)
                metrics[f"plasticity_proto_icl{layer_idx}_entropy_loss"] = float(entropy_loss.detach().item())
            if top1_weight > 0.0:
                top1_loss = F.relu(top1 - top1_limit).pow(2).mean()
                top1_losses.append(top1_loss)
                metrics[f"plasticity_proto_icl{layer_idx}_top1_loss"] = float(top1_loss.detach().item())
            if usage_weight > 0.0:
                usage_loss = usage_concentration.mean()
                usage_losses.append(usage_loss)
                metrics[f"plasticity_proto_icl{layer_idx}_usage_loss"] = float(usage_loss.detach().item())

        terms: list[Tensor] = []
        metrics["plasticity_proto_layers"] = float(active_layers)
        metrics["plasticity_proto_tau"] = tau
        metrics["plasticity_proto_entropy_min"] = entropy_min
        metrics["plasticity_proto_top1_limit"] = top1_limit
        if entropy_means:
            metrics["plasticity_proto_entropy_norm_mean"] = float(sum(entropy_means) / len(entropy_means))
        if top1_means:
            metrics["plasticity_proto_top1_mean"] = float(sum(top1_means) / len(top1_means))
        if usage_concentrations:
            metrics["plasticity_proto_usage_concentration_mean"] = float(
                sum(usage_concentrations) / len(usage_concentrations)
            )
        if usage_effective_fracs:
            metrics["plasticity_proto_usage_effective_frac_mean"] = float(
                sum(usage_effective_fracs) / len(usage_effective_fracs)
            )

        if entropy_losses:
            entropy_loss = torch.stack(entropy_losses).mean()
            terms.append((scale * entropy_weight) * entropy_loss)
            metrics["plasticity_proto_entropy_loss"] = float(entropy_loss.detach().item())
            metrics["plasticity_proto_entropy_weight"] = scale * entropy_weight
        if top1_losses:
            top1_loss = torch.stack(top1_losses).mean()
            terms.append((scale * top1_weight) * top1_loss)
            metrics["plasticity_proto_top1_loss"] = float(top1_loss.detach().item())
            metrics["plasticity_proto_top1_weight"] = scale * top1_weight
        if usage_losses:
            usage_loss = torch.stack(usage_losses).mean()
            terms.append((scale * usage_weight) * usage_loss)
            metrics["plasticity_proto_usage_loss"] = float(usage_loss.detach().item())
            metrics["plasticity_proto_usage_weight"] = scale * usage_weight

        return terms, metrics

    def plasticity_attention_anti_collapse_loss(
        self,
        representations: Tensor | dict[int, Tensor] | None,
        train_size: int,
        scale: float,
    ) -> tuple[list[Tensor], dict[str, float]]:
        metrics: dict[str, float] = {"plasticity_attn_scale": scale}
        if scale <= 0.0 or not bool(getattr(self.config, "plasticity_attn_enabled", False)):
            return [], metrics

        entropy_weight = float(getattr(self.config, "plasticity_attn_entropy_weight", 0.0) or 0.0)
        top1_weight = float(getattr(self.config, "plasticity_attn_top1_weight", 0.0) or 0.0)
        usage_weight = float(getattr(self.config, "plasticity_attn_usage_weight", 0.0) or 0.0)
        if entropy_weight <= 0.0 and top1_weight <= 0.0 and usage_weight <= 0.0:
            return [], metrics

        icl_map = self.plasticity_icl_representation_map(representations)
        if not icl_map:
            return [], metrics

        blocks = getattr(getattr(self.raw_model, "icl_predictor", None), "tf_icl", None)
        blocks = getattr(blocks, "blocks", None)
        if blocks is None:
            return [], metrics

        entropy_min = float(getattr(self.config, "plasticity_attn_entropy_min", 0.60) or 0.0)
        top1_limit = float(getattr(self.config, "plasticity_attn_top1_limit", 0.22) or 1.0)
        query_sample_rows = int(getattr(self.config, "plasticity_attn_query_sample_rows", 128) or 0)
        support_sample_rows = int(getattr(self.config, "plasticity_attn_support_sample_rows", 512) or 0)

        entropy_losses: list[Tensor] = []
        top1_losses: list[Tensor] = []
        usage_losses: list[Tensor] = []
        entropy_means: list[float] = []
        top1_means: list[float] = []
        usage_concentrations: list[float] = []
        usage_effective_fracs: list[float] = []
        active_layers = 0

        for layer_idx in self.plasticity_attn_layer_indices():
            if layer_idx <= 0 or layer_idx >= len(blocks):
                continue
            hidden_in = icl_map.get(layer_idx - 1)
            if hidden_in is None or hidden_in.shape[1] <= train_size:
                continue
            block = blocks[layer_idx]
            attn = getattr(block, "attn", None)
            if attn is None:
                continue

            hidden_normed = block.norm1(hidden_in) if getattr(block, "norm_first", True) else hidden_in
            support = hidden_normed[:, :train_size, :]
            query = hidden_normed[:, train_size:, :]
            if support.shape[1] < 2 or query.shape[1] < 1:
                continue
            support = self.plasticity_subsample_rows(support, support_sample_rows)
            query = self.plasticity_subsample_rows(query, query_sample_rows)

            embed_dim = int(attn.embed_dim)
            num_heads = int(attn.num_heads)
            head_dim = embed_dim // num_heads
            in_proj_weight = attn.in_proj_weight
            in_proj_bias = attn.in_proj_bias
            q_bias = None if in_proj_bias is None else in_proj_bias[:embed_dim]
            k_bias = None if in_proj_bias is None else in_proj_bias[embed_dim : 2 * embed_dim]
            q_proj = F.linear(query, in_proj_weight[:embed_dim], q_bias)
            k_proj = F.linear(support, in_proj_weight[embed_dim : 2 * embed_dim], k_bias)
            q_proj = q_proj.view(q_proj.shape[0], q_proj.shape[1], num_heads, head_dim).transpose(1, 2)
            k_proj = k_proj.view(k_proj.shape[0], k_proj.shape[1], num_heads, head_dim).transpose(1, 2)

            ssmax_layer = getattr(attn, "ssmax_layer", None)
            if ssmax_layer is not None:
                q_proj = ssmax_layer(q_proj, int(k_proj.shape[-2]))

            logits = torch.matmul(q_proj.float(), k_proj.float().transpose(-2, -1)) / math.sqrt(float(head_dim))
            weights = torch.softmax(logits, dim=-1).clamp_min(1e-12)
            support_count = int(weights.shape[-1])
            log_support_count = math.log(max(2, support_count))
            entropy_norm = -(weights * weights.log()).sum(dim=-1) / log_support_count
            top1 = weights.max(dim=-1).values
            usage = weights.mean(dim=(1, 2))
            uniform_usage = 1.0 / float(max(1, support_count))
            usage_concentration = (usage.square().sum(dim=-1) * float(support_count) - 1.0).clamp_min(0.0)
            usage_entropy = -(usage * usage.clamp_min(1e-12).log()).sum(dim=-1)
            usage_effective_frac = usage_entropy.exp() / float(max(1, support_count))

            active_layers += 1
            entropy_mean = float(entropy_norm.detach().mean().item())
            top1_mean = float(top1.detach().mean().item())
            usage_concentration_mean = float(usage_concentration.detach().mean().item())
            entropy_means.append(entropy_mean)
            top1_means.append(top1_mean)
            usage_concentrations.append(usage_concentration_mean)
            usage_effective_fracs.append(float(usage_effective_frac.detach().mean().item()))
            metrics[f"plasticity_attn_icl{layer_idx}_entropy_norm"] = entropy_mean
            metrics[f"plasticity_attn_icl{layer_idx}_top1"] = top1_mean
            metrics[f"plasticity_attn_icl{layer_idx}_usage_concentration"] = usage_concentration_mean
            metrics[f"plasticity_attn_icl{layer_idx}_usage_effective_frac"] = usage_effective_fracs[-1]
            metrics[f"plasticity_attn_icl{layer_idx}_uniform_usage"] = uniform_usage

            if entropy_weight > 0.0:
                entropy_loss = F.relu(entropy_min - entropy_norm).pow(2).mean()
                entropy_losses.append(entropy_loss)
                metrics[f"plasticity_attn_icl{layer_idx}_entropy_loss"] = float(entropy_loss.detach().item())
            if top1_weight > 0.0:
                top1_loss = F.relu(top1 - top1_limit).pow(2).mean()
                top1_losses.append(top1_loss)
                metrics[f"plasticity_attn_icl{layer_idx}_top1_loss"] = float(top1_loss.detach().item())
            if usage_weight > 0.0:
                usage_loss = usage_concentration.mean()
                usage_losses.append(usage_loss)
                metrics[f"plasticity_attn_icl{layer_idx}_usage_loss"] = float(usage_loss.detach().item())

        terms: list[Tensor] = []
        metrics["plasticity_attn_layers"] = float(active_layers)
        metrics["plasticity_attn_entropy_min"] = entropy_min
        metrics["plasticity_attn_top1_limit"] = top1_limit
        if entropy_means:
            metrics["plasticity_attn_entropy_norm_mean"] = float(sum(entropy_means) / len(entropy_means))
        if top1_means:
            metrics["plasticity_attn_top1_mean"] = float(sum(top1_means) / len(top1_means))
        if usage_concentrations:
            metrics["plasticity_attn_usage_concentration_mean"] = float(
                sum(usage_concentrations) / len(usage_concentrations)
            )
        if usage_effective_fracs:
            metrics["plasticity_attn_usage_effective_frac_mean"] = float(
                sum(usage_effective_fracs) / len(usage_effective_fracs)
            )

        if entropy_losses:
            entropy_loss = torch.stack(entropy_losses).mean()
            terms.append((scale * entropy_weight) * entropy_loss)
            metrics["plasticity_attn_entropy_loss"] = float(entropy_loss.detach().item())
            metrics["plasticity_attn_entropy_weight"] = scale * entropy_weight
        if top1_losses:
            top1_loss = torch.stack(top1_losses).mean()
            terms.append((scale * top1_weight) * top1_loss)
            metrics["plasticity_attn_top1_loss"] = float(top1_loss.detach().item())
            metrics["plasticity_attn_top1_weight"] = scale * top1_weight
        if usage_losses:
            usage_loss = torch.stack(usage_losses).mean()
            terms.append((scale * usage_weight) * usage_loss)
            metrics["plasticity_attn_usage_loss"] = float(usage_loss.detach().item())
            metrics["plasticity_attn_usage_weight"] = scale * usage_weight

        return terms, metrics

    def plasticity_regularization_loss(
        self,
        representations: Tensor | dict[int, Tensor] | None,
        reference_representations: Tensor | dict[int, Tensor] | None,
        logits: Tensor,
        y_train: Tensor,
        train_size: int,
    ) -> tuple[Tensor | None, dict[str, float]]:
        scale = self.plasticity_reg_scale()
        proto_scale = self.plasticity_proto_scale()
        attn_scale = self.plasticity_attn_scale()
        metrics: dict[str, float] = {
            "plasticity_reg_scale": scale,
            "plasticity_proto_scale": proto_scale,
            "plasticity_attn_scale": attn_scale,
        }
        if scale <= 0.0 and proto_scale <= 0.0 and attn_scale <= 0.0:
            return None, metrics

        terms: list[Tensor] = []
        cos_weight = float(getattr(self.config, "plasticity_cos_weight", 0.0) or 0.0)
        var_weight = float(getattr(self.config, "plasticity_var_weight", 0.0) or 0.0)
        conf_weight = float(getattr(self.config, "plasticity_conf_weight", 0.0) or 0.0)
        entropy_floor_weight = float(getattr(self.config, "plasticity_entropy_floor_weight", 0.0) or 0.0)
        cka_weight = float(getattr(self.config, "plasticity_cka_weight", 0.0) or 0.0)
        low_high_cka_weight = float(getattr(self.config, "plasticity_low_high_cka_weight", 0.0) or 0.0)

        all_z_tables = []
        if scale > 0.0 and (
            cos_weight > 0.0 or var_weight > 0.0 or cka_weight > 0.0 or low_high_cka_weight > 0.0
        ):
            all_z_tables = self.plasticity_table_representations(representations, train_size)
            if all_z_tables:
                metrics["plasticity_requested_repr_layers"] = float(len(all_z_tables))
        z_tables = all_z_tables
        rep_source = self.plasticity_rep_source()
        if rep_source == "icl":
            base_layer_names = {f"icl{idx}" for idx in self.plasticity_reg_layer_indices()}
            z_tables = [(repr_name, z_table) for repr_name, z_table in all_z_tables if repr_name in base_layer_names]
        elif rep_source == "row":
            z_tables = [(repr_name, z_table) for repr_name, z_table in all_z_tables if repr_name == "row"]
        if z_tables:
            metrics["plasticity_repr_layers"] = float(len(z_tables))
        reference_z_tables = {}
        if scale > 0.0 and cka_weight > 0.0:
            metrics["plasticity_cka_reference_active"] = float(reference_representations is not None)
            reference_z_tables = {
                repr_name: z_table
                for repr_name, z_table in self.plasticity_table_representations(reference_representations, train_size)
            }

        cos_losses: list[Tensor] = []
        var_losses: list[Tensor] = []
        cka_losses: list[Tensor] = []
        var_floor = float(getattr(self.config, "plasticity_var_floor", 0.005) or 0.0)
        for repr_name, z_table in z_tables:
            if z_table.shape[0] < 2:
                continue
            z_centered = z_table - z_table.mean(dim=0, keepdim=True)
            table_var = z_centered.var(dim=0, unbiased=False)
            metrics[f"plasticity_{repr_name}_table_var_mean"] = float(table_var.detach().mean().item())

            if cos_weight > 0.0:
                z_norm = F.normalize(z_centered, dim=-1, eps=1e-6)
                sim = z_norm @ z_norm.transpose(0, 1)
                offdiag = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
                cos_loss = sim[offdiag].pow(2).mean()
                cos_losses.append(cos_loss)
                metrics[f"plasticity_{repr_name}_cos"] = float(cos_loss.detach().item())

            if var_weight > 0.0:
                var_loss = F.relu(var_floor - table_var).mean()
                var_losses.append(var_loss)
                metrics[f"plasticity_{repr_name}_var"] = float(var_loss.detach().item())

            if cka_weight > 0.0 and repr_name in reference_z_tables:
                reference_z_table = reference_z_tables[repr_name]
                if reference_z_table.shape == z_table.shape:
                    cka_loss, cka = self.plasticity_linear_cka_loss(z_table, reference_z_table)
                    cka_losses.append(cka_loss)
                    metrics[f"plasticity_{repr_name}_cka"] = float(cka.detach().item())
                    metrics[f"plasticity_{repr_name}_cka_loss"] = float(cka_loss.detach().item())

        low_high_cka_losses: list[Tensor] = []
        if low_high_cka_weight > 0.0 and all_z_tables:
            z_table_by_name = {repr_name: z_table for repr_name, z_table in all_z_tables}
            teacher_layers = self.plasticity_low_high_teacher_layers()
            student_layers = self.plasticity_low_high_student_layers()
            teacher_zs: list[Tensor] = []
            for layer_idx in teacher_layers:
                z_teacher = z_table_by_name.get(f"icl{layer_idx}")
                if z_teacher is not None and z_teacher.shape[0] >= 2:
                    teacher_zs.append(z_teacher)
            if teacher_zs:
                teacher_shape = teacher_zs[0].shape
                teacher_zs = [z for z in teacher_zs if z.shape == teacher_shape]
                teacher_ref = torch.stack(teacher_zs, dim=0).mean(dim=0)
                detach_teacher = bool(getattr(self.config, "plasticity_low_high_detach_teacher", True))
                metrics["plasticity_low_high_teacher_layers"] = float(len(teacher_zs))
                metrics["plasticity_low_high_student_layers"] = float(len(student_layers))
                for layer_idx in student_layers:
                    z_student = z_table_by_name.get(f"icl{layer_idx}")
                    if z_student is None or z_student.shape != teacher_ref.shape or z_student.shape[0] < 2:
                        continue
                    cka_loss, cka = self.plasticity_linear_cka_loss(
                        z_student,
                        teacher_ref,
                        detach_reference=detach_teacher,
                    )
                    low_high_cka_losses.append(cka_loss)
                    metrics[f"plasticity_icl{layer_idx}_low_high_cka"] = float(cka.detach().item())
                    metrics[f"plasticity_icl{layer_idx}_low_high_cka_loss"] = float(cka_loss.detach().item())

        if cos_losses:
            cos_loss = torch.stack(cos_losses).mean()
            terms.append((scale * cos_weight) * cos_loss)
            metrics["plasticity_cos"] = float(cos_loss.detach().item())
            metrics["plasticity_cos_weight"] = scale * cos_weight
        if var_losses:
            var_loss = torch.stack(var_losses).mean()
            terms.append((scale * var_weight) * var_loss)
            metrics["plasticity_var"] = float(var_loss.detach().item())
            metrics["plasticity_var_weight"] = scale * var_weight
            metrics["plasticity_var_floor"] = var_floor
        if cka_losses:
            cka_loss = torch.stack(cka_losses).mean()
            terms.append((scale * cka_weight) * cka_loss)
            metrics["plasticity_cka_loss"] = float(cka_loss.detach().item())
            metrics["plasticity_cka"] = float((1.0 - cka_loss.detach()).item())
            metrics["plasticity_cka_weight"] = scale * cka_weight
        if low_high_cka_losses:
            low_high_cka_loss = torch.stack(low_high_cka_losses).mean()
            terms.append((scale * low_high_cka_weight) * low_high_cka_loss)
            metrics["plasticity_low_high_cka_loss"] = float(low_high_cka_loss.detach().item())
            metrics["plasticity_low_high_cka"] = float((1.0 - low_high_cka_loss.detach()).item())
            metrics["plasticity_low_high_cka_weight"] = scale * low_high_cka_weight

        if scale > 0.0 and conf_weight > 0.0:
            threshold = float(getattr(self.config, "plasticity_conf_threshold", 0.80) or 0.0)
            max_prob = F.softmax(logits.float(), dim=-1).max(dim=-1).values
            conf_loss = F.relu(max_prob - threshold).pow(2).mean()
            terms.append((scale * conf_weight) * conf_loss)
            metrics["plasticity_conf"] = float(conf_loss.detach().item())
            metrics["plasticity_conf_weight"] = scale * conf_weight
            metrics["plasticity_conf_pmax_mean"] = float(max_prob.detach().mean().item())
            metrics["plasticity_conf_threshold"] = threshold

        if scale > 0.0 and entropy_floor_weight > 0.0:
            entropy_floor = float(getattr(self.config, "plasticity_entropy_floor", 0.40) or 0.0)
            num_classes = int(logits.shape[-1])
            support_labels = y_train.long().clamp(min=0, max=max(0, num_classes - 1))
            valid_class_mask = torch.zeros(
                y_train.shape[0],
                num_classes,
                dtype=torch.bool,
                device=logits.device,
            )
            valid_class_mask.scatter_(1, support_labels.to(logits.device), True)
            valid_class_count = valid_class_mask.sum(dim=-1)
            active_table = valid_class_count > 1
            if bool(active_table.any().detach().item()):
                masked_logits = logits.float().masked_fill(
                    ~valid_class_mask[:, None, :],
                    torch.finfo(logits.float().dtype).min,
                )
                probs = F.softmax(masked_logits, dim=-1)
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                denom = valid_class_count.float().log().clamp_min(1e-12)
                entropy_norm = entropy / denom[:, None]
                active_entropy_norm = entropy_norm[active_table]
                entropy_floor_loss = F.relu(entropy_floor - active_entropy_norm).pow(2).mean()
                terms.append((scale * entropy_floor_weight) * entropy_floor_loss)
                metrics["plasticity_entropy_floor_loss"] = float(entropy_floor_loss.detach().item())
                metrics["plasticity_entropy_floor_weight"] = scale * entropy_floor_weight
                metrics["plasticity_entropy_floor"] = entropy_floor
                metrics["plasticity_entropy_norm_mean"] = float(active_entropy_norm.detach().mean().item())
                metrics["plasticity_entropy_active_table_rate"] = float(active_table.float().mean().item())

        proto_terms, proto_metrics = self.plasticity_support_prototype_loss(representations, train_size, proto_scale)
        terms.extend(proto_terms)
        metrics.update(proto_metrics)

        attn_terms, attn_metrics = self.plasticity_attention_anti_collapse_loss(
            representations,
            train_size,
            attn_scale,
        )
        terms.extend(attn_terms)
        metrics.update(attn_metrics)

        if not terms:
            return None, metrics
        return sum(terms), metrics

    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches, timings=None):
        """T25 continuous regression supervision; native G5SC outer update loop."""
        return run_regression_micro_batch(
            self, micro_batch, micro_batch_idx, num_micro_batches, timings
        )
    @staticmethod
    def _choice_indices(rng: np.random.Generator, indices: np.ndarray, count: int, exclude=None) -> np.ndarray:
        if count <= 0:
            return np.empty((0,), dtype=np.int64)
        pool = np.asarray(indices, dtype=np.int64)
        if exclude is not None and len(exclude) > 0:
            pool = np.setdiff1d(pool, np.asarray(exclude, dtype=np.int64), assume_unique=False)
        if pool.size == 0:
            pool = np.asarray(indices, dtype=np.int64)
        replace = pool.size < count
        return rng.choice(pool, size=count, replace=replace).astype(np.int64, copy=False)

    def _sample_firewall_kd_task(self):
        data = self.firewall_kd_data
        if data is None:
            return None

        x_all = data["X"]
        y_all = data["y"]
        class_indices = data["class_indices"]
        all_indices = np.arange(y_all.shape[0], dtype=np.int64)
        support_rows = int(getattr(self.config, "firewall_kd_support_rows", 8192) or 8192)
        query_rows = int(getattr(self.config, "firewall_kd_query_rows", 512) or 512)
        min_per_class = max(0, int(getattr(self.config, "firewall_kd_min_support_per_class", 4) or 0))

        seed = (
            int(getattr(self.config, "np_seed", 42))
            + 1000003 * int(self.curr_step + 1)
            + 9176 * int(self.ddp_rank + 1)
        )
        rng = np.random.default_rng(seed)

        support_pieces = []
        if min_per_class > 0:
            per_class_cap = max(0, support_rows // max(1, len(class_indices)))
            per_class = min(min_per_class, per_class_cap)
            for indices in class_indices.values():
                support_pieces.append(self._choice_indices(rng, indices, min(per_class, support_rows)))
        support_seed = np.concatenate(support_pieces) if support_pieces else np.empty((0,), dtype=np.int64)
        remaining_support = max(0, support_rows - int(support_seed.size))
        support_rest = self._choice_indices(rng, all_indices, remaining_support, exclude=support_seed)
        support_idx = np.concatenate([support_seed, support_rest]).astype(np.int64, copy=False)
        rng.shuffle(support_idx)

        class2_fraction = float(getattr(self.config, "firewall_kd_query_class2_fraction", 0.5) or 0.0)
        class2_fraction = max(0.0, min(1.0, class2_fraction))
        query_class2 = int(round(query_rows * class2_fraction))
        query_class2 = max(0, min(query_rows, query_class2))
        support_unique = np.unique(support_idx)
        q2_idx = self._choice_indices(rng, class_indices[2], query_class2, exclude=support_unique)
        non2_idx = all_indices[y_all != 2]
        q_rest_idx = self._choice_indices(
            rng,
            non2_idx,
            query_rows - int(q2_idx.size),
            exclude=np.concatenate([support_unique, q2_idx]),
        )
        query_idx = np.concatenate([q2_idx, q_rest_idx]).astype(np.int64, copy=False)
        rng.shuffle(query_idx)

        preprocessor = PreprocessingPipeline(
            normalization_method=str(getattr(self.config, "firewall_kd_norm_method", "none") or "none"),
            outlier_threshold=4.0,
            random_state=seed,
        )
        x_support_raw = x_all[support_idx]
        x_query_raw = x_all[query_idx]
        preprocessor.fit(x_support_raw)
        x_support = preprocessor.X_transformed_.astype(np.float32, copy=False)
        x_query = preprocessor.transform(x_query_raw).astype(np.float32, copy=False)
        y_support = y_all[support_idx].astype(np.int64, copy=False)
        y_query = y_all[query_idx].astype(np.int64, copy=False)

        return x_support, y_support, x_query, y_query

    def _should_run_firewall_kd(self) -> bool:
        if not self.firewall_kd_enabled():
            return False
        interval = max(1, int(getattr(self.config, "firewall_kd_interval", 1) or 1))
        if self.curr_step % interval != 0:
            return False
        prob = float(getattr(self.config, "firewall_kd_prob", 1.0) or 0.0)
        if prob >= 1.0:
            return True
        if prob <= 0.0:
            return False
        # Keep the run/skip decision identical on every DDP rank.
        rng = np.random.default_rng(int(getattr(self.config, "np_seed", 42)) + 7919 * int(self.curr_step + 1))
        return bool(rng.random() < prob)

    def run_firewall_kd_auxiliary(self):
        """Add a small real internet_firewall heldout loss before the optimizer step."""

        if not self._should_run_firewall_kd():
            return {}

        sampled = self._sample_firewall_kd_task()
        if sampled is None:
            return {}

        x_support, y_support, x_query, y_query = sampled
        x_seq_np = np.concatenate([x_support, x_query], axis=0)
        x_seq = torch.tensor(x_seq_np, dtype=torch.float32, device=self.config.device).unsqueeze(0)
        y_train = torch.tensor(y_support, dtype=torch.long, device=self.config.device).unsqueeze(0)
        y_true = torch.tensor(y_query, dtype=torch.long, device=self.config.device)
        d = torch.tensor([x_seq.shape[-1]], dtype=torch.long, device=self.config.device)
        model_d = None if getattr(self.raw_model, "col_feature_group", False) else d

        if self.ddp:
            self.model.require_backward_grad_sync = True

        with self.amp_ctx:
            student_logits = self.model(x_seq, y_train, model_d).squeeze(0)
            teacher_logits = None
            if float(getattr(self.config, "firewall_kd_weight", 0.0) or 0.0) > 0.0:
                if self.stage3_teacher_model is None:
                    raise NonFiniteMicroBatchError("firewall KD requested but no Stage-3 teacher model is loaded")
                with torch.no_grad():
                    teacher_logits = self.stage3_teacher_model(x_seq, y_train, model_d).squeeze(0)

            shared_classes = int(student_logits.shape[-1])
            if teacher_logits is not None:
                shared_classes = min(shared_classes, int(teacher_logits.shape[-1]))
                student_kd_logits = student_logits[..., :shared_classes]
                teacher_logits = teacher_logits[..., :shared_classes]
            else:
                student_kd_logits = student_logits

            y_weights = torch.ones_like(y_true, dtype=torch.float32)
            class2_weight = float(getattr(self.config, "firewall_kd_class2_weight", 3.0) or 1.0)
            if class2_weight != 1.0:
                y_weights = torch.where(y_true == 2, y_weights * class2_weight, y_weights)
            weight_den = y_weights.sum().clamp_min(1.0)

            kd_loss = None
            if teacher_logits is not None:
                temperature = float(getattr(self.config, "firewall_kd_temperature", 2.0) or 2.0)
                kd_per_row = F.kl_div(
                    F.log_softmax(student_kd_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="none",
                ).sum(dim=-1) * (temperature * temperature)
                kd_loss = (kd_per_row * y_weights).sum() / weight_den

            ce_loss = None
            ce_weight = float(getattr(self.config, "firewall_kd_ce_weight", 0.0) or 0.0)
            if ce_weight > 0.0:
                label_smoothing = float(getattr(self.config, "label_smoothing", 0.0) or 0.0)
                ce_per_row = F.cross_entropy(
                    student_logits, y_true, reduction="none", label_smoothing=label_smoothing
                )
                ce_loss = (ce_per_row * y_weights).sum() / weight_den

            loss_terms = []
            kd_weight = float(getattr(self.config, "firewall_kd_weight", 0.0) or 0.0)
            if kd_loss is not None and kd_weight > 0.0:
                loss_terms.append(kd_weight * kd_loss)
            if ce_loss is not None and ce_weight > 0.0:
                loss_terms.append(ce_weight * ce_loss)
            if not loss_terms:
                return {}
            aux_loss = sum(loss_terms)

        loss_issue = self._nonfinite_tensor_reason("firewall_aux_loss", aux_loss)
        if self._sync_nonfinite_issue(loss_issue):
            raise NonFiniteMicroBatchError(loss_issue or "another rank reported invalid firewall auxiliary loss")

        self.scaler.scale(aux_loss).backward()

        with torch.no_grad():
            pred = student_logits.argmax(dim=-1)
            class2_mask = y_true == 2
            class2_recall = (
                (pred[class2_mask] == 2).float().mean()
                if bool(class2_mask.any())
                else torch.tensor(float("nan"), device=self.config.device)
            )
            result = {
                "firewall_aux_used": 1.0,
                "firewall_query_class2_rate": float(class2_mask.float().mean().item()),
                "firewall_class2_recall": float(class2_recall.item()),
            }
            if kd_loss is not None:
                result["firewall_kd"] = float(kd_loss.detach().item())
                result["firewall_kd_weight"] = kd_weight
            if ce_loss is not None:
                result["firewall_ce"] = float(ce_loss.detach().item())
                result["firewall_ce_weight"] = ce_weight
        return result

    def run_batch(self, batch):
        """Train the model on a batch of datasets.

        Handles gradient accumulation by splitting the batch into micro-batches.
        Supports variable-sized datasets by padding. Skips micro-batches on CUDA
        OOM errors. Updates model parameters and returns loss and accuracy metrics.

        Parameters
        ----------
        batch : tuple
            Contains tensors (X, y, d, seq_len, train_size) for the batch.
            X and y can be Tensors or NestedTensors (for variable sequence
            lengths).

        Returns
        -------
        dict
            Dictionary containing 'ce' (cross-entropy loss) and 'accuracy'.

        Raises
        ------
        RuntimeError
            If more than 10% of micro-batches fail due to OOM errors.
        """
        profile_timings = {} if self._profile_timing_enabled() else None

        if bool(getattr(self.config, "set_train_every_step", True)):
            self.model.train()
        if bool(getattr(self.config, "zero_grad_begin", True)):
            with self._timed_phase(profile_timings, "zero_grad_begin"):
                self.optimizer.zero_grad(set_to_none=True)
        self.debug_log("run_batch begin")

        source_metadata = None
        if len(batch) == 7:
            batch_tensors = list(batch[:6])
            source_metadata = batch[6]
        elif len(batch) == 6 and isinstance(batch[5], list):
            batch_tensors = list(batch[:5])
            source_metadata = batch[5]
        else:
            batch_tensors = list(batch)

        # Pad nested tensors to the same size
        with self._timed_phase(profile_timings, "batch_pad_split"):
            batch_tensors = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch_tensors]
            self._maybe_write_paired_batch_audit(batch_tensors[:5])
            self._maybe_log_batch_source(batch_tensors[:5], source_metadata)

            # Split the batch into micro-batches along the first dimension
            num_micro_batches = math.ceil(self.config.batch_size / self.config.micro_batch_size)
            if num_micro_batches <= 1:
                micro_batch = tuple(batch_tensors)
                if source_metadata is not None:
                    micro_batch = micro_batch + (source_metadata,)
                micro_batches = [micro_batch]
            else:
                micro_batches = [torch.split(t, self.config.micro_batch_size, dim=0) for t in batch_tensors]
                micro_batches = list(zip(*micro_batches))
                metadata_chunks = self._split_metadata(source_metadata, self.config.micro_batch_size)
                if metadata_chunks is not None and len(metadata_chunks) == len(micro_batches):
                    micro_batches = [
                        tuple(micro_batch) + (metadata_chunks[idx],)
                        for idx, micro_batch in enumerate(micro_batches)
                    ]
        self.debug_log(f"run_batch split_done num_micro_batches={len(micro_batches)}")

        results = {
            "successful_micro_batches": 0,
            "nonfinite_micro_batches": 0,
            "oom_micro_batches": 0,
            "skipped_update": 0,
        }
        if self._train_metrics_enabled():
            results.update({"pinball": 0.0, "mse": 0.0})
        oom_batches = 0
        nonfinite_batches = 0
        last_micro_batch_succeeded = False

        for idx, micro_batch in enumerate(micro_batches):
            try:
                self.debug_log(f"micro {idx + 1}/{num_micro_batches} begin")
                with self._timed_phase(profile_timings, "micro_total"):
                    micro_results = self.run_micro_batch(micro_batch, idx, num_micro_batches, profile_timings)
                for k, v in micro_results.items():
                    results[k] = results.get(k, 0.0) + v
                results["successful_micro_batches"] += 1
                if idx == len(micro_batches) - 1:
                    last_micro_batch_succeeded = True
                self.debug_log(f"micro {idx + 1}/{num_micro_batches} done")
            except NonFiniteMicroBatchError as exc:
                if getattr(self.config, "abort_on_nonfinite_batch", False) or not getattr(
                    self.config, "skip_nonfinite_batches", True
                ):
                    raise
                self.warning_log(f"Skipping non-finite micro-batch {idx+1}/{num_micro_batches}: {exc}")
                torch.cuda.empty_cache()
                nonfinite_batches += 1
                results["nonfinite_micro_batches"] += 1
                continue
            except torch.cuda.OutOfMemoryError:
                print(
                    f"Warning: OOM error in micro-batch {idx+1}/{num_micro_batches} at step {self.curr_step}. Skipping."
                )
                torch.cuda.empty_cache()
                oom_batches += 1
                results["oom_micro_batches"] += 1
                continue

        oom_ratio = oom_batches / num_micro_batches
        if oom_ratio > 0.1:
            raise RuntimeError(
                f"({oom_ratio:.1%}) of micro-batches failed due to OOM at step {self.curr_step}. "
                f"Please check configuration to reduce memory consumption."
            )

        if results["successful_micro_batches"] == 0:
            self.warning_log("Skipping optimizer update because no finite micro-batches succeeded.")
            self.optimizer.zero_grad(set_to_none=True)
            results["skipped_update"] = 1
            return results

        if self.ddp and not last_micro_batch_succeeded:
            self.warning_log("Skipping optimizer update because the final micro-batch did not run DDP gradient sync.")
            self.optimizer.zero_grad(set_to_none=True)
            results["skipped_update"] = 1
            return results

        bad_fraction = nonfinite_batches / num_micro_batches
        max_bad_fraction = float(getattr(self.config, "nonfinite_max_bad_micro_batch_fraction", 0.1))
        if bad_fraction > max_bad_fraction:
            self.warning_log(
                f"Skipping optimizer update because non-finite micro-batch fraction {bad_fraction:.1%} "
                f"exceeds limit {max_bad_fraction:.1%}."
            )
            self.optimizer.zero_grad(set_to_none=True)
            results["skipped_update"] = 1
            return results

        with self._timed_phase(profile_timings, "stage3_l2sp"):
            l2sp_loss = self.stage3_l2sp_loss()
            if l2sp_loss is not None:
                l2sp_weight = float(getattr(self.config, "stage3_l2sp_weight", 0.0) or 0.0)
                self.scaler.scale(l2sp_weight * l2sp_loss).backward()
                results["stage3_l2sp"] = float(l2sp_loss.detach().item())
                results["stage3_l2sp_weight"] = l2sp_weight

        with self._timed_phase(profile_timings, "firewall_aux"):
            firewall_aux_results = self.run_firewall_kd_auxiliary()
            if firewall_aux_results:
                results.update(firewall_aux_results)

        self.debug_log("unscale_grad begin")
        with self._timed_phase(profile_timings, "unscale_grad"):
            self.scaler.unscale_(self.optimizer)
        self.debug_log("unscale_grad done")

        with self._timed_phase(profile_timings, "late_icl_freeze"):
            late_icl_freeze_metrics = self.apply_late_icl_freeze_to_grads()
            if late_icl_freeze_metrics:
                results.update(late_icl_freeze_metrics)

        with self._timed_phase(profile_timings, "grad_nonfinite_check"):
            grad_issue = self._local_nonfinite_grad_reason()
        if self._sync_nonfinite_issue(grad_issue):
            self.warning_log(f"Skipping optimizer update due to non-finite gradient: {grad_issue or 'another rank'}")
            self.optimizer.zero_grad(set_to_none=True)
            try:
                self.scaler.update()
            except AssertionError:
                pass
            results["nonfinite_grad"] = 1
            results["skipped_update"] = 1
            return results

        collect_train_metrics = self._train_metrics_enabled()
        with self._timed_phase(profile_timings, "module_grad_norms"):
            module_grad_norms = self._module_grad_norms() if self._should_log_grad_norm() else {}

        max_norm = float(self.config.gradient_clipping)
        clip_error = None
        total_norm = None
        if max_norm > 0.0:
            self.debug_log("clip_grad begin")
            with self._timed_phase(profile_timings, "clip_grad"):
                try:
                    total_norm_tensor = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm,
                        error_if_nonfinite=bool(getattr(self.config, "error_if_nonfinite_grad", True)),
                        foreach=self._gradient_clip_foreach(),
                    )
                    if collect_train_metrics or self._should_log_grad_norm():
                        total_norm = float(
                            total_norm_tensor.item() if torch.is_tensor(total_norm_tensor) else total_norm_tensor
                        )
                except RuntimeError as exc:
                    clip_error = str(exc)
            self.debug_log("clip_grad done")

        if self._gradient_clip_error_sync_enabled():
            clip_failed = self._sync_bad_flag(clip_error is not None)
        else:
            if clip_error is not None:
                raise RuntimeError(
                    "Gradient clipping failed while cross-rank clip-error synchronization is disabled: "
                    f"{clip_error}"
                )
            clip_failed = False

        if clip_failed:
            self.warning_log(f"Skipping optimizer update because clip_grad_norm_ failed: {clip_error or 'another rank'}")
            self.optimizer.zero_grad(set_to_none=True)
            try:
                self.scaler.update()
            except AssertionError:
                pass
            results["nonfinite_grad"] = 1
            results["skipped_update"] = 1
            return results

        if total_norm is not None:
            results["grad_norm_total"] = total_norm
            self._log_module_grad_norms(module_grad_norms, total_norm)

        # Update parameters
        self.debug_log("optimizer step begin")
        down_bootstrap_restore = self.activate_swiglu_down_bootstrap()
        with self._timed_phase(profile_timings, "optimizer_step"):
            try:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            finally:
                self.restore_swiglu_down_bootstrap(down_bootstrap_restore)
        if down_bootstrap_restore:
            results["swiglu_down_lr_bootstrap_multiplier"] = float(
                getattr(self.config, "swiglu_down_lr_bootstrap_multiplier", 1.0)
            )
        with self._timed_phase(profile_timings, "stage3_anchor_ema"):
            anchor_pullback_rms = self.apply_stage3_anchor_pullback()
            if anchor_pullback_rms is not None:
                results["stage3_anchor_pullback"] = float(getattr(self.config, "stage3_anchor_pullback", 0.0) or 0.0)
                results["stage3_anchor_rms_before_pullback"] = float(anchor_pullback_rms.detach().item())
            if self.update_stage3_ema():
                results["stage3_ema_decay"] = float(getattr(self.config, "stage3_ema_decay", 0.0) or 0.0)
                results["stage3_ema_checkpoint"] = float(
                    bool(getattr(self.config, "stage3_save_ema_checkpoints", False))
                )
        with self._timed_phase(profile_timings, "continual_bp"):
            continual_bp_metrics = self.continual_bp_step()
            if continual_bp_metrics:
                results.update(continual_bp_metrics)
        self.debug_log("optimizer step done")

        # Update the learning rate
        with self._timed_phase(profile_timings, "zero_grad_end"):
            self.optimizer.zero_grad(set_to_none=True)
        with self._timed_phase(profile_timings, "scheduler_step"):
            self.scheduler.step()
        self.debug_log("scheduler step done")

        if profile_timings is not None:
            results.update({f"timing_{name}": value for name, value in profile_timings.items()})

        return results


if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()

    try:
        # Set the start method for subprocesses to 'spawn'
        set_start_method("spawn")
    except RuntimeError:
        pass  # Ignore the error if the context has already been set

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()
