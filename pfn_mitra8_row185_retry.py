#!/usr/bin/env python3
"""Row185-only Mitra8 retry: split SDPA's independent batch axis, never context.

Frozen inference/data/ensemble code is imported without changing its bytes.
Only this process's functional SDPA is wrapped; all K/V tokens, outer query
chunks, initialization validation and RNG progression remain unchanged.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys

import pfn_mitra8_one as base

BASE_SHA = "e9a00d9be77c3302a662b956d9f18415f3c52d4e4edf16462b58d7d7e1a70c49"
TAB2D_SHA = "a3f4b9ef6fa72ab870c66105683493bd32c4ecaced3af42601146b73b0887826"
BATCH_CHUNK = 8
require = base.require


class BatchChunkSDPA:
    def __init__(self, torch, original):
        self.torch, self.original = torch, original
        self.phase = "parity"
        self.calls = 0
        self.chunked_calls = 0
        self.kernel_calls = 0
        self.shapes = {}

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, *, scale=None, enable_gqa=False):
        require(query.ndim == key.ndim == value.ndim == 4, "Retry SDPA requires native four-dimensional Q/K/V")
        require(query.shape[0] == key.shape[0] == value.shape[0] > 0,
                "Retry SDPA requires equal, nonempty independent batch axes")
        require(query.shape[1] == key.shape[1] == value.shape[1]
                and key.shape[2] == value.shape[2] and query.shape[3] == key.shape[3],
                "Unexpected native SDPA head/key dimensions")
        require(attn_mask is None and dropout_p == 0.0 and not is_causal
                and scale is None and not enable_gqa,
                "Retry only supports original unmasked/default-scale/no-dropout/noncausal SDPA")
        require(not self.torch.is_grad_enabled(), "Retry SDPA is inference-only")
        batch = query.shape[0]
        count = math.ceil(batch / BATCH_CHUNK)
        if self.phase == "evaluation":
            self.calls += 1
            self.chunked_calls += int(count > 1)
            self.kernel_calls += count
            shape_key = json.dumps([list(query.shape), list(key.shape), list(value.shape)])
            if shape_key not in self.shapes:
                require(len(self.shapes) < 128, "Unexpectedly many native SDPA shapes")
                self.shapes[shape_key] = {"query": list(query.shape), "key": list(key.shape),
                    "value": list(value.shape), "dtype": str(query.dtype),
                    "batch_chunks_per_call": count, "calls": 0}
                print(json.dumps({"row185_native_sdpa_shape": self.shapes[shape_key]}), flush=True)
            self.shapes[shape_key]["calls"] += 1
        try:
            if count == 1:
                return self.original(query, key, value)
            parts = [self.original(query[start:start+BATCH_CHUNK],
                                   key[start:start+BATCH_CHUNK], value[start:start+BATCH_CHUNK])
                     for start in range(0, batch, BATCH_CHUNK)]
            return self.torch.cat(parts, dim=0)
        except self.torch.cuda.OutOfMemoryError as exc:
            # The native estimator catches OOM and shrinks context. Preserve
            # the causal stack, but forbid that protocol-changing fallback.
            raise RuntimeError("Row185 SDPA batch-only retry OOM; support/query budgets must not shrink") from exc

    def audit(self):
        return {"native_sdpa_calls": self.calls, "chunked_native_sdpa_calls": self.chunked_calls,
                "actual_sdpa_kernel_calls": self.kernel_calls, "native_shapes": list(self.shapes.values())}


def close_record(torch, expected, actual, *, kind, device, dtype):
    require(expected.shape == actual.shape and bool(torch.isfinite(actual).all()), "Nonfinite/reshaped parity output")
    bf16 = dtype == torch.bfloat16
    rtol, atol = (0.01, 0.01) if bf16 else (1e-5, 2e-6)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return {"kind": kind, "device": str(device), "dtype": str(dtype), "passed": True,
            "shape": list(actual.shape), "maximum_absolute_error": float((actual.float()-expected.float()).abs().max()),
            "rtol": rtol, "atol": atol, "bitwise_equal": bool(torch.equal(actual, expected))}


def sdpa_parity(torch, wrapped):
    records = []
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Single visible ROCm GPU required")
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        for device in ("cpu", "cuda:0"):
            for dtype in (torch.float32, torch.bfloat16):
                q = torch.randn((17, 2, 7, 16), device=device, dtype=dtype)
                k = torch.randn((17, 2, 11, 16), device=device, dtype=dtype)
                v = torch.randn((17, 2, 11, 16), device=device, dtype=dtype)
                records.append(close_record(torch, wrapped.original(q, k, v), wrapped(q, k, v),
                    kind="functional_sdpa", device=device, dtype=dtype))
    return records


def full_model_parity(torch, functional, wrapped, model):
    """Same loaded checkpoint; preserve CPU/CUDA RNG and module training flags."""
    modes = [(module, module.training) for module in model.modules()]
    records = []
    try:
        with torch.random.fork_rng(devices=[0]), torch.no_grad():
            model.eval()
            xs = torch.randn((1, 32, 17), device="cuda:0")
            ys = torch.rand((1, 32), device="cuda:0")
            xq = torch.randn((1, 11, 17), device="cuda:0")
            inputs = (xs, ys, xq, torch.zeros((1, 17), dtype=torch.bool, device="cuda:0"),
                      torch.zeros((1, 32), dtype=torch.bool, device="cuda:0"),
                      torch.zeros((1, 11), dtype=torch.bool, device="cuda:0"))
            for dtype in (torch.float32, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                    functional.scaled_dot_product_attention = wrapped.original
                    expected = model(*inputs)
                    functional.scaled_dot_product_attention = wrapped
                    actual = model(*inputs)
                records.append(close_record(torch, expected, actual, kind="full_native_checkpoint_model",
                                            device="cuda:0", dtype=dtype))
    finally:
        functional.scaled_dot_product_attention = wrapped
        for module, mode in modes:
            module.training = mode
    return records


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    require(args.dataset_index == 185, "This retry is authorized only for row185")
    require(base.sha256_file(base.__file__) == BASE_SHA, "Frozen Mitra8 worker changed")
    require(os.environ.get("MITRA_RETRY_TAB2D_SHA256") == TAB2D_SHA,
            "Explicit audited native tab2d SHA is required")
    manifest = json.loads(args.manifest.read_text())
    row = next(r for r in manifest["rows"] if r["dataset_index"] == 185)
    require(row["dataset"] == "TabArena__QSAR-TID-11" and row["suite"] == "TabArena",
            "Row185 membership changed")
    original_load, original_publish = base.load_runtime, base.common.publish_new
    state = {"parity": [], "full_model_parity_done": False}

    def load_runtime(stage, weights_manifest):
        import torch
        import torch.nn.functional as functional
        native = importlib.import_module("autogluon.tabular.models.mitra._internal.models.tab2d")
        native_path = Path(native.__file__).resolve()
        require(base.sha256_file(native_path) == TAB2D_SHA, "Native tab2d source changed")
        wrapped = BatchChunkSDPA(torch, functional.scaled_dot_product_attention)
        state.update(torch=torch, functional=functional, wrapped=wrapped,
                     native_source={"path": str(native_path), "sha256": TAB2D_SHA})
        state["parity"].extend(sdpa_parity(torch, wrapped))
        functional.scaled_dot_product_attention = wrapped
        estimator, weight, sources = original_load(stage, weights_manifest)
        si = importlib.import_module("autogluon.tabular.models.mitra.sklearn_interface")
        guarded = si.TrainerFinetune
        state.update(si=si, original_trainer=guarded)

        class RetryTrainer(guarded):
            def __init__(self, cfg, model, *a, **kw):
                if not state["full_model_parity_done"]:
                    state["parity"].extend(full_model_parity(torch, functional, wrapped, model))
                    state["full_model_parity_done"] = True
                    wrapped.phase = "evaluation"
                    print(json.dumps({"row185_parity": state["parity"]}), flush=True)
                try:
                    super().__init__(cfg, model, *a, **kw)
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError("Row185 native trainer initialization OOM; context fallback prohibited") from exc

            def train(self, *a, **kw):
                try:
                    return super().train(*a, **kw)
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError("Row185 full-support initialization OOM; context fallback prohibited") from exc
        si.TrainerFinetune = RetryTrainer
        return estimator, weight, sources

    def publish(path, result):
        require(result["dataset_index"] == 185 and result["dataset"] == "TabArena__QSAR-TID-11",
                "Retry attempted to publish another membership")
        require(result["worker_source_sha256"] == BASE_SHA and result["actual8_verified"] is True
                and result["actual_ensemble_count"] == 8 and state["full_model_parity_done"],
                "Base inference/full-model parity audit incomplete")
        require(result["data_audit"]["support_rows"] == 3828 and result["data_audit"]["test_rows"] == 1914
                and result["data_audit"]["features"] == 1024, "Frozen row185 dimensions changed")
        require(state["wrapped"].chunked_calls > 0, "Batch-axis implementation was never exercised")
        result["base_worker_source_sha256"] = result["worker_source_sha256"]
        result["worker_source_sha256"] = base.sha256_file(__file__)
        result["implementation_audit"] = {"retry": "row185_sdpa_independent_batch8_v1",
            "native_source": state["native_source"], "sdpa_batch_chunk": BATCH_CHUNK,
            "query_key_axes_unchanged": True, "native_query_cap_unchanged": 1024,
            "native_support_cap_unchanged": 8192, "full_initialization_validation_preserved": True,
            "model_weights_and_seed_unchanged": True, "rng_preserved_during_parity": True,
            "parity": state["parity"], "bitwise_equivalence_claim": False,
            **state["wrapped"].audit()}
        original_publish(path, result)

    base.load_runtime, base.common.publish_new = load_runtime, publish
    try:
        base.main(argv)
    finally:
        base.load_runtime, base.common.publish_new = original_load, original_publish
        if "wrapped" in state:
            state["functional"].scaled_dot_product_attention = state["wrapped"].original
        if "si" in state:
            state["si"].TrainerFinetune = state["original_trainer"]


if __name__ == "__main__":
    main()
