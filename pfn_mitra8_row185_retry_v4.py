#!/usr/bin/env python3
"""Row185-only Mitra8 retry: split SDPA's independent batch axis, never context.

Frozen inference/data/ensemble code is imported without changing its bytes.
Only this process's SDPA and pinned non-flash Layer.forward are wrapped;
all K/V tokens, outer query
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
import time

import pfn_mitra8_one as base

BASE_SHA = "e9a00d9be77c3302a662b956d9f18415f3c52d4e4edf16462b58d7d7e1a70c49"
TAB2D_SHA = "a3f4b9ef6fa72ab870c66105683493bd32c4ecaced3af42601146b73b0887826"
BATCH_CHUNK = 8
FFN_TOKEN_CHUNK = 8192
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



class StreamedNativeLayer:
    """Pinned non-flash Layer math with bounded FFN and scoped temporaries."""
    def __init__(self, torch, functional, native):
        self.torch, self.functional, self.native = torch, functional, native
        self.original = native.Layer.forward
        self.phase = "parity"
        self.layer_calls = 0
        self.ffn_calls = 0
        self.ffn_chunks = 0
        self.multichunk_ffn_calls = 0
        self.parity_multichunk_ffn_calls = 0
        self.shapes = {}
        controller = self
        def replacement(layer, support, query__, padder_support, padder_query__,
                        batch_size=None, padding_obs_support=None,
                        padding_obs_query__=None, padding_features=None):
            return controller.forward(layer, support, query__, padder_support, padder_query__,
                batch_size, padding_obs_support, padding_obs_query__, padding_features)
        self.replacement = replacement

    def install(self):
        self.native.Layer.forward = self.replacement

    def restore(self):
        self.native.Layer.forward = self.original

    def ffn(self, value, norm, linear_in, linear_out, name):
        require(value.is_contiguous() and value.ndim == 4, "Native residual token layout changed")
        shape, dtype = value.shape, value.dtype
        flat = value.view(-1, shape[-1])
        output = output_flat = None
        chunks = math.ceil(flat.shape[0] / FFN_TOKEN_CHUNK)
        if self.phase == "evaluation":
            self.ffn_calls += 1
            self.ffn_chunks += chunks
            self.multichunk_ffn_calls += int(chunks > 1)
            key = json.dumps([name, list(shape), str(dtype)])
            if key not in self.shapes:
                require(len(self.shapes) < 128, "Unexpected native FFN shapes")
                self.shapes[key] = {"block": name, "shape": list(shape), "dtype": str(dtype),
                                    "token_chunks_per_call": chunks, "calls": 0}
                print(json.dumps({"row185_streamed_ffn_shape": self.shapes[key]}), flush=True)
            self.shapes[key]["calls"] += 1
        else:
            self.parity_multichunk_ffn_calls += int(chunks > 1)
        for start in range(0, flat.shape[0], FFN_TOKEN_CHUNK):
            residual = flat[start:start+FFN_TOKEN_CHUNK]
            # Each original operation acts independently on the final dim.
            # Autocast, GELU approximation, addition order and dtype stay native.
            transformed = linear_out(self.functional.gelu(linear_in(norm(residual))))
            summed = residual + transformed
            require(summed.dtype == dtype, "Native residual output dtype changed")
            if output is None:
                output = self.torch.empty(shape, dtype=summed.dtype, device=summed.device)
                output_flat = output.view(-1, shape[-1])
            output_flat[start:start+FFN_TOKEN_CHUNK].copy_(summed)
            del residual, transformed, summed
        return output

    def row_attention(self, layer, support, query, batch):
        rearrange = self.native.einops.rearrange
        normalized_support = layer.layer_norm1(support)
        normalized_query = layer.layer_norm1(query)
        support_flat = rearrange(normalized_support, 'b s f d -> (b f) s d')
        query_flat = rearrange(normalized_query, 'b s f d -> (b f) s d')
        support_att = rearrange(layer.attention1(support_flat, support_flat, support_flat),
                                '(b f) s d -> b s f d', b=batch)
        query_att = rearrange(layer.attention1(query_flat, support_flat, support_flat),
                              '(b f) s d -> b s f d', b=batch)
        del normalized_support, normalized_query, support_flat, query_flat
        # Scope ends here: no flattened attention or normalization locals leak
        # into either FFN or the following attention stage.
        return support + support_att, query + query_att

    def feature_attention(self, layer, support, query, batch):
        rearrange = self.native.einops.rearrange
        normalized_support = layer.layer_norm3(support)
        normalized_query = layer.layer_norm3(query)
        support_flat = rearrange(normalized_support, 'b s f d -> (b s) f d')
        query_flat = rearrange(normalized_query, 'b s f d -> (b s) f d')
        support_att = rearrange(layer.attention2(support_flat, support_flat, support_flat),
                                '(b s) f d -> b s f d', b=batch)
        query_att = rearrange(layer.attention2(query_flat, query_flat, query_flat),
                              '(b s) f d -> b s f d', b=batch)
        del normalized_support, normalized_query, support_flat, query_flat
        return support + support_att, query + query_att

    def forward(self, layer, support, query, padder_support, padder_query,
                batch_size=None, padding_obs_support=None, padding_obs_query=None, padding_features=None):
        require(not layer.use_flash_attn and padder_support is None and padder_query is None,
                "Streamed retry only implements pinned non-flash native Layer")
        require(not self.torch.is_grad_enabled() and not layer.training,
                "Streamed native Layer is no-grad/eval only")
        require(support.ndim == query.ndim == 4 and support.shape[0] == query.shape[0]
                and support.shape[2:] == query.shape[2:], "Native Layer support/query schema changed")
        batch = support.shape[0] if batch_size is None else batch_size
        require(batch == support.shape[0], "Native batch size changed")
        expected_support, expected_query = tuple(support.shape), tuple(query.shape)
        dtype_support, dtype_query = support.dtype, query.dtype
        if self.phase == "evaluation":
            self.layer_calls += 1
        try:
            support, query = self.row_attention(layer, support, query, batch)
            support = self.ffn(support, layer.layer_norm2, layer.linear1, layer.linear2, "support_mlp1")
            query = self.ffn(query, layer.layer_norm2, layer.linear1, layer.linear2, "query_mlp1")
            support, query = self.feature_attention(layer, support, query, batch)
            support = self.ffn(support, layer.layer_norm4, layer.linear3, layer.linear4, "support_mlp2")
            query = self.ffn(query, layer.layer_norm4, layer.linear3, layer.linear4, "query_mlp2")
            require(tuple(support.shape) == expected_support and tuple(query.shape) == expected_query
                    and support.dtype == dtype_support and query.dtype == dtype_query,
                    "Streamed Layer changed native dimensions/dtypes")
            return support, query
        except self.torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("Row185 bounded native Layer OOM; context fallback prohibited") from exc

    def audit(self):
        return {"native_layer_calls": self.layer_calls, "ffn_token_chunk": FFN_TOKEN_CHUNK,
                "streamed_ffn_calls": self.ffn_calls, "streamed_ffn_chunks": self.ffn_chunks,
                "multi_chunk_ffn_calls": self.multichunk_ffn_calls,
                "parity_multi_chunk_ffn_calls": self.parity_multichunk_ffn_calls,
                "ffn_shapes": list(self.shapes.values()),
                "attention_norm_and_flat_temporaries_released_between_stages": True,
                "ffn_residual_addition_order_and_dtype_preserved": True,
                "layer_scope": "pinned native Layer.forward non-flash path only"}


def close_record(torch, expected, actual, *, kind, device, dtype):
    require(expected.shape == actual.shape and bool(torch.isfinite(actual).all()), "Nonfinite/reshaped parity output")
    bf16 = dtype == torch.bfloat16
    rtol, atol = (0.01, 0.01) if bf16 else (1e-5, 2e-6)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return {"kind": kind, "device": str(device), "dtype": str(dtype), "passed": True,
            "native_output_dtype": str(expected.dtype), "retry_output_dtype": str(actual.dtype),
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


def full_model_parity(torch, functional, wrapped, streamed, model):
    """Both patches disabled for native reference; two support/feature sizes."""
    modes = [(module, module.training) for module in model.modules()]
    records = []
    try:
        with torch.random.fork_rng(devices=[0]), torch.no_grad():
            model.eval()
            for support_rows, query_rows, features in ((32, 11, 17), (129, 129, 67)):
                xs = torch.randn((1, support_rows, features), device="cuda:0")
                ys = torch.rand((1, support_rows), device="cuda:0")
                xq = torch.randn((1, query_rows, features), device="cuda:0")
                inputs = (xs, ys, xq, torch.zeros((1, features), dtype=torch.bool, device="cuda:0"),
                          torch.zeros((1, support_rows), dtype=torch.bool, device="cuda:0"),
                          torch.zeros((1, query_rows), dtype=torch.bool, device="cuda:0"))
                for dtype in (torch.float32, torch.bfloat16):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                        functional.scaled_dot_product_attention = wrapped.original
                        streamed.restore()
                        expected = model(*inputs)
                        functional.scaled_dot_product_attention = wrapped
                        streamed.install()
                        actual = model(*inputs)
                    record = close_record(torch, expected, actual, kind="full_native_checkpoint_model",
                                          device="cuda:0", dtype=dtype)
                    record["fixture"] = {"support_rows": support_rows, "query_rows": query_rows,
                                         "features": features}
                    records.append(record)
                    del expected, actual
                del xs, ys, xq, inputs
            require(streamed.parity_multichunk_ffn_calls > 0,
                    "Full model parity never exercised multi-chunk token streaming")
    finally:
        functional.scaled_dot_product_attention = wrapped
        streamed.install()
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
    state = {"parity": [], "full_model_parity_done": False, "member_constructors": 0}

    def load_runtime(stage, weights_manifest):
        import torch
        import torch.nn.functional as functional
        native = importlib.import_module("autogluon.tabular.models.mitra._internal.models.tab2d")
        native_path = Path(native.__file__).resolve()
        require(base.sha256_file(native_path) == TAB2D_SHA, "Native tab2d source changed")
        wrapped = BatchChunkSDPA(torch, functional.scaled_dot_product_attention)
        streamed = StreamedNativeLayer(torch, functional, native)
        state["streamed"] = streamed
        state.update(torch=torch, functional=functional, wrapped=wrapped,
                     native_source={"path": str(native_path), "sha256": TAB2D_SHA})
        state["parity"].extend(sdpa_parity(torch, wrapped))
        functional.scaled_dot_product_attention = wrapped
        streamed.install()
        estimator, weight, sources = original_load(stage, weights_manifest)
        si = importlib.import_module("autogluon.tabular.models.mitra.sklearn_interface")
        guarded = si.TrainerFinetune
        state.update(si=si, original_trainer=guarded)

        class RetryTrainer(guarded):
            def __init__(self, cfg, model, *a, **kw):
                self.retry_member_ordinal = state["member_constructors"]
                state["member_constructors"] += 1
                try:
                    super().__init__(cfg, model, *a, **kw)
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError("Row185 native trainer initialization OOM; context fallback prohibited") from exc
                # Native Trainer moves the HF-loaded model onto cfg.device.
                # Run parity only after that exact constructor and before fit.
                if not state["full_model_parity_done"]:
                    state["parity"].extend(full_model_parity(torch, functional, wrapped, streamed, self.model))
                    state["full_model_parity_done"] = True
                    wrapped.phase = "evaluation"
                    streamed.phase = "evaluation"
                    print(json.dumps({"row185_parity": state["parity"]}), flush=True)

            def train(self, *a, **kw):
                started = time.monotonic()
                print(json.dumps({"row185_member_initialization": "start",
                                  "member": self.retry_member_ordinal}), flush=True)
                try:
                    result = super().train(*a, **kw)
                    print(json.dumps({"row185_member_initialization": "complete",
                        "member": self.retry_member_ordinal,
                        "elapsed_seconds": time.monotonic()-started}), flush=True)
                    return result
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError("Row185 full-support initialization OOM; context fallback prohibited") from exc

            def predict(self, *a, **kw):
                started = time.monotonic()
                print(json.dumps({"row185_member_prediction": "start",
                                  "member": self.retry_member_ordinal}), flush=True)
                result = super().predict(*a, **kw)
                print(json.dumps({"row185_member_prediction": "complete",
                    "member": self.retry_member_ordinal,
                    "elapsed_seconds": time.monotonic()-started}), flush=True)
                return result
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
        require(state["streamed"].layer_calls > 0 and state["streamed"].multichunk_ffn_calls > 0,
                "Native Layer/FFN streaming was not exercised")
        result["base_worker_source_sha256"] = result["worker_source_sha256"]
        result["worker_source_sha256"] = base.sha256_file(__file__)
        result["implementation_audit"] = {"retry": "row185_sdpa_batch8_token_streamed_native_layer_v4",
            "native_source": state["native_source"], "sdpa_batch_chunk": BATCH_CHUNK,
            "query_key_axes_unchanged": True, "native_query_cap_unchanged": 1024,
            "native_support_cap_unchanged": 8192, "full_initialization_validation_preserved": True,
            "model_weights_and_seed_unchanged": True, "rng_preserved_during_parity": True,
            "parity": state["parity"], "bitwise_equivalence_claim": False,
            **state["wrapped"].audit(), **state["streamed"].audit()}
        original_publish(path, result)

    base.load_runtime, base.common.publish_new = load_runtime, publish
    try:
        base.main(argv)
    finally:
        base.load_runtime, base.common.publish_new = original_load, original_publish
        if "wrapped" in state:
            state["functional"].scaled_dot_product_attention = state["wrapped"].original
        if "streamed" in state:
            state["streamed"].restore()
        if "si" in state:
            state["si"].TrainerFinetune = state["original_trainer"]


if __name__ == "__main__":
    main()

