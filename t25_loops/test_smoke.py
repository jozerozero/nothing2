#!/usr/bin/env python3
"""Small, source-pinned T25 G5SC regression-loop contract checks.

This test does not construct a trainer, generate prior batches, load external
checkpoints, or submit work.  A receipt is written only when explicitly requested.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path


def require(condition: bool, name: str, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}" if detail else name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    source_src = source / "src"
    require((source_src / "tabicl").is_dir(), "SOURCE_ROOT_MISSING", str(source))
    sys.path.insert(0, str(source_src))

    import torch
    from torch import nn

    import tabicl._model.tabicl as tabicl_module
    from tabicl._model.g5sc_regression_loop import GatedLoopEncoder, SupportSchemaStatistics
    from tabicl._model.kv_cache import KVCache, TabICLCache
    from tabicl._model.tabicl import TabICL
    from tabicl.train._muon import Muon

    require(
        Path(tabicl_module.__file__).resolve().is_relative_to(source_src),
        "WRONG_TABICL_SOURCE",
        str(tabicl_module.__file__),
    )
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Another import may already have initialized the inter-op pool.
        pass
    if args.device == "cuda":
        require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
        torch.empty(1, device="cuda")
    device = torch.device(args.device)

    config = {
        "max_classes": 0,
        "num_quantiles": 17,
        "embed_dim": 16,
        "col_num_blocks": 1,
        "col_nhead": 4,
        "col_num_inds": 8,
        "col_affine": False,
        "col_feature_group": "same",
        "col_feature_group_size": 3,
        "col_target_aware": True,
        "col_ssmax": "qassmax-mlp-elementwise",
        "row_num_blocks": 1,
        "row_nhead": 4,
        "row_num_cls": 2,
        "row_rope_base": 100000,
        "row_rope_interleaved": False,
        "icl_num_blocks": 2,
        "icl_nhead": 4,
        "icl_ssmax": "qassmax-mlp-elementwise",
        "ff_factor": 2,
        "dropout": 0.0,
        "norm_first": True,
        "bias_free_ln": True,
        "zero_init": False,
        "recompute": False,
    }
    seed = 43
    models = {}
    rng_states = {}
    for passes in (1, 3, 4):
        torch.manual_seed(seed)
        model = TabICL(**config, shared_depth_icl_num_passes=passes)
        rng_states[passes] = torch.random.get_rng_state().clone()
        models[passes] = model.to(device)
    require(
        torch.equal(rng_states[1], rng_states[3])
        and torch.equal(rng_states[1], rng_states[4]),
        "INITIALIZATION_RNG_DRIFT",
    )

    base_state = models[1].state_dict()
    expected_extra = {
        "icl_predictor.tf_icl.shared_depth_gate",
        "icl_predictor.tf_icl.shared_depth_condition_weight",
    }
    base_parameters = sum(p.numel() for p in models[1].parameters())
    checks = {}
    for passes in (3, 4):
        model = models[passes]
        state = model.state_dict()
        require(set(state) - set(base_state) == expected_extra, "STATE_KEY_DELTA", str(passes))
        require(set(base_state) <= set(state), "BASE_STATE_KEYS_MISSING", str(passes))
        require(
            all(torch.equal(value, state[name]) for name, value in base_state.items()),
            "BASELINE_TENSOR_INITIALIZATION_DRIFT",
            str(passes),
        )
        require(
            sum(p.numel() for p in model.parameters()) - base_parameters == 52,
            "GATE_PARAMETER_DELTA_NOT_52",
            str(passes),
        )
        encoder = model.icl_predictor.tf_icl
        require(isinstance(encoder, GatedLoopEncoder), "WRONG_LOOP_ENCODER", str(passes))
        require(encoder.shared_depth_num_passes == passes, "PASS_COUNT_NOT_PLUMBED", str(passes))
        require(encoder.shared_depth_gate.shape == torch.Size([]), "A_NOT_SCALAR")
        require(encoder.shared_depth_condition_weight.shape == (51,), "W_NOT_51_VECTOR")
        require(torch.count_nonzero(encoder.shared_depth_gate).item() == 0, "A_NOT_ZERO")
        require(torch.count_nonzero(encoder.shared_depth_condition_weight).item() == 0, "W_NOT_ZERO")
        require(isinstance(model.shared_depth_condition_stats, SupportSchemaStatistics), "WRONG_CONTEXT_STATS")
        require(
            sum(p.numel() for p in model.shared_depth_condition_stats.parameters()) == 0,
            "CONTEXT_STATS_NOT_PARAMETER_FREE",
        )
    require(set(models[3].state_dict()) == set(models[4].state_dict()), "LOOP_PARAMETER_SET_DRIFT")
    require(
        all(torch.equal(value, models[4].state_dict()[name]) for name, value in models[3].state_dict().items()),
        "LOOP_INITIALIZATION_DRIFT",
    )
    checks["initialization_rng_and_52_parameter_delta"] = "PASS"

    generator = torch.Generator(device="cpu").manual_seed(2026091243)
    X = torch.randn(2, 12, 5, generator=generator).to(device)
    y = torch.randn(2, 8, generator=generator).to(device)
    d = torch.tensor([5, 4], device=device)
    train_size = y.shape[1]
    outputs = {}
    call_counts = {}
    for passes, model in models.items():
        model.train()
        counts = [0] * config["icl_num_blocks"]
        handles = []
        for block_index, block in enumerate(model.icl_predictor.tf_icl.blocks):
            def count_forward(_module, _inputs, _output, index=block_index, target=counts):
                target[index] += 1
            handles.append(block.register_forward_hook(count_forward))
        try:
            # Grouped column embedding accepts no per-table d.  Keep d only
            # for the independent support-statistics masking tests below.
            outputs[passes] = model(X.clone(), y.clone())
        finally:
            for handle in handles:
                handle.remove()
        # Hooks are removed before backward, so activation recomputation cannot
        # accidentally inflate the measured number of forward stack passes.
        require(counts == [passes] * config["icl_num_blocks"], "FORWARD_BLOCK_CALL_COUNT", repr(counts))
        require(outputs[passes].shape == (2, 4, 17), "REGRESSION_OUTPUT_SHAPE", str(outputs[passes].shape))
        require(torch.isfinite(outputs[passes]).all().item(), "NONFINITE_ZERO_GATE_OUTPUT")
        call_counts[str(passes)] = counts
    require(torch.equal(outputs[1], outputs[3]), "ZERO_GATE_LOOP3_BASELINE_DRIFT")
    require(torch.equal(outputs[1], outputs[4]), "ZERO_GATE_LOOP4_BASELINE_DRIFT")
    checks["zero_gate_bitwise_outputs_and_forward_pass_counts"] = "PASS"

    gradient_norms = {}
    for passes in (3, 4):
        model = models[passes]
        model.zero_grad(set_to_none=True)
        loss = outputs[passes].square().mean() + 0.31 * outputs[passes].mean()
        loss.backward()
        encoder = model.icl_predictor.tf_icl
        gate_grad = encoder.shared_depth_gate.grad
        weight_grad = encoder.shared_depth_condition_weight.grad
        require(gate_grad is not None, "MISSING_A_GRAD", str(passes))
        require(weight_grad is not None, "MISSING_W_GRAD", str(passes))
        require(torch.isfinite(gate_grad).all().item(), "NONFINITE_A_GRAD", str(passes))
        require(torch.isfinite(weight_grad).all().item(), "NONFINITE_W_GRAD", str(passes))
        require(gate_grad.abs().item() > 0, "ZERO_A_GRAD", str(passes))
        require(weight_grad.norm().item() > 0, "ZERO_W_GRAD", str(passes))
        gradient_norms[str(passes)] = {
            "a_abs": gate_grad.abs().item(),
            "w_norm": weight_grad.norm().item(),
        }
    checks["actual_regression_loss_gate_gradients"] = "PASS"

    stats = models[3].shared_depth_condition_stats
    context = stats(X, y, d=d, total_seq_len=X.shape[1])
    require(context.shape == (2, 51), "CONTEXT_SHAPE", str(context.shape))
    require(torch.isfinite(context).all().item(), "NONFINITE_CONTEXT")
    require(torch.count_nonzero(context[:, 7:11]).item() == 0, "REGRESSION_LABEL_STATS_NOT_ZERO")
    query_changed = X.clone()
    query_changed[:, train_size:] = query_changed[:, train_size:] * 1000.0 + 7000.0
    require(
        torch.equal(context, stats(query_changed, y, d=d, total_seq_len=X.shape[1])),
        "QUERY_FEATURE_LEAKAGE_IN_CONTEXT",
    )
    labels_changed = y * 100.0 - 23.0
    require(
        torch.equal(context, stats(X, labels_changed, d=d, total_seq_len=X.shape[1])),
        "REGRESSION_LABEL_VALUE_LEAKAGE_IN_CONTEXT",
    )
    support_changed = X.clone()
    support_changed[:, 0, 0] += 13.0
    require(
        not torch.equal(context, stats(support_changed, y, d=d, total_seq_len=X.shape[1])),
        "CONTEXT_IGNORES_SUPPORT",
    )
    padded_changed = X.clone()
    padded_changed[1, :, 4] += 10000.0
    require(
        torch.equal(context, stats(padded_changed, y, d=d, total_seq_len=X.shape[1])),
        "PADDED_FEATURE_LEAKAGE_IN_CONTEXT",
    )
    checks["support_only_context_and_zero_regression_label_slots"] = "PASS"

    # The top-level cache must retain the context through the public cache
    # operations used for inference device placement and ensemble batching.
    context_cache = TabICLCache(
        train_shape=(2, train_size, X.shape[-1]),
        num_classes=0,
        g5sc_dataset_context=context.detach().clone(),
    )
    moved_cache = context_cache.to("cpu", dtype=torch.float64)
    require(moved_cache.g5sc_dataset_context is not None, "CACHE_TO_DROPS_CONTEXT")
    require(moved_cache.g5sc_dataset_context.device.type == "cpu", "CACHE_TO_CONTEXT_DEVICE")
    require(moved_cache.g5sc_dataset_context.dtype == torch.float64, "CACHE_TO_CONTEXT_DTYPE")
    require(
        torch.equal(moved_cache.g5sc_dataset_context, context.to(device="cpu", dtype=torch.float64)),
        "CACHE_TO_CONTEXT_VALUE_DRIFT",
    )
    first_cache = context_cache.slice_batch(0, 1)
    second_cache = context_cache.slice_batch(1, 2)
    require(first_cache.g5sc_dataset_context is not None, "CACHE_SLICE_DROPS_CONTEXT")
    require(second_cache.g5sc_dataset_context is not None, "CACHE_SLICE_DROPS_CONTEXT")
    require(torch.equal(first_cache.g5sc_dataset_context, context[:1]), "CACHE_FIRST_SLICE_CONTEXT_DRIFT")
    require(torch.equal(second_cache.g5sc_dataset_context, context[1:]), "CACHE_SECOND_SLICE_CONTEXT_DRIFT")
    merged_cache = TabICLCache.concat([first_cache, second_cache], dim=0)
    require(merged_cache.g5sc_dataset_context is not None, "CACHE_CONCAT_DROPS_CONTEXT")
    require(torch.equal(merged_cache.g5sc_dataset_context, context), "CACHE_CONCAT_CONTEXT_DRIFT")
    require(merged_cache.train_shape == context_cache.train_shape, "CACHE_CONCAT_TRAIN_SHAPE_DRIFT")
    require(torch.equal(context_cache.g5sc_dataset_context, context), "CACHE_OPERATIONS_MUTATE_CONTEXT")
    plain_cache = TabICLCache(train_shape=(2, train_size, X.shape[-1]), num_classes=0)
    require(plain_cache.to("cpu").g5sc_dataset_context is None, "PLAIN_CACHE_TO_ADDS_CONTEXT")
    plain_slices = [plain_cache.slice_batch(0, 1), plain_cache.slice_batch(1, 2)]
    require(all(item.g5sc_dataset_context is None for item in plain_slices), "PLAIN_CACHE_SLICE_ADDS_CONTEXT")
    require(TabICLCache.concat(plain_slices).g5sc_dataset_context is None, "PLAIN_CACHE_CONCAT_ADDS_CONTEXT")
    checks["top_level_cache_context_to_slice_concat"] = "PASS"

    # All remaining checks use deliberately nonzero gates.  Zero gates alone
    # could conceal broken recurrent execution or incorrectly indexed caches.
    with torch.no_grad():
        for passes in (3, 4):
            encoder = models[passes].icl_predictor.tf_icl
            encoder.shared_depth_gate.fill_(0.35)
            encoder.shared_depth_condition_weight.copy_(
                torch.linspace(-0.02, 0.02, 51, device=device)
            )
    baseline_eval = models[1].eval()(X.clone(), y.clone())
    cache_sizes = {}
    for passes in (3, 4):
        model = models[passes].eval()
        encoder = model.icl_predictor.tf_icl
        with torch.no_grad():
            nonzero_output = model(X.clone(), y.clone())
            require(torch.isfinite(nonzero_output).all().item(), "NONFINITE_NONZERO_GATE_OUTPUT")
            require(not torch.equal(nonzero_output, baseline_eval), "NONZERO_GATE_HAS_NO_EFFECT", str(passes))
            ctx = context.to(dtype=encoder.shared_depth_condition_weight.dtype)
            logits = (ctx * encoder.shared_depth_condition_weight).sum(dim=-1)
            bounded = 0.1 * (2.0 * torch.sigmoid(logits) - 1.0)
            expected_alpha = torch.tanh(encoder.shared_depth_gate + bounded).view(-1, 1, 1)
            actual_alpha = encoder._shared_depth_alpha(context, nonzero_output)
            require(torch.equal(actual_alpha, expected_alpha.to(dtype=nonzero_output.dtype)), "ALPHA_FORMULA_DRIFT")
            require(bounded.abs().max().item() <= 0.100001, "INNER_GATE_BOUND_DRIFT")

            checkpoint_config = dict(config, shared_depth_icl_num_passes=passes)
            buffer = io.BytesIO()
            torch.save({"config": checkpoint_config, "state_dict": model.state_dict()}, buffer)
            buffer.seek(0)
            saved = torch.load(buffer, map_location=device, weights_only=True)
            reloaded = TabICL(**saved["config"]).to(device).eval()
            reloaded.load_state_dict(saved["state_dict"], strict=True)
            restored_output = reloaded(X.clone(), y.clone())
            require(torch.equal(nonzero_output, restored_output), "CONFIG_STATE_ROUNDTRIP_DRIFT", str(passes))
            require(reloaded.icl_predictor.tf_icl.shared_depth_num_passes == passes, "RELOADED_PASS_COUNT_DRIFT")

            src = torch.randn(
                2, 12, config["embed_dim"] * config["row_num_cls"], generator=generator
            ).to(device)

            def run_stack(value):
                for block in encoder.blocks:
                    value = block(q=value, train_size=train_size, rope=encoder.rope)
                return value

            expected = run_stack(src.clone())
            for _pass_index in range(1, passes):
                previous = expected
                candidate = run_stack(previous)
                alpha = encoder._shared_depth_alpha(context, candidate)
                expected = previous + alpha * (candidate - previous)
            actual = encoder(src.clone(), train_size=train_size, dataset_context=context)
            require(torch.equal(expected, actual), "MANUAL_RECURRENCE_DRIFT", str(passes))

            cache = KVCache()
            cached_full = encoder.forward_with_cache(
                src.clone(), icl_cache=cache, train_size=train_size,
                use_cache=False, store_cache=True, dataset_context=context,
            )
            expected_keys = list(range(passes * config["icl_num_blocks"]))
            require(sorted(cache.kv) == expected_keys, "PER_PASS_KV_INDEX_DRIFT", repr(sorted(cache.kv)))
            require(torch.allclose(cached_full, actual, atol=2e-5, rtol=2e-5), "CACHE_PREFILL_RECURRENCE_DRIFT", str(passes))
            cached_query = encoder.forward_with_cache(
                src[:, train_size:].clone(), icl_cache=cache,
                use_cache=True, store_cache=False, dataset_context=context,
            )
            require(
                torch.allclose(cached_query, actual[:, train_size:], atol=2e-5, rtol=2e-5),
                "CACHE_QUERY_RECURRENCE_DRIFT", str(passes),
            )
            cache_sizes[str(passes)] = len(cache.kv)
    checks["nonzero_gate_formula_effect_and_checkpoint_roundtrip"] = "PASS"
    checks["manual_recurrence_and_per_pass_kv_cache"] = "PASS"

    # Exercise the actual Muon branch with a rank-zero parameter.  The baseline
    # implementation's len(g) fails here, even though all old tensors work.
    scalar = nn.Parameter(torch.zeros((), device=device))
    vector = nn.Parameter(torch.zeros(51, device=device))
    optimizer = Muon(
        param_groups=[dict(params=[scalar, vector], use_muon=True)],
        lr=8e-4, weight_decay=0.01, matched_adamw_rms=0.2,
        momentum=0.9, nesterov=True, ns_steps=5,
        adamw_betas=(0.9, 0.999), adamw_eps=1e-8, use_cautious_wd=False,
    )
    scalar.grad = torch.ones_like(scalar)
    vector.grad = torch.linspace(0.1, 0.2, 51, device=device)
    optimizer.step()
    require(torch.isfinite(scalar).item(), "MUON_SCALAR_NONFINITE")
    require(scalar.abs().item() > 0.0, "MUON_SCALAR_NOT_UPDATED")
    require(torch.isfinite(vector).all().item(), "MUON_VECTOR_NONFINITE")
    require(vector.norm().item() > 0.0, "MUON_VECTOR_NOT_UPDATED")
    checks["muon_scalar_and_vector_step"] = "PASS"

    receipt = {
        "status": "PASS_T25_G5SC_LOOP34_SMOKE",
        "source": str(source),
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": seed,
        "config": config,
        "tested_passes": [1, 3, 4],
        "new_parameters_over_t25": 52,
        "baseline_parameters": base_parameters,
        "block_forward_counts": call_counts,
        "gate_gradient_norms": gradient_norms,
        "per_pass_kv_entries": cache_sizes,
        "checks": checks,
        "cache_scope": "encoder parity with an identical fixed support context; total_seq_len held fixed",
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
