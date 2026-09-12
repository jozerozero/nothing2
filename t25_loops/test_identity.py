#!/usr/bin/env python3
"""Native G5SC identity, regression, cache, prior, and actual Trainer smoke.

Small feature/embedding dimensions keep CPU work bounded; all twelve ICL
layers and the real 999-quantile head/loss remain enabled. No formal checkpoint
or distributed job is read, modified, or submitted by this test.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from unittest import mock

from runtime_probe import require, tensor_state_hash, verify_model_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--regression-target-profile", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--identity-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    identity = verify_model_identity(source, args.reference_source)
    if args.identity_only:
        # Deliberately NOT the full PASS status accepted by submission guards.
        print(json.dumps(dict(identity, status="PASS_SOURCE_IDENTITY_ONLY"), sort_keys=True))
        return
    require(args.regression_target_profile is not None and args.regression_target_profile.is_file(),
            "REAL_T25_TARGET_PROFILE_REQUIRED")
    require(identity["assembled_python_identity_verified"], "FULL_ASSEMBLED_TRAINER_IDENTITY_REQUIRED")
    require("RANK" not in os.environ, "SMOKE_MUST_NOT_RUN_INSIDE_FORMAL_TORCHRUN")
    sys.path.insert(0, str(source / "src"))

    import numpy as np
    import torch
    import torch.nn.functional as F
    import tabicl._model.tabicl as model_module
    from tabicl._model.tabicl import TabICL
    from tabicl._model.encoders import Encoder
    from tabicl._model.kv_cache import KVCache, TabICLCache
    from tabicl.train._t25_regression_adapter import build_t25_prior, pinball_loss
    from tabicl.train._train_config import build_parser
    from tabicl.train import _g5sc_runtime_probe as live_probe

    require(Path(model_module.__file__).resolve() == source / "src/tabicl/_model/tabicl.py",
            "WRONG_IMPORTED_MODEL_SOURCE", str(model_module.__file__))
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if args.device == "cuda":
        require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
        torch.cuda.set_device(0)
    device = torch.device(args.device)
    seed = 2026080101
    common = dict(max_classes=0, num_quantiles=999, embed_dim=16,
                  col_num_blocks=3, col_nhead=8, col_num_inds=8,
                  col_feature_group="same", col_feature_group_size=3,
                  col_target_aware=True, col_ssmax="qassmax-mlp-elementwise",
                  row_num_blocks=3, row_nhead=8, row_num_cls=4,
                  icl_num_blocks=12, icl_nhead=8, icl_ssmax="qassmax-mlp-elementwise",
                  ff_factor=2, activation="gelu", dropout=0.0,
                  norm_first=True, bias_free_ln=False, recompute=False,
                  shared_depth_icl_rho=1.0)
    models, configs, rng = {}, {}, {}
    for passes in (1, 3, 4):
        torch.manual_seed(seed)
        configs[passes] = dict(common, shared_depth_icl_enabled=passes > 1,
                              shared_depth_icl_dataset_conditioned=passes > 1,
                              shared_depth_icl_num_passes=passes)
        models[passes] = TabICL(**configs[passes]).to(device)
        rng[passes] = torch.get_rng_state().clone()
    base = models[1].state_dict()
    gate_keys = {"icl_predictor.tf_icl.shared_depth_gate",
                 "icl_predictor.tf_icl.shared_depth_condition_weight"}
    base_count = sum(p.numel() for p in models[1].parameters())
    initial_hashes = {}
    checks = {"every_model_source_file_byte_identity": "PASS"}
    for passes in (3, 4):
        model, state = models[passes], models[passes].state_dict()
        enc = model.icl_predictor.tf_icl
        require(type(enc) is Encoder, "NON_NATIVE_ENCODER")
        require(torch.equal(rng[1], rng[passes]), "INIT_RNG_DRIFT")
        require(set(state) - set(base) == gate_keys and set(base) <= set(state), "STATE_KEY_DELTA")
        require(all(torch.equal(value, state[key]) for key, value in base.items()), "BASE_TENSOR_INIT_DRIFT")
        require(sum(p.numel() for p in model.parameters()) - base_count == 52, "GATE_PARAMETER_DELTA")
        require(enc.shared_depth_gate.shape == torch.Size([]) and enc.shared_depth_condition_weight.shape == (51,),
                "GATE_SHAPES")
        require(enc.shared_depth_rho == 1.0 and not enc.recompute, "RHO_RECOMPUTE_DRIFT")
        require(all(torch.count_nonzero(block.attn.out_proj.weight).item() == 0
                    and torch.count_nonzero(block.linear2.weight).item() == 0 for block in enc.blocks),
                "ORIGINAL_ZERO_RESIDUAL_INIT_LOST")
        initial_hashes[str(passes)] = tensor_state_hash(model)
    require(initial_hashes["3"] == initial_hashes["4"], "LOOP34_INITIAL_TENSORS_DIFFER")
    checks["same_initial_tensors_rng_and_52_gate_parameters"] = "PASS"

    generator = torch.Generator(device="cpu").manual_seed(2026091243)
    X = torch.randn(2, 12, 6, generator=generator).to(device)
    all_y = torch.randn(2, 12, generator=generator).to(device)
    support_size, y = 8, all_y[:, :8]
    outputs, counts_by_pass, gate_gradients = {}, {}, {}
    for passes, model in models.items():
        model.train()
        counts, handles = [0] * 12, []
        for index, block in enumerate(model.icl_predictor.tf_icl.blocks):
            def hook(_module, _args, _output, index=index, counts=counts):
                counts[index] += 1
            handles.append(block.register_forward_hook(hook))
        try:
            outputs[passes] = model(X, y)
        finally:
            for handle in handles:
                handle.remove()
        require(counts == [passes] * 12, "FORWARD_PASS_COUNT", repr(counts))
        require(outputs[passes].shape == (2, 4, 999), "REAL_QUANTILE_HEAD_SHAPE")
        require(torch.isfinite(outputs[passes]).all().item(), "NONFINITE_PREDICTIONS")
        counts_by_pass[str(passes)] = counts
    require(torch.equal(outputs[1], outputs[3]) and torch.equal(outputs[1], outputs[4]), "ZERO_GATE_FIRST_PASS_DRIFT")
    for passes in (3, 4):
        model, prediction = models[passes], outputs[passes]
        loss = pinball_loss(prediction, all_y[:, support_size:])
        alpha = torch.linspace(0, 1, 1001, device=device, dtype=prediction.dtype)[1:-1].view(1, 1, -1)
        error = all_y[:, support_size:].unsqueeze(-1) - prediction
        reference_loss = torch.maximum(alpha * error, (alpha - 1) * error).mean()
        require(torch.equal(loss, reference_loss), "T25_PINBALL_FORMULA_DRIFT")
        require(torch.isfinite(loss).item(), "NONFINITE_PINBALL")
        loss.backward()
        require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), "NONFINITE_GRADIENT")
        enc = model.icl_predictor.tf_icl
        require(enc.shared_depth_gate.grad is not None and enc.shared_depth_condition_weight.grad is not None,
                "DISCONNECTED_GATE")
        # Native zero-residual initialization makes F(h)-h exactly zero.
        require(torch.count_nonzero(enc.shared_depth_gate.grad).item() == 0
                and torch.count_nonzero(enc.shared_depth_condition_weight.grad).item() == 0,
                "ZERO_RESIDUAL_GATE_GRAD_NOT_ZERO")
        gate_gradients[str(passes)] = {"initial_a_abs": enc.shared_depth_gate.grad.abs().item(),
                                    "initial_w_norm": enc.shared_depth_condition_weight.grad.norm().item()}
    checks["zero_gate_first_pass_and_12_layer_call_counts"] = "PASS"
    checks["real_999_pinball_forward_backward_native_zero_gate_gradient"] = "PASS"

    stats = models[3].shared_depth_condition_stats
    d = torch.tensor([6, 5], device=device)
    context = stats(X, y, d=d, total_seq_len=12)
    require(context.shape == (2, 51) and torch.isfinite(context).all().item(), "SUPPORT_CONTEXT_SHAPE")
    require(torch.count_nonzero(context[:, 7:11]).item() == 0, "REGRESSION_LABEL_SLOTS")
    changed = X.clone()
    changed[:, support_size:] = changed[:, support_size:] * 1000 + 2000
    require(torch.equal(context, stats(changed, y, d=d, total_seq_len=12)), "QUERY_FEATURE_CONTEXT_LEAK")
    require(torch.equal(context, stats(X, y * 100 - 20, d=d, total_seq_len=12)), "REGRESSION_LABEL_CONTEXT_LEAK")
    changed = X.clone()
    changed[1, :, 5] += 1000
    require(torch.equal(context, stats(changed, y, d=d, total_seq_len=12)), "PADDING_CONTEXT_LEAK")
    changed = X.clone()
    changed[:, 0, 0] += 10
    require(not torch.equal(context, stats(changed, y, d=d, total_seq_len=12)), "CONTEXT_IGNORES_SUPPORT")
    top_cache = TabICLCache(train_shape=(2, support_size, 6), num_classes=0,
                           attention_gate_context=context.detach().clone())
    require(torch.equal(top_cache.to("cpu", dtype=torch.float64).attention_gate_context,
                        context.to("cpu", dtype=torch.float64)), "CACHE_TO_CONTEXT")
    slices = [top_cache.slice_batch(0, 1), top_cache.slice_batch(1, 2)]
    require(torch.equal(TabICLCache.concat(slices).attention_gate_context, context), "CACHE_SLICE_CONCAT_CONTEXT")
    checks["native_support_statistics_and_cache_context_operations"] = "PASS"

    cache_sizes = {}
    for passes in (3, 4):
        model = models[passes]
        enc = model.icl_predictor.tf_icl
        # Deliberately perturb only this disposable test model. Nonzero residuals
        # are necessary to expose gate/recurrence bugs hidden by native zero init.
        with torch.no_grad():
            residual = enc.blocks[0].linear2.weight
            residual.copy_(torch.linspace(-0.01, 0.01, residual.numel(), device=device).reshape_as(residual))
            enc.shared_depth_gate.fill_(0.35)
            enc.shared_depth_condition_weight.copy_(torch.linspace(-0.02, 0.02, 51, device=device))
        model.zero_grad(set_to_none=True)
        prediction = model.train()(X, y)
        loss = pinball_loss(prediction, all_y[:, support_size:])
        loss.backward()
        require(torch.isfinite(enc.shared_depth_gate.grad).item() and enc.shared_depth_gate.grad.abs().item() > 0,
                "NONZERO_RESIDUAL_GATE_HAS_NO_GRADIENT")
        require(torch.isfinite(enc.shared_depth_condition_weight.grad).all().item()
                and enc.shared_depth_condition_weight.grad.norm().item() > 0, "CONDITION_WEIGHT_HAS_NO_GRADIENT")
        gate_gradients[str(passes)].update(nonzero_residual_a_abs=enc.shared_depth_gate.grad.abs().item(),
                                          nonzero_residual_w_norm=enc.shared_depth_condition_weight.grad.norm().item())
        model.eval()
        with torch.no_grad():
            src = torch.randn(2, 12, 64, generator=generator).to(device)
            def stack(value):
                for block in enc.blocks:
                    value = block(q=value, train_size=support_size, rope=enc.rope, ffn_context=context)
                return value
            first = stack(src)
            expected = first
            formula = torch.tanh(enc.shared_depth_gate + 0.1 * (
                2 * torch.sigmoid((context * enc.shared_depth_condition_weight).sum(-1)) - 1)).view(-1, 1, 1)
            require(torch.equal(enc._shared_depth_alpha(context, src), formula), "GATE_FORMULA_DRIFT")
            for _ in range(1, passes):
                candidate = stack(expected)
                expected = expected + formula * (candidate - expected)
            actual = enc(src, train_size=support_size, dataset_context=context)
            require(torch.equal(actual, expected), "MANUAL_RECURRENCE_DRIFT")
            require(not torch.allclose(actual, first, atol=1e-7, rtol=1e-7), "NONZERO_GATE_HAS_NO_EFFECT")
            cache = KVCache()
            prefill = enc.forward_with_cache(src, icl_cache=cache, train_size=support_size, dataset_context=context)
            require(sorted(cache.kv) == list(range(12 * passes)), "PER_PASS_KV_INDEX")
            query = enc.forward_with_cache(src[:, support_size:], icl_cache=cache, use_cache=True,
                                           store_cache=False, dataset_context=context)
            require(torch.allclose(prefill, actual, atol=3e-5, rtol=3e-5), "ENCODER_CACHE_PREFILL_DRIFT")
            require(torch.allclose(query, actual[:, support_size:], atol=3e-5, rtol=3e-5), "ENCODER_QUERY_CACHE_DRIFT")
            cache_sizes[str(passes)] = len(cache.kv)
            full_output = model(X, y)
            cached_output = model.forward_with_cache(X_train=X[:, :support_size], y_train=y,
                                                      X_test=X[:, support_size:], cache_mode="kv")
            require(torch.allclose(full_output, cached_output, atol=3e-5, rtol=3e-5), "MODEL_CACHE_PREFILL_DRIFT")
            fixed_context = model._cache.attention_gate_context.clone()
            cached_query = model.forward_with_cache(X_test=X[:, support_size:], use_cache=True, store_cache=False)
            require(torch.allclose(cached_query, full_output, atol=3e-5, rtol=3e-5), "MODEL_CACHE_QUERY_DRIFT")
            require(torch.equal(model._cache.attention_gate_context, fixed_context), "QUERY_MUTATED_CACHED_CONTEXT")
            invalid_cache = dataclasses.replace(model._cache, attention_gate_context=None)
            try:
                model.forward_with_cache(X_test=X[:, support_size:], cache=invalid_cache)
            except ValueError as error:
                require("context" in str(error).lower(), "UNEXPECTED_MISSING_CONTEXT_ERROR", str(error))
            else:
                raise AssertionError("MISSING_GATED_CACHE_CONTEXT_ACCEPTED")
            buffer = io.BytesIO()
            torch.save({"config": configs[passes], "state_dict": model.state_dict()}, buffer)
            buffer.seek(0)
            saved = torch.load(buffer, map_location=device, weights_only=True)
            restored = TabICL(**saved["config"]).to(device).eval()
            restored.load_state_dict(saved["state_dict"], strict=True)
            require(torch.equal(restored(X, y), full_output), "CHECKPOINT_ROUNDTRIP")
    checks["nonzero_residual_gate_grad_effect_manual_recurrence"] = "PASS"
    checks["native_encoder_and_model_cache_parity_fixed_total_length"] = "PASS"
    checks["native_model_config_state_roundtrip"] = "PASS"

    # Verify the deployed probe itself, including preservation of an existing
    # gradient object/value and mixed train/eval flags. Never bypass identity.
    fresh = TabICL(**configs[3]).to(device).eval()
    first_parameter = next(fresh.parameters())
    first_parameter.grad = torch.full_like(first_parameter, 0.125)
    fresh.icl_predictor.train()
    old_flags = [m.training for m in fresh.modules()]
    old_python_rng, old_numpy_rng = random.getstate(), np.random.get_state()
    with tempfile.TemporaryDirectory(prefix="t25-native-probe-") as directory:
        probe_receipt = live_probe.probe(fresh, directory, strict_resources=False)
    require([m.training for m in fresh.modules()] == old_flags, "PROBE_TRAINING_FLAGS_CHANGED")
    require(random.getstate() == old_python_rng, "PROBE_PYTHON_RNG_CHANGED")
    current_numpy_rng = np.random.get_state()
    require(old_numpy_rng[0] == current_numpy_rng[0] and np.array_equal(old_numpy_rng[1], current_numpy_rng[1])
            and old_numpy_rng[2:] == current_numpy_rng[2:], "PROBE_NUMPY_RNG_CHANGED")
    require(probe_receipt["gradients_preserved"] and probe_receipt["rng_preserved"], "PROBE_STATE_CONTRACT")
    checks["runtime_probe_rng_parameter_gradient_and_mode_preservation"] = "PASS"

    def make_train_config(directory, passes=3):
        config = build_parser().parse_args([])
        # col_target_aware is the canonical model default, not a G5 CLI option.
        cli_common = {key: value for key, value in common.items() if key != "col_target_aware"}
        overrides = dict(cli_common, device=str(device), dtype="float32", np_seed=seed, torch_seed=seed,
                         prior_loader_seed=seed, regression_method="quantile", prior_type="graph_scm",
                         regression_target_prior="rw_sample50",
                         regression_target_profile=str(args.regression_target_profile.resolve()),
                         regression_target_mix_probability=0.25,
                         shared_depth_icl_enabled=True, shared_depth_icl_dataset_conditioned=True,
                         shared_depth_icl_num_passes=passes, optimizer="muon", lr=6e-4,
                         muon_momentum=0.95, cautious_weight_decay=True, weight_decay=0.01,
                         scheduler="cosine_warmup", warmup_proportion=0.02, max_steps=25,
                         batch_size=2, micro_batch_size=1, batch_size_per_gp=1,
                         min_seq_len=32, max_seq_len=32, min_features=4, max_features=6,
                         min_train_size=0.5, max_train_size=0.75, log_seq_len=False,
                         log_n_features=False, seq_len_per_gp=False, replay_small=False,
                         prior_device="cpu", prior_n_jobs=1, prior_num_workers=0,
                         prior_cache_enabled=False, prior_num_threads_per_generate=1,
                         amp=False, grad_scaler=False, model_compile=False,
                         wandb_log=False, wandb_mode="disabled", checkpoint_dir=str(directory),
                         checkpoint_path=None, save_temp_every=1, save_perm_every=1,
                         max_checkpoints=0, strict_training_stage_manifest=False)
        for key, value in overrides.items():
            require(hasattr(config, key), "PATCHED_TRAIN_PARSER_FIELD_MISSING", key)
            setattr(config, key, value)
        return config

    from tabicl.prior import _regression_target_prior as target_prior
    from tabicl.prior import _t25_safe_tail as safe_tail
    require(target_prior.map_to_template_from_support is safe_tail._safe_map_to_template_from_support,
            "ACTUAL_GENERATOR_SAFE_TAIL_PATCH_MISSING")
    require(target_prior.support_only_standardize is safe_tail._safe_support_only_standardize,
            "ACTUAL_GENERATOR_CLIP_PATCH_MISSING")
    generator_mixes = {}
    with tempfile.TemporaryDirectory(prefix="t25-native-data-") as directory:
        for mix in (0.0, 1.0):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            config = make_train_config(directory)
            config.regression_target_mix_probability = mix
            dataset = build_t25_prior(config)
            require(dataset.regression and dataset.prior_type == "graph_scm", "NOT_ACTUAL_REGRESSION_GRAPH_PRIOR")
            require(dataset.prior.n_jobs == 1, "NESTED_GENERATOR_MULTIPROCESSING")
            with mock.patch.object(target_prior, "map_to_template_from_support",
                                   wraps=target_prior.map_to_template_from_support) as mapped:
                batch = dataset.get_batch()
            require(len(batch) == 5, "PRIOR_BATCH_CONTRACT", str(len(batch)))
            bx, by, bd, lengths, supports = batch
            bx = bx.to_padded_tensor(0.0) if bx.is_nested else bx
            by = by.to_padded_tensor(0.0) if by.is_nested else by
            require(torch.isfinite(bx).all().item() and torch.isfinite(by).all().item(), "NONFINITE_REAL_PRIOR")
            require(by.dtype.is_floating_point and by.abs().max().item() <= 8.00001, "SAFE_TAIL_TARGET_CLIP")
            require(mapped.call_count == 0 if mix == 0 else mapped.call_count > 0, "RW_BRANCH_NOT_EXERCISED")
            t, n = int(lengths[0].item()), int(supports[0].item())
            prediction = fresh.train()(bx[:1, :t].to(device), by[:1, :n].to(device))
            actual_loss = pinball_loss(prediction, by[:1, n:t].to(device))
            fresh.zero_grad(set_to_none=True)
            actual_loss.backward()
            require(torch.isfinite(actual_loss).item(), "REAL_GENERATOR_PINBALL_NONFINITE")
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in fresh.parameters()),
                    "REAL_GENERATOR_GRADIENT_NONFINITE")
            generator_mixes[str(mix)] = {"status": "PASS", "mapped_calls": mapped.call_count,
                                         "shape": list(bx.shape), "support_size": n,
                                         "target_abs_max": by.abs().max().item(), "pinball_loss": actual_loss.item()}
    checks["actual_t25_graph_generator_mix0_mix1_and_999_pinball"] = "PASS"

    # Construct the real patched Trainer, obtain its real DataLoader batch, run
    # its own batch/optimizer path, and inspect the checkpoint it actually saves.
    from tabicl.train._run import Trainer
    trainer_steps = {}
    original_probe = live_probe.probe
    with tempfile.TemporaryDirectory(prefix="t25-native-trainer-") as directory:
        for passes in (3, 4):
            checkpoint_dir = Path(directory) / f"loop{passes}"
            config = make_train_config(checkpoint_dir, passes)
            with mock.patch.object(live_probe, "probe", side_effect=lambda m, p: original_probe(m, p, strict_resources=False)):
                trainer = Trainer(config)
            require(trainer.raw_model.max_classes == 0 and trainer.raw_model.num_quantiles == 999,
                    "TRAINER_CLASSIFICATION_MODEL")
            require(type(trainer.optimizer).__name__ == "Muon", "TRAINER_NOT_NATIVE_MUON")
            require("_muon" not in type(trainer.optimizer).__module__, "T25_OPTIMIZER_USED_INSTEAD_OF_G5SC")
            iterator = iter(trainer.dataloader)
            before = tensor_state_hash(trainer.raw_model)
            with mock.patch.object(F, "cross_entropy", side_effect=AssertionError("REGRESSION_CALLED_CROSS_ENTROPY")), \
                 mock.patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as optimizer_step:
                # Native cosine warmup starts at LR=0. Two actual updates prove
                # parameter motion without modifying the production schedule.
                for step in range(2):
                    metrics = trainer.run_batch(next(iterator))
                    require(not metrics.get("skipped_update", False), "TRAINER_SKIPPED_UPDATE", str(metrics))
                    trainer.curr_step = step + 1
            require(optimizer_step.call_count == 2, "TRAINER_DID_NOT_EXECUTE_MUON_STEPS")
            require(not metrics.get("skipped_update", False), "TRAINER_SKIPPED_UPDATE", str(metrics))
            require("pinball" in metrics and "ce" not in metrics and "accuracy" not in metrics,
                    "TRAINER_CLASSIFICATION_METRICS", str(metrics))
            require(math.isfinite(float(metrics["pinball"])), "TRAINER_NONFINITE_PINBALL")
            require(tensor_state_hash(trainer.raw_model) != before, "MUON_DID_NOT_UPDATE_PARAMETERS")
            trainer.save_checkpoint("smoke-step-2.ckpt")
            saved = torch.load(checkpoint_dir / "smoke-step-2.ckpt", map_location="cpu", weights_only=False)
            saved_config = saved["config"]
            require(saved_config["max_classes"] == 0 and saved_config["num_quantiles"] == 999
                    and saved_config["shared_depth_icl_num_passes"] == passes
                    and saved_config["shared_depth_icl_enabled"]
                    and saved_config["shared_depth_icl_dataset_conditioned"], "TRAINER_SAVED_WRONG_MODEL_CONFIG")
            require(saved["curr_step"] == 2 and saved["optimizer_state"]["state"], "TRAINER_CHECKPOINT_NO_OPTIMIZER_STATE")
            trainer_steps[str(passes)] = {"status": "PASS", "pinball": float(metrics["pinball"]),
                                         "optimizer": f"{type(trainer.optimizer).__module__}.Muon",
                                         "optimizer_step_calls": optimizer_step.call_count,
                                         "checkpoint_step": saved["curr_step"], "model_config": saved_config}
            if trainer.prior_cache is not None:
                trainer.prior_cache.stop()
            del trainer
    checks["actual_trainer_dataloader_muon_step_checkpoint_loop34_no_cross_entropy"] = "PASS"
    receipt = dict(identity, status="PASS_T25_G5SC_NATIVE_IDENTITY", device=str(device),
                   regression_target_profile_sha256=hashlib.sha256(args.regression_target_profile.read_bytes()).hexdigest(),
                   torch_version=torch.__version__, seed=seed, small_model_config=common,
                   tested_passes=[1, 3, 4], initial_state_hashes=initial_hashes,
                   base_parameter_count=base_count, added_gate_parameters=52,
                   block_forward_counts=counts_by_pass, gate_gradient_norms=gate_gradients,
                   encoder_cache_entries=cache_sizes, generator_mixes=generator_mixes,
                   trainer_steps=trainer_steps, checks=checks,
                   cache_scope="Original G5SC semantics, with identical support and total_seq_len at prefill/reuse")
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
