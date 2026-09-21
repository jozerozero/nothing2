#!/usr/bin/env python3
"""Separate, strict actual-eight-member original-Mitra all224 experiment.

Reuses the historical checkpoint, raw split audit, target transforms, BF16,
native 8192 support cap, 1024 query cap, and support-only initialization
validation. Eight native trainers share one advancing RNG; no reseeded copies.
Test-forward hooks verify exactly eight member contributions for every test row.
This is ensemble-count matching, not a claim of equal FLOPs or full context.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import time

import pfn_mitra_one as historical

common = historical.common
require = common.require
sha256_file = historical.sha256_file
HISTORICAL_WORKER_SHA256 = "070d4f6f24ac5183654191e5c870a1dfb2a2ec7de57e153f72a33c4dd3fe4500"
RUNTIME_PINS = {
    "autogluon.tabular.models.mitra.sklearn_interface": "bf283c9249982b310945689e844a84b2065e382576d2d9d901969a6899300ee0",
    "autogluon.tabular.models.mitra._internal.core.trainer_finetune": "3dc3f65b7dcb6a4525a35c7557e686c05b752984e424dda7febb1d8421587e1b",
    "autogluon.tabular.models.mitra._internal.data.dataset_finetune": "018c72435349677a40a7a9f2f98fe1ab0ff856dc96e57d400017217fba55b123",
    "autogluon.tabular.models.mitra._internal.data.preprocessor": "cf1da95aba436864926b03da8fb89e46314eb5cb7f17287c2d88e8fd8b65346d",
}
RECIPE = {**historical.RECIPE, "experiment": "strict_actual8_original_mitra_all224_v1",
          "n_estimators": 8, "actual_members_per_test_row": 8, "seed": 42,
          "historical_seed_change": "historical one-member seed0 -> strict-eight common seed42",
          "native_cfg_n_ensembles": 8,
          "scope": "all224 original regression memberships",
          "ensemble_realization": "eight native trainers, each covers every test row once in native contiguous query chunks",
          "ensemble_rng": "one native RandomState(seed=42), advancing across all trainers; global NumPy seed42 for native mirrors",
          "ensemble_average": "native arithmetic mean in outer-transformed target units, then outer inverse",
          "equal_compute_claim": False, "equal_full_context_claim": False,
          "support_context_exception": "all official support retained by loader; native context caps8192 per member/chunk when support exceeds8192"}


def load_source_result224(path, manifest, row, checkpoint):
    source = json.loads(Path(path).read_text())
    counts = {suite: sum(r["suite"] == suite for r in manifest["rows"])
              for suite in ("talent", "BCCO", "CTR23", "TabArena", "PFN")}
    require(counts == {"talent": 100, "BCCO": 50, "CTR23": 33, "TabArena": 13, "PFN": 28},
            "Expected exact all224 regression suite membership")
    expected = {"complete": True, "checkpoint_step": 22175,
        "dataset_index": row["dataset_index"], "row_id": row["row_id"], "dataset": row["dataset"],
        "suite": row["suite"], "task_kind": "regression", "manifest_id": manifest["manifest_id"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "dataset_protocol_fingerprint": row["protocol_fingerprint"], "input_fingerprint": row["input_fingerprint"]}
    for key, value in expected.items():
        require(source.get(key) == value, f"Original source result mismatch: {key}")
    require(source.get("checkpoint") == checkpoint and checkpoint.get("finetune_step") == 0
            and checkpoint.get("kind") == "source_baseline", "Reference must be original unfinetuned step22175")
    require(source.get("strict_checkpoint_load") is True and source.get("checkpoint_load_weights_only") is True,
            "Reference checkpoint strict load was not verified")
    calls = source.get("actual_forward_block_calls", [])
    require(calls and all(call == [3] * 12 for call in calls), "Reference Loop3 execution not verified")
    require(all(math.isfinite(float(source["metrics"][key])) for key in ("rmse", "mae", "r2")),
            "Reference metrics are nonfinite")
    return source


def verify_data_parity224(audit, source, transform_source, transform, suite):
    require(audit == source.get("data_audit"), "Raw support/test data audit differs from original step22175")
    if suite == "PFN":
        require(audit["split"]["split"] == "OpenML official repeat=0/fold=0", "Incorrect official PFN split")
    for key in ("canonical_support_frame_sha256", "canonical_test_frame_sha256",
                "support_targets_sha256", "test_targets_sha256"):
        require(isinstance(audit.get(key), str) and len(audit[key]) == 64, f"Missing exact-data hash: {key}")
    require(transform_source == source.get("target_transform_source"), "Outer target transform source differs")
    require(transform == source.get("target_transform"), "Support-fitted outer target transform differs")


def array_digest(value):
    import hashlib
    import numpy as np
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def rng_digest(rng):
    algorithm, keys, position, has_gauss, cached_gauss = rng.get_state()
    return common.object_digest({"algorithm": algorithm, "keys": keys.tolist(),
        "position": int(position), "has_gauss": int(has_gauss), "cached_gauss": float(cached_gauss)})


def check_cfg8(cfg):
    expected = {"max_epochs": 0, "max_samples_support": 8192, "max_samples_query": 1024,
                "precision": "bfloat16", "dim_output": 1, "n_ensembles": 8,
                "grad_scaler_enabled": False, "random_mirror_regression": True,
                "random_mirror_x": True, "shuffle_classes": False,
                "shuffle_features": False, "use_random_transforms": False}
    for key, value in expected.items():
        require(cfg.hyperparams.get(key) == value, f"Strict-eight Mitra configuration changed: {key}")
    require(cfg.seed == 42, "Strict-eight common base seed changed")
    return expected


def guarded_trainer8(original):
    class GuardedTrainer8(original):
        def __init__(self, cfg, model, *args, **kwargs):
            check_cfg8(cfg)
            require(model.dim_output == 1 and not model.use_flash_attn,
                    "Expected original scalar regression head and stock ROCm attention")
            rng = kwargs.get("rng")
            require(rng is not None, "Native shared ensemble RNG must be explicit")
            self.pfn_rng_before_init = rng_digest(rng)
            super().__init__(cfg, model, *args, **kwargs)
            self.pfn_optimizer_step_attempts = 0
            self.pfn_fit_calls = 0

            def forbidden_step(*_args, **_kwargs):
                self.pfn_optimizer_step_attempts += 1
                raise RuntimeError("optimizer.step is prohibited in strict-eight original-Mitra inference")
            self.optimizer.step = forbidden_step

        def train(self, x_train, y_train, x_val, y_val):
            import numpy as np
            check_cfg8(self.cfg)
            require(x_train is x_val and y_train is y_val,
                    "Initialization validation must be identical support, never test or holdout")
            self.pfn_fit_calls += 1
            require(self.pfn_fit_calls == 1, "Unexpected native member refit")
            self.pfn_support_rows = len(y_train)
            self.pfn_preprocessing_rng_before_fit = rng_digest(np.random)
            result = super().train(x_train, y_train, x_val, y_val)
            self.pfn_rng_after_fit = rng_digest(self.rng)
            self.pfn_native_preprocessor = {
                "feature_mirror": self.preprocessor.mirror.tolist(),
                "regression_mirror": bool(self.preprocessor.regression_mirror),
                "inner_y_min": float(self.preprocessor.y_min),
                "inner_y_max": float(self.preprocessor.y_max),
                "singular_features_removed": self.preprocessor.singular_features.tolist(),
            }
            require(self.pfn_optimizer_step_attempts == 0, "Unexpected optimizer update")
            return result
    return GuardedTrainer8


def load_runtime(stage, weights_manifest):
    # The historical helper verifies pinned weights and offline `main`. It
    # installs a one-member guard, which is replaced BEFORE any model creation.
    adapter, weight = historical.load_historical_adapter(stage, weights_manifest)
    modules, sources = {}, {}
    for name, expected in RUNTIME_PINS.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        require(sha256_file(path) == expected, f"Audited native runtime source changed: {name}")
        modules[name] = module
        sources[name] = {"path": str(path), "sha256": expected}
    si = modules["autogluon.tabular.models.mitra.sklearn_interface"]
    trainer_module = modules["autogluon.tabular.models.mitra._internal.core.trainer_finetune"]
    si.TrainerFinetune = guarded_trainer8(trainer_module.TrainerFinetune)
    adapter.reset_seed(42)
    estimator = si.MitraRegressor(n_estimators=8, device="cuda", fine_tune=False,
        fine_tune_steps=0, hf_model=historical.REGRESSOR_REPO, seed=42, verbose=False)
    return estimator, weight, sources


def verify_fitted_ensemble(estimator, support_rows):
    require(estimator.n_estimators == 8 and estimator.fine_tune is False
            and estimator.fine_tune_steps == 0 and estimator.seed == 42,
            "Strict-eight native estimator constructor contract changed")
    trainers = estimator.trainers
    require(len(trainers) == 8 and len({id(t) for t in trainers}) == 8,
            "Exactly eight distinct native trainers required")
    require(len({id(t.model) for t in trainers}) == 8, "Native trainers must own eight distinct models")
    require(len(estimator.X) == len(estimator.y) == support_rows, "Full supplied support changed")
    require(len({id(t.rng) for t in trainers}) == 1, "Native ensemble must share one advancing RNG")
    require(len({t.pfn_rng_before_init for t in trainers}) == 8,
            "Native ensemble RNG did not advance distinctly across eight members")
    require(len({t.pfn_preprocessing_rng_before_fit for t in trainers}) == 8,
            "Native preprocessing RNG did not advance distinctly across eight members")
    for trainer in trainers:
        check_cfg8(trainer.cfg)
        require(trainer.pfn_fit_calls == 1 and trainer.pfn_support_rows == support_rows
                and trainer.pfn_optimizer_step_attempts == 0,
                "Native member fit/support/no-update contract changed")
    return trainers


def predict_actual8(estimator, query, np):
    """Audit the native mean and one actual model contribution/member/test row."""
    trainers = verify_fitted_ensemble(estimator, len(estimator.y))
    n_query, support_rows = len(query), len(estimator.y)
    require(n_query > 0, "Empty official test split")
    expected_chunks = math.ceil(n_query / 1024)
    contributions = np.zeros((8, n_query), dtype=np.uint8)
    member_outputs, records, hooks, originals = [None] * 8, [], [], []
    query_snapshot = query.copy()
    query_known = ~np.isnan(query_snapshot)
    support_hash, target_hash = array_digest(estimator.X), array_digest(estimator.y)
    for member, trainer in enumerate(trainers):
        record = {"member_index": member, "native_cfg_seed": trainer.cfg.seed,
                  "native_rng_before_init_sha256": trainer.pfn_rng_before_init,
                  "native_rng_after_fit_sha256": trainer.pfn_rng_after_fit,
                  "native_preprocessing_rng_before_fit_sha256": trainer.pfn_preprocessing_rng_before_fit,
                  "native_preprocessor": trainer.pfn_native_preprocessor,
                  "native_rng_before_predict_sha256": None,
                  "predict_calls": 0, "test_forward_calls": 0,
                  "query_chunk_ranges": [], "forward_contexts": [], "observed_model_output_shapes": [],
                  "test_rows_contributed": 0, "native_config": check_cfg8(trainer.cfg)}
        records.append(record)
        original = trainer.predict
        originals.append((trainer, original))

        def predict_member(xs, ys, xt, *, member=member, trainer=trainer, record=record, original=original):
            require(array_digest(xs) == support_hash and array_digest(ys) == target_hash
                    and xt.shape == query_snapshot.shape
                    and np.array_equal(xt[query_known], query_snapshot[query_known]),
                    "Native member received different support/query input")
            # Stock Preprocessor fills query NaNs in place; later members see
            # the same support-fitted means. Preserve that historical behavior
            # while asserting all original nonmissing entries and row order.
            record["query_entry_sha256"] = array_digest(xt)
            record["predict_calls"] += 1
            require(record["predict_calls"] == 1, "Native member predicted more than once")
            record["native_rng_before_predict_sha256"] = rng_digest(trainer.rng)
            raw_prediction = original(xs, ys, xt)
            native_prediction = np.asarray(raw_prediction)
            require(native_prediction.shape in ((n_query,), (n_query, 1))
                    and np.isfinite(native_prediction).all(),
                    "Invalid native member prediction")
            prediction = native_prediction.reshape(-1)
            member_outputs[member] = prediction.copy()
            record["native_prediction_shape"] = list(native_prediction.shape)
            record["prediction_sha256"] = array_digest(prediction)
            record["optimizer_step_attempts"] = trainer.pfn_optimizer_step_attempts
            # Preserve the native scalar/vector return shape for its averaging
            # implementation. Flatten only the separate audit copy.
            return raw_prediction

        def forward_check(module, inputs, output, *, member=member, record=record):
            require(len(inputs) == 6, "Unaudited native Mitra forward signature")
            xs, ys, xq, _padding_features, padding_support, padding_query = inputs
            start = record["test_rows_contributed"]
            chunk_rows = min(1024, n_query - start)
            end = start + chunk_rows
            require(chunk_rows > 0 and tuple(xq.shape[:2]) == (1, chunk_rows)
                    and tuple(xs.shape[:2]) == (1, min(8192, support_rows)),
                    "Native model changed context/query rows or added hidden test ensembles")
            require(not bool(padding_query.any()) and not bool(padding_support.any()),
                    "Unexpected padded rows in native single-table prediction")
            output_shape = tuple(output.shape)
            require(module.dim_output == 1 and output_shape in ((1, chunk_rows), (1, chunk_rows, 1)),
                    "Native scalar-head output shape changed")
            record["observed_model_output_shapes"].append(list(output_shape))
            record["test_forward_calls"] += 1
            require(record["test_forward_calls"] <= expected_chunks, "Too many test chunk forwards per native member")
            xsh, ysh, xqh = (array_digest(value.detach().float().cpu().numpy()) for value in (xs, ys, xq))
            record["query_chunk_ranges"].append([start, end])
            record["forward_contexts"].append({"query_start": start, "query_end": end,
                "support_sha256": xsh, "targets_sha256": ysh, "query_sha256": xqh})
            record.update(actual_support_rows=int(xs.shape[1]), actual_query_rows=n_query,
                          forward_context_sha256=common.object_digest(record["forward_contexts"]))
            contributions[member, start:end] += 1
            record["test_rows_contributed"] = end
        trainer.predict = predict_member
        hooks.append(trainer.model.register_forward_hook(forward_check))
    try:
        averaged = np.asarray(estimator.predict(query)).reshape(-1)
    finally:
        for hook in hooks:
            hook.remove()
        for trainer, original in originals:
            trainer.predict = original
    require(all(p is not None for p in member_outputs) and bool((contributions == 1).all()),
            "Every test row must receive exactly one contribution from each of eight members")
    require(all(r["predict_calls"] == 1 and r["test_forward_calls"] == expected_chunks
                and r["test_rows_contributed"] == n_query
                and r["optimizer_step_attempts"] == 0 for r in records),
            "Eight native member calls/no-update verification failed")
    expected_mean = sum(member_outputs) / 8
    require(averaged.shape == (n_query,) and np.isfinite(averaged).all()
            and np.array_equal(averaged, expected_mean), "Native output is not the arithmetic mean of the eight members")
    require(len({r["native_rng_before_predict_sha256"] for r in records}) == 8,
            "Native predict RNG did not advance across all members")
    # Native random preprocessing/context draws may legitimately coincide.
    # Actual-eight means eight separately executed native members covering all
    # rows, not eight guaranteed-distinct transforms or numeric predictions.
    distinct_contexts = len({r["forward_context_sha256"] for r in records})
    for trainer in trainers:
        check_cfg8(trainer.cfg)
    return averaged, {"actual_estimators": 8, "actual_ensemble_count": 8,
        "actual_members_per_test_row": 8, "actual8_verified": True,
        "all_test_rows_covered": True, "test_rows": n_query,
        "minimum_contributions_per_test_row": int(contributions.sum(axis=0).min()),
        "maximum_contributions_per_test_row": int(contributions.sum(axis=0).max()),
        "contribution_matrix_sha256": array_digest(contributions),
        "total_test_model_forward_calls": sum(r["test_forward_calls"] for r in records),
        "native_query_chunks_per_member": expected_chunks,
        "native_arithmetic_mean_verified": True, "distinct_native_forward_contexts": distinct_contexts,
        "member_audits": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source-result", "mitra-stage", "weights-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    require(1 <= args.threads <= 64, "Invalid CPU thread count")
    require(not args.output.exists() and not args.output.is_symlink(), "Refusing to overwrite a result")
    require(sha256_file(historical.__file__) == HISTORICAL_WORKER_SHA256, "Historical helper changed")
    require(sha256_file(common.__file__) == historical.RAW_LOADER_SHA256, "Frozen raw-data helper changed")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ.update(HF_HOME=str(args.mitra_stage / "hf_cache"), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    manifest, row, checkpoint = common.load_manifest(args.manifest, 22175, args.dataset_index)
    reference = load_source_result224(args.source_result, manifest, row, checkpoint)
    reference_sha = sha256_file(args.source_result)
    started = time.monotonic()
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    gpu = historical.gpu_identity(torch)
    torch.cuda.reset_peak_memory_stats(0)
    with threadpool_limits(limits=args.threads):
        support, ys, test, yt, data_audit, Transform, transform_source = common.load_raw_data(manifest, row, np, pd)
        transform = Transform.fit(ys)
        verify_data_parity224(data_audit, reference, transform_source, transform.public_record(), row["suite"])
        encoder = importlib.import_module("official_talent_regression_worker")
        xs, xt, encoding = encoder.support_only_numeric_encoding(support, test)
        require(xs.shape == support.shape and xt.shape == test.shape, "Numeric encoding changed rows/features")
        model_y = transform.transform(ys)
        estimator, weight, runtime_sources = load_runtime(args.mitra_stage, args.weights_manifest)
        fit_started = time.monotonic()
        estimator.fit(xs, model_y, X_val=xs, y_val=model_y)
        fit_seconds = time.monotonic() - fit_started
        predict_started = time.monotonic()
        predicted_transformed, ensemble_audit = predict_actual8(estimator, xt, np)
        prediction = np.asarray(transform.inverse_transform(predicted_transformed), dtype=np.float64).reshape(-1)
        predict_seconds = time.monotonic() - predict_started
        require(prediction.shape == yt.shape and np.isfinite(prediction).all(), "Invalid full-test prediction")
        metrics = {"rmse": float(np.sqrt(mean_squared_error(yt, prediction))),
                   "r2": float(r2_score(yt, prediction, force_finite=True)),
                   "mae": float(mean_absolute_error(yt, prediction))}
        require(all(math.isfinite(x) for x in metrics.values()), "Nonfinite metrics")
        torch.cuda.synchronize(0)
        require(sha256_file(args.source_result) == reference_sha, "Reference changed during evaluation")
        result = {"schema_version": 1, "complete": True, "protocol_validation": True,
            "model_name": "mitra", "experiment": RECIPE["experiment"],
            "model_repo": weight["repo_id"], "model_revision": weight["revision"],
            "model_sha256": weight["sha256"], "dataset_index": row["dataset_index"],
            "dataset": row["dataset"], "row_id": row["row_id"], "suite": row["suite"], "task_kind": "regression",
            "manifest_id": manifest["manifest_id"], "input_fingerprint": row["input_fingerprint"],
            "dataset_protocol_fingerprint": row["protocol_fingerprint"],
            "metrics": metrics, **metrics, "protocol": RECIPE,
            "protocol_fingerprint": common.object_digest(RECIPE),
            "data_audit": data_audit, "source_result_audit_match": True,
            "source_checkpoint_step": 22175,
            "source_result": {"path": str(args.source_result.resolve()), "sha256": reference_sha},
            "target_transform": transform.public_record(), "target_transform_source": transform_source,
            "feature_encoding": encoding, "ensemble_audit": ensemble_audit,
            "actual_estimators": 8, "actual_ensemble_count": 8, "actual8_verified": True,
            "actual_members_per_test_row": 8,
            "native_context_audit": {"official_support_rows": len(ys), "native_support_cap": 8192,
                "actual_support_rows_per_member": min(8192, len(ys)), "context_is_capped": len(ys) > 8192,
                "native_query_cap": 1024, "test_rows": len(yt), "full_official_test": True,
                "equal_full_context_claim": False, "precision": "bfloat16"},
            "runtime_sources": runtime_sources, "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds, "elapsed_seconds": time.monotonic() - started,
            "prediction_sha256": array_digest(prediction), "cpu_threads": torch.get_num_threads(),
            "gpu": gpu, "node": socket.gethostname(), "pid": os.getpid(),
            "parent_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                         "pandas": str(pd.__version__), "autogluon.tabular": "1.5.0"},
            "historical_worker_sha256": HISTORICAL_WORKER_SHA256,
            "worker_source_sha256": sha256_file(__file__)}
        common.publish_new(args.output, result)
    print(json.dumps({"complete": True, "model_name": "mitra", "actual_estimators": 8,
        "dataset_index": args.dataset_index, "metrics": metrics, "output": str(args.output)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
