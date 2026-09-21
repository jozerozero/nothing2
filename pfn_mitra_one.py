#!/usr/bin/env python3
"""Run the historical original-Mitra v3 recipe on one frozen PFN28 membership.

The unmodified eval_one.load_raw_data provides official repeat=0/fold=0
support/test data. Its complete raw-data audit and fitted outer target transform
must match an already-complete ORIGINAL Loop3 step-22175 receipt before fit.
Mitra keeps its historical one-estimator/BF16/8192-support/1024-query recipe;
these native inference differences are reported separately from raw-data parity.
No downloads, training, resplitting, implicit fallback, or result overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time

import eval_one as common


HISTORICAL_ADAPTER_SHA256 = "1f1b81e0f757026a2b94ae1b6f205776f2c3d695581f2278b79d092afc3340c7"
RAW_LOADER_SHA256 = "73dffec1851e73c6a188218d375b79b8ba864b3b58772e7edeba93ba61ab03d5"
REGRESSOR_REPO = "autogluon/mitra-regressor"
REGRESSOR_REVISION = "5f277aa8f69042d39d6ac3612aed18bb9279bd95"
REGRESSOR_SHA256 = "d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642"
RECIPE = {
    "model_name": "mitra", "model_generation": "original_v1",
    "historical_campaign": "mitra_all_classification_regression_20260823_v3",
    "autogluon_tabular": "1.5.0", "n_estimators": 1,
    "fine_tune": False, "max_epochs": 0, "seed": 0,
    "precision": "bfloat16", "max_samples_support": 8192,
    "max_samples_query": 1024, "full_test_split": True,
    "outer_target_transform": "frozen support-only gt_aware identity/asinh",
    "initialization_validation": "complete support, same as historical v3",
    "inner_target_transform": "official Mitra support-fitted min-max",
    "raw_support_subsampling": False, "model_native_support_cap": 8192,
    "feature_subsampling": False, "prediction_units": "original_target_units",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_result(path, manifest, row, checkpoint):
    """Bind the reference to the exact original checkpoint and PFN membership."""
    source = json.loads(Path(path).read_text())
    common.require(row["suite"] == "PFN" and row["task_kind"] == "regression",
                   "Only PFN28 regression memberships are allowed")
    common.require(sum(r["suite"] == "PFN" for r in manifest["rows"]) == 28,
                   "Manifest must contain exactly PFN28")
    expected = {
        "complete": True, "checkpoint_step": 22175,
        "dataset_index": row["dataset_index"], "row_id": row["row_id"],
        "dataset": row["dataset"], "suite": "PFN", "task_kind": "regression",
        "manifest_id": manifest["manifest_id"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
        "dataset_protocol_fingerprint": row["protocol_fingerprint"],
        "input_fingerprint": row["input_fingerprint"],
    }
    for key, value in expected.items():
        common.require(source.get(key) == value, f"Original source result mismatch: {key}")
    common.require(source.get("checkpoint") == checkpoint and checkpoint.get("finetune_step") == 0
                   and checkpoint.get("kind") == "source_baseline",
                   "Reference must be the original unfinetuned step-22175 checkpoint")
    common.require(source.get("strict_checkpoint_load") is True
                   and source.get("checkpoint_load_weights_only") is True,
                   "Original source checkpoint load was not verified")
    calls = source.get("actual_forward_block_calls", [])
    common.require(calls and all(call == [3] * 12 for call in calls),
                   "Original source receipt does not verify Loop3")
    common.require(all(math.isfinite(float(source["metrics"][key]))
                       for key in ("rmse", "mae", "r2")), "Original source metrics are nonfinite")
    return source


def verify_data_parity(data_audit, source, transform_record, transform):
    # Compare every raw-audit field, including source identities, row-id hashes,
    # canonical frames, targets, row ordering, and the support-only schema.
    common.require(data_audit == source.get("data_audit"),
                   "Raw support/test data audit differs from original step-22175")
    common.require(data_audit["split"]["split"] == "OpenML official repeat=0/fold=0",
                   "PFN split is not official repeat=0/fold=0")
    for key in ("canonical_support_frame_sha256", "canonical_test_frame_sha256",
                "support_targets_sha256", "test_targets_sha256"):
        common.require(isinstance(data_audit.get(key), str) and len(data_audit[key]) == 64,
                       f"Missing exact-data hash: {key}")
    common.require(transform_record == source.get("target_transform_source"),
                   "Outer target transform source differs from original step-22175")
    common.require(transform == source.get("target_transform"),
                   "Support-fitted outer target transform differs from original step-22175")


def check_trainer_cfg(cfg):
    expected = {"max_epochs": 0, "max_samples_support": 8192,
                "max_samples_query": 1024, "precision": "bfloat16",
                "dim_output": 1, "n_ensembles": 1, "grad_scaler_enabled": False}
    for key, value in expected.items():
        common.require(cfg.hyperparams.get(key) == value,
                       f"Historical original-Mitra trainer configuration changed: {key}")
    return expected


def guarded_trainer(si):
    """Add fail-closed guards without changing the historical estimator recipe."""
    original = si.TrainerFinetune

    class GuardedTrainer(original):
        def __init__(self, cfg, model, *args, **kwargs):
            check_trainer_cfg(cfg)  # Reject any training before trainer construction.
            common.require(model.dim_output == 1 and not model.use_flash_attn,
                           "Expected original regression head and stock ROCm attention")
            super().__init__(cfg, model, *args, **kwargs)
            self.pfn_optimizer_step_attempts = 0

            def forbidden_step(*_args, **_kwargs):
                self.pfn_optimizer_step_attempts += 1
                raise RuntimeError("optimizer.step is prohibited for original-Mitra inference")

            self.optimizer.step = forbidden_step

        def train(self, *args, **kwargs):
            check_trainer_cfg(self.cfg)
            result = super().train(*args, **kwargs)
            common.require(self.pfn_optimizer_step_attempts == 0,
                           "Unexpected optimizer step during original-Mitra fit")
            return result

    return GuardedTrainer


def load_historical_adapter(stage, weights_manifest):
    adapter_path = Path(stage) / "mitra_common.py"
    common.require(sha256_file(adapter_path) == HISTORICAL_ADAPTER_SHA256,
                   "Historical v3 Mitra adapter identity changed")
    version = importlib.metadata.version("autogluon.tabular")
    common.require(version == "1.5.0", f"Historical Mitra requires AutoGluon 1.5.0, found {version}")
    adapter = common.import_path("_pfn_original_mitra_v3", adapter_path)
    weight = adapter.validate_weights(Path(weights_manifest), "regressor")
    common.require((weight["repo_id"], weight["revision"], weight["sha256"]) ==
                   (REGRESSOR_REPO, REGRESSOR_REVISION, REGRESSOR_SHA256),
                   "Weights differ from historical original-Mitra v3")
    from huggingface_hub import hf_hub_download
    # The historical constructor loads `main`. Check its actual local resolution,
    # not merely the pinned revision, to avoid silently loading another snapshot.
    for filename, expected in (("model.safetensors", weight["sha256"]),
                               ("config.json", weight["config_sha256"])):
        resolved = Path(hf_hub_download(repo_id=REGRESSOR_REPO, filename=filename,
                                       revision="main", local_files_only=True))
        common.require(sha256_file(resolved) == expected,
                       f"Offline main does not resolve to historical pinned {filename}")
    import autogluon.tabular.models.mitra.sklearn_interface as si
    si.TrainerFinetune = guarded_trainer(si)
    return adapter, weight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--mitra-stage", type=Path, required=True)
    parser.add_argument("--weights-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    common.require(1 <= args.threads <= 64, "threads outside supported bounds")
    common.require(not args.output.exists() and not args.output.is_symlink(),
                   "Result already exists; refusing overwrite")
    common.require(sha256_file(common.__file__) == RAW_LOADER_SHA256,
                   "Frozen eval_one raw-data helper changed")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["HF_HOME"] = str(args.mitra_stage / "hf_cache")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    manifest, row, checkpoint = common.load_manifest(args.manifest, 22175, args.dataset_index)
    source = load_source_result(args.source_result, manifest, row, checkpoint)
    source_sha256 = sha256_file(args.source_result)
    started = time.monotonic()
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    physical_gpu = common.gpu_identity(torch)
    torch.cuda.reset_peak_memory_stats(0)
    with threadpool_limits(limits=args.threads):
        support, ys, test, yt, data_audit, TargetTransform, transform_record = common.load_raw_data(
            manifest, row, np, pd)
        transform = TargetTransform.fit(ys)
        verify_data_parity(data_audit, source, transform_record, transform.public_record())
        # This is the same support-only encoding used by the historical v3
        # worker, loaded from the manifest-verified vendor helper.
        old_worker = sys.modules["official_talent_regression_worker"]
        xs, xt, encoding = old_worker.support_only_numeric_encoding(support, test)
        common.require(xs.shape == support.shape and xt.shape == test.shape,
                       "Historical numeric encoding changed feature shape or row count")
        adapter, weight = load_historical_adapter(args.mitra_stage, args.weights_manifest)
        adapter.reset_seed(0)
        estimator = adapter.make_regressor()
        common.require(estimator.n_estimators == 1 and estimator.fine_tune is False and estimator.seed == 0,
                       "Historical Mitra estimator configuration changed")
        fit_started = time.monotonic()
        estimator.fit(xs, transform.transform(ys))
        fit_seconds = time.monotonic() - fit_started
        common.require(len(estimator.trainers) == 1 and len(estimator.X) == len(ys),
                       "Mitra changed full support or estimator count")
        trainer = estimator.trainers[0]
        actual_cfg = check_trainer_cfg(trainer.cfg)
        predict_started = time.monotonic()
        predicted_transformed = np.asarray(estimator.predict(xt), dtype=np.float64).reshape(-1)
        prediction = np.asarray(transform.inverse_transform(predicted_transformed), dtype=np.float64).reshape(-1)
        predict_seconds = time.monotonic() - predict_started
        common.require(prediction.shape == yt.shape and np.isfinite(prediction).all(),
                       "Invalid full-test Mitra prediction")
        common.require(trainer.pfn_optimizer_step_attempts == 0, "Unexpected optimizer update")
        metric = {"rmse": float(np.sqrt(mean_squared_error(yt, prediction))),
                  "r2": float(r2_score(yt, prediction, force_finite=True)),
                  "mae": float(mean_absolute_error(yt, prediction))}
        common.require(all(math.isfinite(value) for value in metric.values()), "Nonfinite metrics")
        torch.cuda.synchronize(0)
        common.require(sha256_file(args.source_result) == source_sha256,
                       "Original source receipt changed during evaluation")
        model_audit = {**actual_cfg, "support_rows_before_native_cap": len(ys),
                       "support_subsampled_by_native_model": len(ys) > 8192,
                       "native_query_chunk_size": 1024, "full_test_rows_predicted": len(prediction),
                       "optimizer_step_attempts": trainer.pfn_optimizer_step_attempts,
                       "initialization_validation_rows": len(ys),
                       "test_labels_used_for_fit": False}
        result = {"schema_version": 1, "complete": True, "protocol_validation": True,
            "model_name": "mitra", "model_repo": weight["repo_id"],
            "model_revision": weight["revision"], "model_sha256": weight["sha256"],
            "dataset_index": row["dataset_index"], "dataset": row["dataset"],
            "row_id": row["row_id"], "suite": "PFN", "task_kind": "regression",
            "metrics": metric, **metric, "manifest_id": manifest["manifest_id"],
            "input_fingerprint": row["input_fingerprint"],
            "dataset_protocol_fingerprint": row["protocol_fingerprint"],
            "protocol": RECIPE, "protocol_fingerprint": common.object_digest(RECIPE),
            "comparison_data_protocol_fingerprint": manifest["protocol_fingerprint"],
            "data_audit": data_audit,
            "data_audit_scope": "unchanged shared raw-data audit; historical route label refers to reference only",
            "source_input_parity_verified": True, "source_result_audit_match": True,
            "source_checkpoint_step": 22175,
            "source_result": {"path": str(args.source_result.resolve()), "sha256": source_sha256},
            "feature_encoding": encoding, "target_transform": transform.public_record(),
            "target_transform_source": transform_record, "model_audit": model_audit,
            "prediction_sha256": hashlib.sha256(prediction.astype("<f8").tobytes()).hexdigest(),
            "encoded_support_sha256": hashlib.sha256(xs.tobytes()).hexdigest(),
            "encoded_test_sha256": hashlib.sha256(xt.tobytes()).hexdigest(),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "elapsed_seconds": time.monotonic() - started, "cpu_threads": torch.get_num_threads(),
            "node": socket.gethostname(), "pid": os.getpid(),
            "parent_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "gpu": physical_gpu, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                         "pandas": str(pd.__version__), "autogluon.tabular": "1.5.0"},
            "historical_adapter_sha256": HISTORICAL_ADAPTER_SHA256,
            "raw_loader_sha256": RAW_LOADER_SHA256,
            "worker_source_sha256": sha256_file(__file__)}
        common.publish_new(args.output, result)
    print(json.dumps({"complete": True, "model_name": "mitra", "dataset_index": args.dataset_index,
                      "dataset": row["dataset"], "metrics": metric, "output": str(args.output),
                      "elapsed_seconds": time.monotonic() - started}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
