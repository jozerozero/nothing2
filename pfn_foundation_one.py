#!/usr/bin/env python3
"""Evaluate one pinned official foundation regressor on one frozen PFN28 row.

This extends the previously validated 196-row foundation protocol. It reuses
the frozen raw-data loader and official support-only prediction adapters, and
requires exact agreement with the original unfinetuned Loop3-22175 data audit.
It never trains, downloads models, resplits data, drops test rows, retries a
failed dataset, or overwrites a result. TabPFN's existing query-chunk/OOM policy
and LimiX's official retrieval configuration are retained and recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import eval_one as native


MODELS = {"tabiclv2": "tabicl", "tabpfn2": "tabpfn", "tabpfn25": "tabpfn",
          "tabpfn3": "tabpfn", "limix2m": "limix", "limix16m": "limix"}
VERSIONS = {"tabpfn2": "v2", "tabpfn25": "v2.5", "tabpfn3": "v3"}
require = native.require


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_model(path, expected_sha256, model_name):
    require(sha256_file(path) == expected_sha256, "Prior196 model manifest SHA256 changed")
    payload = json.loads(path.read_text())
    records = payload.get("models", [])
    require(payload.get("complete") is True and payload.get("official_pretrained_models") is True,
            "Model manifest is not a completed official-pretrained inventory")
    require(len(records) == 6 and {r["name"] for r in records} == set(MODELS),
            "Model manifest does not contain exactly the six prior196 models")
    model = next(r for r in records if r["name"] == model_name)
    require(model.get("complete") is True and model.get("family") == MODELS[model_name],
            "Selected official model family is inconsistent")
    checkpoint = Path(model["path"])
    before = checkpoint.stat()
    require(before.st_size == model["size_bytes"] > 0, "Checkpoint size changed")
    require(sha256_file(checkpoint) == model["sha256"], "Checkpoint SHA256 changed")
    after = checkpoint.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), "Checkpoint changed during hashing")
    source = Path(model["source_root"])
    require(source.is_dir(), "Official model source directory is missing")
    if model["family"] in ("tabpfn", "limix"):
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        require(head == model["source_commit"], "Official model source commit changed")
    if model["family"] == "tabpfn":
        require(model.get("version") == VERSIONS[model_name], "TabPFN version mismatch")
    if model["family"] == "limix":
        require(sha256_file(model["config"]) == model["config_sha256"], "LimiX config changed")
    return model


def checked_source_result(path, manifest, row, checkpoint, audit, transform, transform_source):
    result = json.loads(path.read_text())
    require(result.get("complete") is True and result.get("checkpoint_step") == 22175,
            "Reference must be a completed original step22175 result")
    require(result.get("checkpoint") == checkpoint and checkpoint.get("finetune_step") == 0,
            "Reference is not the original unfinetuned checkpoint")
    for key, value in (("dataset_index", row["dataset_index"]), ("row_id", row["row_id"]),
                       ("dataset", row["dataset"]), ("suite", "PFN"),
                       ("manifest_id", manifest["manifest_id"]),
                       ("protocol_fingerprint", manifest["protocol_fingerprint"]),
                       ("dataset_protocol_fingerprint", row["protocol_fingerprint"]),
                       ("input_fingerprint", row["input_fingerprint"])):
        require(result.get(key) == value, f"Original result identity mismatch: {key}")
    require(result.get("strict_checkpoint_load") is True and result.get("checkpoint_load_weights_only") is True,
            "Original checkpoint did not pass strict loading")
    for key, value in audit.items():
        require(result["data_audit"].get(key) == value, f"Original/raw foundation data disagree: {key}")
    require(result["target_transform"] == transform.public_record() and
            result["target_transform_source"] == transform_source, "Support-only outer transform differs")
    require(all(math.isfinite(float(result["metrics"][key])) for key in ("rmse", "mae", "r2")),
            "Original reference metrics are nonfinite")
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "checkpoint_step": 22175, "checkpoint_sha256": checkpoint["sha256"],
            "exact_data_audit_match": True, "metrics": result["metrics"]}


def predict(model, support, ys, test, adapter, suite, chunk):
    """Invoke the existing validated adapters, without copying model logic."""
    source = Path(model["source_root"])
    family = model["family"]
    helper_args = SimpleNamespace(source_root=source, model_path=Path(model["path"]),
        config=Path(model["config"]) if model.get("config") else None,
        random_state=42, n_estimators=8, inference_batch_size=8)
    sys.path.insert(0, str(source / "src" if family in ("tabpfn", "tabicl") else source))
    if family == "tabpfn":
        estimator = suite.make_tabpfn(VERSIONS[model["name"]], helper_args.model_path)
        module = importlib.import_module("tabpfn")
        require(Path(module.__file__).resolve().is_relative_to(source.resolve()), "Unexpected TabPFN source")
        prediction, record = suite.predict_tabpfn(estimator, support, ys, test, chunk)
    elif family == "limix":
        estimator = adapter.load_limix(helper_args)
        module = importlib.import_module("inference.predictor")
        require(Path(module.__file__).resolve().is_relative_to(source.resolve()), "Unexpected LimiX source")
        prediction, record = adapter.predict_limix(estimator, support, ys, test)
    else:
        # Import provenance is checked before the helper constructs its model.
        module = importlib.import_module("tabicl")
        require(Path(module.__file__).resolve().is_relative_to(source.resolve()), "Unexpected TabICL source")
        prediction, record = adapter.predict_tabicl(helper_args, support, ys, test)
    return prediction, record, {"path": str(Path(module.__file__).resolve()),
                                "sha256": sha256_file(module.__file__)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--model-name", choices=sorted(MODELS), required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--test-chunk", type=int, default=4096)
    args = parser.parse_args(argv)
    require(1 <= args.threads <= 64, "Invalid CPU thread count")
    require(args.test_chunk == 4096, "Retain the prior196 initial TabPFN query chunk of 4096")
    require(not args.output.exists() and not args.output.is_symlink(), "Refusing to overwrite result")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    manifest, row, checkpoint = native.load_manifest(args.manifest, 22175, args.dataset_index)
    require(row["suite"] == "PFN" and sum(r["suite"] == "PFN" for r in manifest["rows"]) == 28,
            "This worker is restricted to the exact frozen PFN28 inventory")
    require(row.get("membership_source") == "official_PFN28_regression", "Unexpected PFN membership source")
    model = checked_model(args.model_manifest, args.model_manifest_sha256, args.model_name)
    started = time.monotonic()
    import numpy as np
    import pandas as pd
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    physical_gpu = native.gpu_identity(torch)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.reset_peak_memory_stats(0)
    with threadpool_limits(limits=args.threads):
        support, ys, test, yt, audit, Transform, transform_source = native.load_raw_data(manifest, row, np, pd)
        transform = Transform.fit(ys)
        reference = checked_source_result(args.source_result, manifest, row, checkpoint,
                                          audit, transform, transform_source)
        adapter = importlib.import_module("official_talent_regression_worker")
        suite = importlib.import_module("regression_suite_worker")
        require(adapter.RegressionTargetTransform is Transform and suite.RegressionTargetTransform is Transform,
                "Foundation adapters and source must use the identical frozen outer-target implementation")
        prediction, transform_details, imported_source = predict(
            model, support, ys, test, adapter, suite, args.test_chunk)
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        require(prediction.shape == yt.shape and np.isfinite(prediction).all(), "Invalid full-test predictions")
        require(transform_details["target_transform_record"] == transform.public_record(),
                "Foundation prediction used a different outer target transform")
        metrics = suite.metrics(yt, prediction, ys)
        require(all(math.isfinite(float(metrics[key])) for key in ("rmse", "mae", "r2", "support_normalized_rmse")),
                "Nonfinite test metrics")
        torch.cuda.synchronize(0)
        protocol = {"parent_foundation_protocol": "regression_six_models_six_suites_bg7_20260822_v3",
                    "official_split": "OpenML repeat=0/fold=0; unchanged row order",
                    "full_test_split": True, "test_rows_filtered": 0,
                    "external_support_subsampling": False, "external_feature_subsampling": False,
                    "feature_transform_fit_split": "support_only", "target_transform_fit_split": "support_only",
                    "random_state": 42, "precision": "FP32", "amp": False,
                    "outer_target_transform": transform.public_record(),
                    "model_family": model["family"], "n_estimators": 8,
                    "tabicl_inference_batch_size": 8 if model["family"] == "tabicl" else None,
                    "tabpfn_initial_test_chunk": 4096 if model["family"] == "tabpfn" else None,
                    "tabpfn_minimum_test_chunk_used": transform_details.get("minimum_test_chunk_used"),
                    "tabpfn_oom_chunk_backoff": model["family"] == "tabpfn",
                    "native_limix_retrieval_config": json.loads(Path(model["config"]).read_text())
                        if model["family"] == "limix" else None,
                    "same_inference_settings_as_loop3": False,
                    "prediction_units": "original_target_units", "training_performed": False}
        # This audit is explicitly about raw inputs, not model-native retrieval.
        audit = {**audit, "data_route": "official raw DataFrame -> support-fitted canonicalizer -> official foundation adapter",
                 "query_chunking": bool(model["family"] == "tabpfn" and
                                        transform_details["minimum_test_chunk_used"] < len(yt)),
                 "support_subsampling": False,
                 "support_subsampling_scope": "external input loader only; native LimiX retrieval retained"}
        legacy_row = {"row_id": row["row_id"], "suite": "PFN", "dataset": row["dataset"],
                      "dataset_index": args.dataset_index, "support_rows": len(ys), "test_rows": len(yt),
                      "features": support.shape[1], **metrics, **transform_details}
        result = {"schema_version": 1, "complete": True, "model_name": args.model_name,
                  "model_sha256": model["sha256"], "model": model,
                  "model_manifest_sha256": args.model_manifest_sha256,
                  "manifest_id": manifest["manifest_id"], "membership_sha256": manifest["membership_sha256"],
                  "dataset_protocol_fingerprint": row["protocol_fingerprint"],
                  "input_fingerprint": row["input_fingerprint"],
                  "row_id": row["row_id"], "dataset_index": args.dataset_index,
                  "dataset": row["dataset"], "suite": "PFN", "task_kind": "regression",
                  "row": legacy_row, "metrics": metrics, **metrics, "data_audit": audit,
                  "protocol": protocol, "protocol_fingerprint": native.object_digest(protocol),
                  "source_comparison": reference, "source_result_audit_match": True,
                  "target_transform": transform.public_record(),
                  "target_transform_source": transform_source, "transform_details": transform_details,
                  "adapter_sources": manifest["data_loader"]["vendor_files"],
                  "imported_model_source": imported_source,
                  "prediction_sha256": hashlib.sha256(prediction.astype("<f8").tobytes()).hexdigest(),
                  "elapsed_seconds": time.monotonic() - started, "cpu_threads": args.threads,
                  "gpu": physical_gpu, "node": socket.gethostname(), "pid": os.getpid(),
                  "parent_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
                  "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
                  "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__), "pandas": str(pd.__version__)},
                  "worker_source_sha256": sha256_file(__file__),
                  "native_data_worker_sha256": sha256_file(native.__file__)}
        native.publish_new(args.output, result)
    print(json.dumps({"complete": True, "model_name": args.model_name, "dataset_index": args.dataset_index,
                      "dataset": row["dataset"], "metrics": metrics, "output": str(args.output.resolve())},
                     allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
