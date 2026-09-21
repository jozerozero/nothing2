#!/usr/bin/env python3
"""Evaluate one frozen native Loop3 checkpoint/membership on one isolated GPU.

Consumes a prepare_eval224.py manifest. No training, data resplitting, test-row
filtering, query chunking, model downloads, or overwrite/retry are performed.
Raw official support/test DataFrames pass through the frozen canonicalizer and
native TabICL preprocessing; Mitra's final encoded arrays are never consumed.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import tempfile
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_file(record, allow_empty=False):
    path = Path(record["path"])
    before = path.stat()
    require(path.is_file() and (allow_empty or before.st_size > 0), f"Missing input: {path}")
    require((before.st_size, before.st_mtime_ns) == (record["size_bytes"], record["mtime_ns"]),
            f"Frozen file metadata changed: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), f"File changed during read: {path}")
    require(digest.hexdigest() == record["sha256"], f"Frozen file SHA256 changed: {path}")
    return path.resolve()


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot import: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def publish_new(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic publication; never overwrite a result.
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def gpu_identity(torch):
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one CUDA/ROCm GPU must be visible to the runtime")
    require(bool(torch.version.hip), "This sidecar requires the allocated AMD ROCm runtime")
    expected_uuid = os.environ.get("EXPECTED_GPU_UUID", "").lower().removeprefix("gpu-")
    expected_pci = os.environ.get("EXPECTED_GPU_PCI_BUS_ID", "").lower()
    require(expected_uuid and expected_pci, "EXPECTED_GPU_UUID and EXPECTED_GPU_PCI_BUS_ID are mandatory")
    lib = ctypes.CDLL(ctypes.util.find_library("amdhip64") or "/opt/rocm/lib/libamdhip64.so")
    fn = lib.hipDeviceGetPCIBusId
    fn.argtypes, fn.restype = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int], ctypes.c_int
    buffer = ctypes.create_string_buffer(64)
    require(fn(buffer, len(buffer), 0) == 0, "HIP physical PCI lookup failed")
    domain, bus, slot = buffer.value.decode().lower().split(":")
    pci = f"{int(domain, 16):04x}:{bus}:{slot}"
    uuid = (Path("/sys/bus/pci/devices") / pci / "unique_id").read_text().strip().lower()
    require(uuid == expected_uuid and pci == expected_pci, f"Physical GPU mismatch: {(uuid, pci)}")
    torch.cuda.set_device(0)
    smoke = torch.ones((4, 4), dtype=torch.float32, device="cuda:0")
    require(float(smoke.sum().cpu()) == 16, "GPU compute binding probe failed")
    del smoke
    return {"uuid": uuid, "pci_bus_id": pci, "runtime_visible_count": 1,
            "name": torch.cuda.get_device_name(0), "hip_version": str(torch.version.hip),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES")}


def load_manifest(path, step, dataset_index):
    manifest = json.loads(Path(path).read_text())
    require(manifest.get("manifest_id") == object_digest({k: v for k, v in manifest.items() if k != "manifest_id"}),
            "Manifest content identity mismatch")
    require(manifest.get("membership_count") == 224 and len(manifest["rows"]) == 224,
            "Expected all224 regression memberships")
    require(manifest.get("inference_loop") == 3 and manifest.get("source_step") == 22175,
            "Wrong model lineage/loop contract")
    require(manifest["protocol_fingerprint"] == object_digest(manifest["protocol"]), "Protocol fingerprint mismatch")
    require(0 <= dataset_index < 224, "dataset-index outside all224 manifest")
    row = manifest["rows"][dataset_index]
    require(row["dataset_index"] == dataset_index and row["task_kind"] == "regression", "Dataset index mismatch")
    require(row["protocol_fingerprint"] == object_digest({"protocol_fingerprint": manifest["protocol_fingerprint"],
            "row_id": row["row_id"], "input_fingerprint": row["input_fingerprint"]}), "Dataset protocol mismatch")
    selected = [record for record in manifest["checkpoints"] if record["step"] == step]
    require(len(selected) == 1, "checkpoint-step missing or duplicated in manifest")
    expected = {"n_estimators": 8, "batch_size": 1, "random_state": 42,
                "use_amp": False, "use_fa3": False, "kv_cache": False, "n_jobs": None,
                "norm_methods": ["none", "power"], "feat_shuffle_method": "latin",
                "outlier_threshold": 4.0, "offload_mode": "auto",
                "target_transform": "gt_aware", "full_test_split": True,
                "query_chunking": False, "support_subsampling": False,
                "feature_subsampling": False, "dtype": "float32"}
    inference = manifest["protocol"]["inference"]
    for key, value in expected.items():
        require(key in inference and inference[key] == value, f"Inference protocol mismatch: {key}")
    return manifest, row, selected[0]


def load_raw_data(manifest, row, np, pd):
    loader = manifest["data_loader"]
    for identity in loader["vendor_files"].values():
        verify_file(identity)
    helper_path = verify_file(loader["eval_data"])
    sys.path.insert(0, loader["vendor_dir"])
    official = importlib.import_module("standard_loader")
    base = importlib.import_module("regression_suite_worker")
    old_worker = importlib.import_module("official_talent_regression_worker")
    for name, module in (("standard_loader", official), ("regression_suite_worker", base),
                         ("official_talent_regression_worker", old_worker)):
        require(Path(module.__file__).resolve() == Path(loader["vendor_dir"]) / (name + ".py"),
                f"Imported unintended data helper: {name}")
    helper = import_path("_ft50_official_raw_splits", helper_path)
    before = helper.verify_input_files(row)
    helper._require_bound(helper._regression_source_files(row), before)
    if row["suite"] == "PFN":
        support, ys, test, yt, split = helper._pfn_split(row, official)
    else:
        reader = {"talent_npy": official.talent_split, "bcco_csv": official.bcco_split,
                  "openml_arff": official.openml_split}[row["format"]]
        support, ys, test, yt, split = reader(Path(row["source_path"]))
    support_count, test_count = len(ys), len(yt)
    ys = np.asarray(base.numeric_target(ys, row["dataset"] + "/support"), dtype=np.float64)
    yt = np.asarray(base.numeric_target(yt, row["dataset"] + "/test"), dtype=np.float64)
    require(ys.ndim == yt.ndim == 1 and len(ys) == support_count and len(yt) == test_count,
            "Target conversion changed row count or dimensions")
    require(len(ys) >= 2 and len(yt) >= 2 and np.isfinite(ys).all() and np.isfinite(yt).all(),
            "Official targets are invalid; no filtering or imputation is allowed")
    require(float(np.ptp(ys)) > 0, "Constant official support target")
    support, test, schema = base.canonicalize_features(support, test)
    require(len(support) == support_count and len(test) == test_count and list(support.columns) == list(test.columns),
            "Raw official feature rows/schema changed")
    # A support-all-missing feature has no fitted information. Drop it in both
    # frames before native sklearn imputers can change their dimensionality.
    dropped = [str(column) for column in support if bool(support[column].isna().all())]
    if dropped:
        support, test = support.drop(columns=dropped), test.drop(columns=dropped)
    require(support.shape[1] > 0, "No usable support features")
    after = helper.verify_input_files(row)
    require(before == after, "Official inputs changed during data loading")
    def frame_digest(frame):
        result = hashlib.sha256(json.dumps({"columns": list(frame.columns),
                                           "dtypes": [str(x) for x in frame.dtypes]}, sort_keys=True).encode())
        for start in range(0, len(frame), 4096):
            result.update(pd.util.hash_pandas_object(frame.iloc[start:start + 4096], index=False).values.tobytes())
        return result.hexdigest()
    record = {"input_fingerprint": row["input_fingerprint"], "source_identity": before,
              "split": split, "support_rows": support_count, "test_rows": test_count,
              "features": support.shape[1], "feature_schema_record": schema,
              "support_all_missing_columns_dropped": dropped,
              "canonical_support_frame_sha256": frame_digest(support),
              "canonical_test_frame_sha256": frame_digest(test),
              "support_targets_sha256": hashlib.sha256(ys.astype("<f8").tobytes()).hexdigest(),
              "test_targets_sha256": hashlib.sha256(yt.astype("<f8").tobytes()).hexdigest(),
              "support_row_order": "unchanged official order", "test_row_order": "unchanged official order",
              "test_rows_filtered": 0, "test_targets_masked_or_imputed": False,
              "support_subsampling": False, "query_chunking": False,
              "data_route": "official raw DataFrame -> support-fitted canonicalizer -> native TabICL"}
    transform_record = manifest.get("helpers", {}).get("target_transform",
        loader["vendor_files"]["official_talent_regression_worker.py"])
    transform_path = verify_file(transform_record)
    if transform_path == Path(old_worker.__file__).resolve():
        transform_class = old_worker.RegressionTargetTransform
    else:
        transform_class = import_path("_ft50_frozen_target_transform", transform_path).RegressionTargetTransform
    return support, ys, test, yt, record, transform_class, transform_record


def load_regressor(manifest, checkpoint, torch):
    native = manifest["native_source"]
    for identity in native["files"].values():
        verify_file(identity, allow_empty=True)
    require(native["sha256"] == object_digest({key: record["sha256"] for key, record in native["files"].items()}),
            "Native source aggregate hash mismatch")
    sys.path.insert(0, native["path"])
    from tabicl._model.tabicl import TabICL
    from tabicl import TabICLRegressor
    imported = importlib.import_module("tabicl._model.tabicl")
    require(Path(imported.__file__).resolve() == Path(native["path"]) / "tabicl/_model/tabicl.py",
            "Wrong native model import")
    path = verify_file(checkpoint)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(saved.get("curr_step") == checkpoint["step"], "Checkpoint step mismatch")
    if checkpoint["step"] != 22175:
        require(saved.get("source_step") == 22175 and saved.get("finetune_step") == checkpoint["step"] - 22175,
                "Fine-tuning lineage mismatch")
        require(saved.get("training_contract", {}).get("contract_sha256") == checkpoint["training_contract_sha256"],
                "Checkpoint fine-tuning contract mismatch")
    config = saved["config"]
    for key, expected in {"max_classes": 0, "num_quantiles": 999, "icl_num_blocks": 12,
            "shared_depth_icl_enabled": True, "shared_depth_icl_dataset_conditioned": True,
            "shared_depth_icl_num_passes": 3, "shared_depth_icl_rho": 1.0, "bias_free_ln": False}.items():
        require(config.get(key) == expected, f"Native Loop3 regression config mismatch: {key}")
    model = TabICL(**config).float()
    model.load_state_dict(saved["state_dict"], strict=True)
    require(all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()), "Nonfinite model weights")
    del saved
    model.eval()
    regressor = TabICLRegressor(n_estimators=8, batch_size=1, norm_methods=["none", "power"],
        feat_shuffle_method="latin", outlier_threshold=4.0, kv_cache=False,
        model_path=str(path), allow_auto_download=False, device="cuda:0", use_amp=False,
        use_fa3=False, offload_mode="auto", random_state=42, n_jobs=None, verbose=False)
    regressor.model_, regressor.model_config_, regressor.model_path_ = model, config, path
    # A single strict, verified safe load; fit normally reloads the same file.
    regressor._load_model = lambda: None
    return regressor, config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    require(1 <= args.threads <= 64, "threads outside supported bounds")
    require(not args.output.exists() and not args.output.is_symlink(), "Result already exists; refusing overwrite")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(args.threads)
    manifest, row, checkpoint = load_manifest(args.manifest, args.checkpoint_step, args.dataset_index)
    require(args.threads == manifest["protocol"]["inference"].get("torch_threads"),
            "--threads must match the frozen inference protocol")
    started = time.monotonic()
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    physical_gpu = gpu_identity(torch)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.reset_peak_memory_stats(0)
    with threadpool_limits(limits=args.threads):
        support, ys, test, yt, data_audit, TargetTransform, transform_record = load_raw_data(manifest, row, np, pd)
        regressor, config = load_regressor(manifest, checkpoint, torch)
        transform = TargetTransform.fit(ys)  # Frozen exact gt_aware identity/asinh implementation.
        model_labels = transform.transform(ys)
        stack = regressor.model_.icl_predictor.tf_icl
        require(stack.shared_depth_num_passes == 3 and len(stack.blocks) == 12, "Wrong realized ICL stack")
        counts, realized_calls, hooks = [0] * 12, [], []
        def begin(_module, _inputs):
            counts[:] = [0] * 12
        def end(_module, _inputs, _output):
            require(counts == [3] * 12, f"Realized Loop3 block calls differ: {counts}")
            realized_calls.append(list(counts))
        hooks.extend([stack.register_forward_pre_hook(begin), stack.register_forward_hook(end)])
        for index, block in enumerate(stack.blocks):
            def count(_module, _inputs, _output, index=index):
                counts[index] += 1
            hooks.append(block.register_forward_hook(count))
        fit_started = time.monotonic()
        try:
            with torch.inference_mode():
                regressor.fit(support, model_labels)
                fit_seconds = time.monotonic() - fit_started
                predict_started = time.monotonic()
                predicted_transformed = regressor.predict(test.copy(deep=True))
                prediction = np.asarray(transform.inverse_transform(predicted_transformed), dtype=np.float64).reshape(-1)
                predict_seconds = time.monotonic() - predict_started
        finally:
            for hook in hooks:
                hook.remove()
        require(realized_calls and all(values == [3] * 12 for values in realized_calls), "No verified Loop3 forward")
        require(prediction.shape == yt.shape and np.isfinite(prediction).all(), "Invalid full-test prediction")
        metric = {"rmse": float(np.sqrt(mean_squared_error(yt, prediction))),
                  "r2": float(r2_score(yt, prediction, force_finite=True)),
                  "mae": float(mean_absolute_error(yt, prediction))}
        require(all(math.isfinite(value) for value in metric.values()), "Nonfinite official-test metrics")
        torch.cuda.synchronize(0)
        result = {"schema_version": 1, "complete": True, "checkpoint_step": args.checkpoint_step,
            "dataset_index": args.dataset_index, "row_id": row["row_id"], "dataset": row["dataset"],
            "suite": row["suite"], "task_kind": "regression", "metrics": metric, **metric,
            "manifest_id": manifest["manifest_id"], "protocol_fingerprint": manifest["protocol_fingerprint"],
            "dataset_protocol_fingerprint": row["protocol_fingerprint"], "input_fingerprint": row["input_fingerprint"],
            "checkpoint": checkpoint, "native_source_sha256": manifest["native_source"]["sha256"],
            "data_audit": data_audit, "protocol": manifest["protocol"],
            "target_transform": transform.public_record(),
            "target_transform_source": transform_record,
            "inner_target_scaler": "native sklearn StandardScaler, official support only",
            "r2_constant_test_target_policy": "sklearn force_finite=True; no test rows masked",
            "prediction_sha256": hashlib.sha256(prediction.astype("<f8").tobytes()).hexdigest(),
            "strict_checkpoint_load": True, "checkpoint_load_weights_only": True,
            "model_config": config, "actual_forward_block_calls": realized_calls,
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "elapsed_seconds": time.monotonic() - started, "cpu_threads": torch.get_num_threads(),
            "node": socket.gethostname(), "pid": os.getpid(), "parent_pid": os.getppid(),
            "parent_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "gpu": physical_gpu, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__), "pandas": str(pd.__version__)},
            "worker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        publish_new(args.output, result)
    print(json.dumps({"complete": True, "checkpoint_step": args.checkpoint_step,
                      "dataset_index": args.dataset_index, "dataset": row["dataset"],
                      "metrics": metric, "output": str(args.output.resolve()),
                      "elapsed_seconds": time.monotonic() - started}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
