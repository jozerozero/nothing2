#!/usr/bin/env python3
"""One frozen classification457/regression224 membership with native TabFM defaults.

No estimator, dtype, support/feature cap, calibration, or ensembling override.
Only the official PyTorch backend is used. Classification >native class capacity
uses the separately audited, user-authorized balanced class hierarchy, with an
unchanged native-default classifier at each node (not an official TabFM default).
Raw canonical DataFrames enter native preprocessing; encoded classification
caches are used ONLY to verify unchanged official support/test membership.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time
import traceback

import eval_one as common


require = common.require
COMMIT = "fbb665569425fd2f490c6576b3af967876fe11ff"
PROTOCOL = "tabfm-1.0.1-native-defaults-standard681-raw-data-hierarchical-extension-v1"
CLASS_COUNTS = {"talent": 200, "BCCO": 106, "OpenML-CC18": 62,
                "PFN": 29, "TabArena": 33, "TabZilla": 27}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    require(value.dtype.kind != "O", "Object array cannot be content-hashed safely")
    h = hashlib.sha256(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}, sort_keys=True).encode())
    h.update(memoryview(value).cast("B"))
    return h.hexdigest()


def frame_digest(frame, pd):
    h = hashlib.sha256(json.dumps({"columns": list(frame.columns),
            "dtypes": [str(x) for x in frame.dtypes]}, sort_keys=True).encode())
    for start in range(0, len(frame), 4096):
        h.update(pd.util.hash_pandas_object(frame.iloc[start:start + 4096], index=False).values.tobytes())
    return h.hexdigest()


def json_value(value):
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    require(value is None or isinstance(value, (str, int, float, bool)),
            f"Non-JSON native configuration: {type(value).__name__}")
    return value


def verify_manifest(identity):
    path = common.verify_file(identity)
    value = json.loads(path.read_text())
    require(value.get("manifest_id") == common.object_digest({k: v for k, v in value.items() if k != "manifest_id"}),
            f"Frozen manifest digest mismatch: {path}")
    return value


def raw_snapshot(records):
    """Check frozen metadata, then lazily hash original files without modifying them."""
    require(records and len({r["path"] for r in records}) == len(records), "Missing/duplicate raw input identities")
    result = []
    for expected in records:
        path = Path(expected["path"]).resolve(strict=True)
        before = path.stat()
        require(before.st_size == expected["size_bytes"] and before.st_mtime_ns == expected["mtime_ns"],
                f"Raw input metadata changed: {path}")
        h = sha_file(path)
        after = path.stat()
        require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_ino, after.st_size, after.st_mtime_ns), f"Raw file changed while hashing: {path}")
        require("sha256" not in expected or expected["sha256"] == h, f"Raw SHA changed: {path}")
        result.append({"path": str(path), "size_bytes": after.st_size,
                       "mtime_ns": after.st_mtime_ns, "sha256": h})
    return result


def load_campaign(path):
    value = json.loads(Path(path).read_text())
    require(value.get("manifest_id") == common.object_digest({k: v for k, v in value.items() if k != "manifest_id"}),
            "Campaign identity mismatch")
    source = value["official_source"]
    require(source["commit"] == COMMIT, "Wrong official TabFM source revision")
    root = Path(source["path"]).resolve(strict=True)
    files = source["files"]
    require(files, "Official source identities missing")
    for identity in files:
        require(common.verify_file(identity, allow_empty=True).is_relative_to(root), "Source file outside frozen source")
    workers = value["worker_sources"]
    require(workers, "Worker identities missing")
    verified = {common.verify_file(r, allow_empty=True) for r in workers}
    require(Path(__file__).resolve() in verified and Path(common.__file__).resolve() in verified,
            "Both worker and frozen eval_one helper must be pinned")
    for name in ("pfn_mitra_one.py", "tabfm_hierarchical.py"):
        require(Path(__file__).resolve().with_name(name) in verified,
                f"Required worker dependency must be pinned: {name}")
    verify_manifest(value["classification_manifest"])
    verify_manifest(value["regression_manifest"])
    return value


def import_data_helpers(reg_manifest):
    loader = reg_manifest["data_loader"]
    for identity in loader["vendor_files"].values():
        common.verify_file(identity)
    vendor = Path(loader["vendor_dir"]).resolve(strict=True)
    sys.path.insert(0, str(vendor))
    official = importlib.import_module("standard_loader")
    canonical = importlib.import_module("regression_suite_worker")
    for name, module in (("standard_loader", official), ("regression_suite_worker", canonical)):
        require(Path(module.__file__).resolve() == vendor / (name + ".py"), f"Wrong data helper import: {name}")
    return official, canonical


def exact_cache_check(row, arrays, np):
    path = common.verify_file(row["cache"])
    hashes = {}
    with np.load(path, allow_pickle=False) as frozen:
        for name, actual in arrays.items():
            require(name in frozen and np.asarray(actual).dtype == frozen[name].dtype
                    and np.array_equal(actual, frozen[name]),
                    f"{row['dataset']}: exact frozen cache mismatch: {name}")
            hashes[name] = array_digest(frozen[name])
    common.verify_file(row["cache"])
    return hashes


def load_classification(campaign, index, np, pd):
    manifest = verify_manifest(campaign["classification_manifest"])
    rows = manifest["rows"]
    require(manifest["membership_count"] == len(rows) == 457
            and [r["dataset_index"] for r in rows] == list(range(457))
            and dict(Counter(r["suite"] for r in rows)) == CLASS_COUNTS, "Classification457 scope changed")
    require(0 <= index < 457, "Classification index outside0..456")
    row = rows[index]
    require(row["input_fingerprint"] == common.object_digest(row["cache"]), "Classification cache identity mismatch")
    source = Path(row.get("source_path", row.get("rawsource_path", ""))).resolve(strict=True)
    raw_inputs = campaign["classification_raw_inputs"]
    records = raw_inputs.get(str(index), raw_inputs.get(row["dataset"]))
    before = raw_snapshot(records)
    reg_manifest = verify_manifest(campaign["regression_manifest"])
    official, canonical = import_data_helpers(reg_manifest)
    reader = official.talent_split if row["suite"] == "talent" else (
        official.bcco_split if row["suite"] == "BCCO" else official.openml_split)
    tx, ty, vx, vy, split = reader(source)
    raw_support, raw_test = len(ty), len(vy)
    tx, ty, vx, vy, dropped = official.drop_missing_targets(tx, ty, vx, vy)
    ys, yt, class_labels = official.preprocess_labels(ty, vy)
    encoded_train, encoded_test, encoded_audit = official.preprocess_features(tx, vx)
    cache_hashes = exact_cache_check(row, {"X_train": encoded_train, "y_train": ys,
        "X_test": encoded_test, "y_test": yt}, np)
    if "class_labels" in row:
        require(class_labels == row["class_labels"], "Frozen classification vocabulary changed")
    require(len(ys) == row["train_rows"] and len(yt) == row["test_rows"]
            and len(class_labels) == row["classes"] and encoded_train.shape[1] == row["features"],
            "Classification row metadata changed")
    del encoded_train, encoded_test
    # Support-only dtype restoration; no feature encoding or scaling is reused.
    tx, vx, schema = canonical.canonicalize_features(tx, vx)
    dropped_features = [str(name) for name in tx if bool(tx[name].isna().all())]
    if dropped_features:
        tx, vx = tx.drop(columns=dropped_features), vx.drop(columns=dropped_features)
    require(tx.shape[1] > 0 and len(tx) == len(ys) and len(vx) == len(yt), "Raw schema/row count changed")
    require(raw_snapshot(records) == before, "Raw classification inputs changed during loading")
    audit = {"data_route": "official raw split -> exact frozen-cache verification (audit only) -> support-only dtype canonicalization -> native TabFM",
        "source_path": str(source), "source_identity": before, "split": split,
        "cache": row["cache"], "cache_exact_match": True, "cache_arrays_sha256": cache_hashes,
        "cache_feature_audit": encoded_audit, "cached_encoded_features_fed_to_model": False,
        "support_rows": len(ys), "test_rows": len(yt), "raw_support_rows": raw_support,
        "raw_test_rows": raw_test, "classes": len(class_labels), "class_labels": class_labels,
        "frozen_target_filter": dropped, "additional_test_rows_filtered": 0,
        "support_all_missing_columns_dropped": dropped_features, "feature_schema_record": schema,
        "canonical_support_frame_sha256": frame_digest(tx, pd),
        "canonical_test_frame_sha256": frame_digest(vx, pd),
        "support_targets_sha256": array_digest(ys), "test_targets_sha256": array_digest(yt),
        "support_row_order": "unchanged frozen benchmark order", "test_row_order": "unchanged frozen benchmark order",
        "full_test_split": True, "test_labels_used_for_fit": False}
    return manifest, row, tx, ys, vx, yt, audit, before


def load_regression(campaign, index, np, pd):
    path = common.verify_file(campaign["regression_manifest"])
    manifest, row, _ = common.load_manifest(path, 22175, index)
    records = row.get("input_files", row.get("required_files"))
    before = raw_snapshot(records)
    tx, ys, vx, yt, audit, _unused_transform_class, _unused_transform_source = common.load_raw_data(manifest, row, np, pd)
    require(raw_snapshot(records) == before, "Raw regression inputs changed during loading")
    audit = dict(audit, data_route="official raw split -> frozen support-only dtype canonicalizer -> native TabFM",
                 raw_file_sha256_audit=before, full_test_split=True,
                 external_target_transform="none; ORIGINAL target units passed to native TabFM",
                 source_taffy_target_transform_applied=False, test_labels_used_for_fit=False)
    return manifest, row, tx, ys, vx, yt, audit, before


def weight_directory(campaign, task):
    spec = campaign["weights"][task]
    directory = Path(spec["directory"]).resolve(strict=True)
    config_path, weights_path = common.verify_file(spec["config"]), common.verify_file(spec["weights"])
    require(config_path.parent == weights_path.parent == directory
            and config_path.name == "config.json" and weights_path.name == "model.safetensors",
            "Expected direct task directory containing pinned config.json/model.safetensors")
    return directory, json.loads(config_path.read_text())


def load_native(campaign, task, directory):
    root = Path(campaign["official_source"]["path"]).resolve(strict=True)
    sys.path.insert(0, str(root))
    tabfm = importlib.import_module("tabfm")
    backend = importlib.import_module("tabfm.src.pytorch.tabfm_v1_0_0")
    classifier = importlib.import_module("tabfm.src.classifier_and_regressor")
    for module in (tabfm, backend, classifier):
        require(Path(module.__file__).resolve().is_relative_to(root), "Wrong TabFM import source")
    require(tabfm.__version__ == "1.0.1", "Unexpected TabFM wrapper version")
    # Exactly the official default loader/construction; never .ensemble().
    model = backend.load(model_type=task, checkpoint_path=str(directory), device="cuda:0")
    require(not model.training, "Native loader did not return an eval-mode model")
    cls = tabfm.TabFMClassifier if task == "classification" else tabfm.TabFMRegressor
    estimator = cls(model=model)
    defaults = {}
    for name, parameter in inspect.signature(cls.__init__).parameters.items():
        if name in ("self", "model"):
            continue
        require(parameter.default is not inspect.Parameter.empty, f"Unexpected required native argument: {name}")
        require(hasattr(estimator, name), f"Native default has no observable attribute: {name}")
        declared, actual = json_value(parameter.default), json_value(getattr(estimator, name))
        require(actual == declared, f"Native default constructor changed attribute: {name}")
        defaults[name] = actual
    return estimator, {"constructor": cls.__name__ + "(model=model)", "overrides": {},
        "parameters": defaults, "loader_explicit_kwargs": {"model_type": task,
            "checkpoint_path": str(directory), "device": "cuda:0"},
        "loader_dtype_override": None, "model_parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "use_amp_semantics": "native wrapper flag is informational; PyTorch loader defaults BF16 with internal FP32 upcasts",
        "native_class_limit": int(model.max_classes), "version": tabfm.__version__}


def native_predict(estimator, task, tx, ys, vx, *, return_probabilities=False):
    """Observe native forwards/default ensemble; never change aggregation or caps."""
    import numpy as np
    require(not return_probabilities or task == "classification", "Only classification provides probabilities")
    calls, batches = [], []
    phase = ["fit"]
    def forward_hook(_module, args, kwargs, output):
        require(phase[0] == "predict", "Native default fit unexpectedly forwarded a model")
        require(len(args) >= 3, "Native model forward signature changed")
        X, labels, train_sizes = args[:3]
        count, total = int(X.shape[0]), int(X.shape[1])
        sizes = np.asarray(train_sizes.detach().cpu().numpy()).reshape(-1)
        require(len(sizes) == count and np.all(sizes == len(ys))
                and total - len(ys) == len(vx), "Native default changed support/query row coverage")
        require(tuple(labels.shape) == (count, total) and tuple(output.shape[:2]) == (count, total),
                "Native forward output row coverage changed")
        start = sum(c["member_count"] for c in calls)
        calls.append({"member_indices": list(range(start, start + count)), "member_count": count,
                      "support_rows": len(ys), "test_rows": len(vx), "features": int(X.shape[2]),
                      "output_shape": [int(v) for v in output.shape], "output_dtype": str(output.dtype)})
    handle = estimator.model.register_forward_hook(forward_hook, with_kwargs=True)
    original_batch = estimator._batch_forward
    had_instance = "_batch_forward" in estimator.__dict__
    old_instance = estimator.__dict__.get("_batch_forward")
    def capture_batch(*args, **kwargs):
        output = original_batch(*args, **kwargs)
        require(output.ndim == 3 and output.shape[1] == len(vx) and np.isfinite(output).all(),
                "Nonfinite/incomplete native ensemble predictions")
        batches.append({"members": int(output.shape[0]), "test_rows": int(output.shape[1]),
                        "output_shape": list(output.shape)})
        return output
    try:
        estimator.fit(tx, ys)  # Original-unit y for regression; native StandardScaler only.
        generator = estimator.ensemble_generator_
        configs = [{"normalization": str(norm), "configuration": json_value(config)}
                   for norm, values in generator.ensemble_configs_.items() for config in values]
        require(configs, "Native generated no members")
        configuration_hashes = [common.object_digest(config) for config in configs]
        estimator._batch_forward = capture_batch
        phase[0] = "predict"
        if task == "classification":
            probabilities = np.asarray(estimator.predict_proba(vx.copy(deep=True)))
            require(probabilities.shape == (len(vx), len(estimator.classes_))
                    and np.isfinite(probabilities).all() and (probabilities >= 0).all()
                    and np.allclose(probabilities.sum(axis=1), 1., atol=1e-5), "Invalid native classification probabilities")
            # Matches official predict: argmax -> native class decoder.
            prediction = estimator.y_encoder_.inverse_transform(probabilities.argmax(axis=1).reshape(-1, 1))
            prediction = np.asarray(prediction).reshape(-1).astype(estimator.classes_.dtype)
            prediction_audit = {"probability_sha256": array_digest(probabilities),
                "probability_shape": list(probabilities.shape), "classes": json_value(estimator.classes_)}
        else:
            prediction = np.asarray(estimator.predict(vx.copy(deep=True))).reshape(-1)
            prediction_audit = {"native_target_scaler": type(estimator.y_scaler_).__name__,
                "native_target_scaler_mean": json_value(estimator.y_scaler_.mean_),
                "native_target_scaler_scale": json_value(estimator.y_scaler_.scale_)}
        require(prediction.shape == (len(vx),) and np.isfinite(prediction).all(), "Invalid complete-test prediction")
        count = sum(c["member_count"] for c in calls)
        require(count == len(configs) == sum(b["members"] for b in batches), "Generated and observed native member counts differ")
        audit = {"configured_n_estimators": int(estimator.n_estimators), "actual_ensemble_count": count,
            "native_generated_configurations": len(configs), "unique_configuration_count": len(set(configuration_hashes)),
            "configuration_sha256": configuration_hashes, "native_member_forward_verified": True,
            "forced_estimator_count": False, "strict32_protocol_claimed": False,
            "full_test_split": True, "full_support_received_by_each_forward": True,
            "test_rows": len(vx), "support_rows": len(ys), "forward_calls": calls, "batch_predictions": batches,
            "norm_methods": json_value(generator.norm_methods_),
            "max_num_features": estimator.max_num_features, "max_num_rows": estimator.max_num_rows,
            "native_feature_subsampling_permitted": True, "native_duplicate_configurations_permitted": True,
            "external_query_chunking": False, "external_support_subsampling": False,
            "prediction_sha256": array_digest(prediction), **prediction_audit}
        return (probabilities if return_probabilities else prediction), audit
    finally:
        handle.remove()
        if had_instance:
            estimator._batch_forward = old_instance
        elif "_batch_forward" in estimator.__dict__:
            del estimator._batch_forward


def evaluate(args):
    started = time.monotonic()
    campaign = load_campaign(args.campaign)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    # This helper reads only Torch's already-mapped HIP runtime with RTLD_NOLOAD.
    # Loading a system HIP library here can mix incompatible ROCm versions.
    from pfn_mitra_one import gpu_identity
    gpu = gpu_identity(torch)
    torch.cuda.reset_peak_memory_stats(0)
    loader = load_classification if args.task_kind == "classification" else load_regression
    with threadpool_limits(limits=args.threads):
        manifest, row, tx, ys, vx, yt, data_audit, raw_files = loader(campaign, args.dataset_index, np, pd)
        directory, weight_config = weight_directory(campaign, args.task_kind)
        result = {"schema_version": 1, "protocol": PROTOCOL, "model_name": "tabfm_default",
            "task_kind": args.task_kind, "dataset_index": args.dataset_index, "dataset": row["dataset"],
            "suite": row["suite"], "manifest_id": campaign["manifest_id"],
            "data_manifest_id": manifest["manifest_id"], "input_fingerprint": row["input_fingerprint"],
            "official_source_commit": COMMIT, "model_sha256": campaign["weights"][args.task_kind]["weights"]["sha256"],
            "worker_source_sha256": sha_file(__file__), "data_audit": data_audit,
            "physical_gpu": gpu, "full_test_split": True, "native_defaults": True,
            "host": socket.gethostname(), "pid": os.getpid(), "threads": args.threads,
            "feature_dtypes": {"support": [str(t) for t in tx.dtypes], "test": [str(t) for t in vx.dtypes]}}
        estimator, default_audit = load_native(campaign, args.task_kind, directory)
        with torch.inference_mode():
            hierarchical = args.task_kind == "classification" and len(np.unique(ys)) > int(weight_config["max_classes"])
            if hierarchical:
                from tabfm_hierarchical import hierarchical_predict_proba
                probabilities, ensemble_audit = hierarchical_predict_proba(
                    estimator.model, tx, ys, vx,
                    classifier_factory=lambda model: type(estimator)(model=model),
                    node_predictor=lambda node, sx, sy, qx, info: native_predict(
                        node, "classification", sx, sy, qx, return_probabilities=True))
                require(probabilities.shape == (len(vx), len(np.unique(ys)))
                        and np.isfinite(probabilities).all() and (probabilities >= 0).all()
                        and np.allclose(probabilities.sum(axis=1), 1., atol=1e-5), "Invalid hierarchy probabilities")
                prediction = np.unique(ys)[probabilities.argmax(axis=1)]
                ensemble_audit.update(probability_sha256=array_digest(probabilities),
                                      prediction_sha256=array_digest(prediction))
                result.update(native_defaults=False, native_defaults_per_node=True,
                    hierarchy_added=True, protocol_extension="balanced contiguous encoded-class tree; native-default TabFM per node")
            else:
                prediction, ensemble_audit = native_predict(estimator, args.task_kind, tx, ys, vx)
                result["hierarchy_added"] = False
        torch.cuda.synchronize(0)
        if args.task_kind == "classification":
            metrics = {"accuracy": float(np.mean(prediction == yt))}
        else:
            metrics = {"rmse": float(np.sqrt(mean_squared_error(yt, prediction))),
                "mae": float(mean_absolute_error(yt, prediction)), "r2": float(r2_score(yt, prediction, force_finite=True))}
        require(all(math.isfinite(v) for v in metrics.values()), "Nonfinite full-test metrics")
        require(raw_snapshot(raw_files) == raw_files, "Raw inputs changed during inference")
        if args.task_kind == "classification":
            common.verify_file(row["cache"])
        for identity in (campaign["weights"][args.task_kind]["config"], campaign["weights"][args.task_kind]["weights"]):
            common.verify_file(identity)
        result.update(complete=True, status="complete", metrics=metrics, **metrics,
            defaults_audit=default_audit, ensemble_audit=ensemble_audit,
            actual_ensemble_count=ensemble_audit["actual_ensemble_count"], seconds=time.monotonic() - started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--task-kind", choices=("classification", "regression"), required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    require(args.threads == 4, "This campaign freezes four CPU threads per worker")
    require(not args.output.exists() and not args.output.is_symlink(), "Output exists; no overwrite permitted")
    try:
        result = evaluate(args)
    except Exception as exc:
        common.publish_new(args.output, {"complete": False, "status": "error", "task_kind": args.task_kind,
            "dataset_index": args.dataset_index, "model_name": "tabfm_default", "error": repr(exc),
            "traceback": traceback.format_exc(), "worker_source_sha256": sha_file(__file__)})
        raise
    common.publish_new(args.output, result)
    print(json.dumps({"status": result["status"], "dataset": result["dataset"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
