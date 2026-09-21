#!/usr/bin/env python3
"""One frozen standard681 split with the official TabSwift recipe or audited budget.

The official16 recipe is NOT a reproduction of the paper's benchmark/seeds: it
uses our unchanged canonical support/test rows and seed42. Official TALENT
preprocessing is fitted on support only. No external row/feature cap, training,
test-label preprocessing, hierarchy replacement, or checkpoint conversion.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import random
import resource
import socket
import sys
import time
import traceback

import eval_one as common
import tabfm_default_one as data_helper

require = common.require
COMMIT = "8edf8f0b4225bc03e1f5db011912619cd92b798d"
WEIGHTS_SHA256 = "16e324177be2ab9e2bac15e5edf7867329e6595e134a5c4c9b6d79d3b657b363"
WEIGHTS_REVISION = "b829456edb7c41ad93a2851a8df245db362e1c83"
DEPENDENCIES = ("eval_one.py", "tabfm_default_one.py", "pfn_mitra_one.py", "tabswift_ensemble.py")


def protocol_settings(campaign, task):
    require(task in ("classification", "regression"), "Invalid task kind")
    protocol = campaign["protocol"]
    variant = protocol["variant"]
    require(variant in ("official16", "budget32x8"), "Unknown TabSwift protocol")
    counts = {"classification": 16, "regression": 16} if variant == "official16" else {
        "classification": 32, "regression": 8}
    require(protocol["n_estimators"] == counts, "Protocol estimator budgets changed")
    require(protocol["strict_actual_count"] is (variant == "budget32x8"), "Protocol strict-count flag mismatch")
    require(protocol["batch_size"] == 16 and protocol["random_state"] == 42,
            "This recipe freezes batch_size16 and random_state42")
    return {"n_estimators": counts[task], "norm_methods": ["none", "power"],
        "feat_shuffle_method": "latin", "class_shift": task == "classification",
        "outlier_threshold": 4.0, "softmax_temperature": 0.9, "average_logits": True,
        "use_hierarchical": True, "batch_size": 16, "use_amp": True,
        "allow_auto_download": False, "device": "cuda:0", "random_state": 42, "verbose": False}


def load_campaign(path):
    man = json.loads(Path(path).read_text())
    require(man.get("manifest_id") == common.object_digest({k: v for k, v in man.items() if k != "manifest_id"}),
            "Campaign digest mismatch")
    for task in ("classification", "regression"):
        protocol_settings(man, task)
    source = man["official_source"]
    require(source["commit"] == COMMIT, "Wrong official TabSwift revision")
    root = Path(source["path"]).resolve(strict=True)
    records = source["files"]
    require(records and len({r["path"] for r in records}) == len(records), "Missing/duplicate source identities")
    for rec in records:
        require(common.verify_file(rec, allow_empty=True).is_relative_to(root), "Source identity outside official tree")
    workers = man["worker_sources"]
    require(workers and len({r["path"] for r in workers}) == len(workers), "Missing/duplicate worker identities")
    verified = {common.verify_file(r, allow_empty=True) for r in workers}
    here = Path(__file__).resolve()
    require(here in verified, "Worker must be hash-pinned")
    for name in DEPENDENCIES:
        require(here.with_name(name) in verified, f"Unpinned worker dependency: {name}")
    require(Path(common.__file__).resolve() == here.with_name("eval_one.py")
            and Path(data_helper.__file__).resolve() == here.with_name("tabfm_default_one.py"), "Wrong data helper imports")
    require(man["weights"]["shared"]["sha256"] == WEIGHTS_SHA256, "Wrong shared swift.ckpt")
    common.verify_file(man["weights"]["shared"])
    data_helper.verify_manifest(man["classification_manifest"])
    data_helper.verify_manifest(man["regression_manifest"])
    return man


def checked_output(campaign, output):
    root = Path(campaign["output_root"]).resolve()
    path = Path(output).resolve()
    require(path.is_relative_to(root) and path != root, "Output must be inside this isolated campaign")
    require(not Path(output).exists() and not Path(output).is_symlink(), "Output exists; immutable publication only")
    return path


def check_official_imports(campaign):
    root = Path(campaign["official_source"]["path"]).resolve(strict=True)
    pinned = {Path(r["path"]).resolve() for r in campaign["official_source"]["files"]}
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name == "tabswift" or name.startswith("tabswift.") or name == "TALENT" or name.startswith("TALENT."):
            filename = getattr(module, "__file__", None)
            if filename is None:  # Namespace packages have no executable source.
                continue
            path = Path(filename).resolve()
            require(path.is_relative_to(root) and path in pinned, f"Unpinned/wrong official import: {name}: {path}")
            result[name] = str(path)
    return result


def import_native(campaign, task):
    root = Path(campaign["official_source"]["path"]).resolve(strict=True)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "TALENT/model/lib"))
    processing = importlib.import_module("TALENT.model.lib.data")
    module = importlib.import_module("tabswift.classifier" if task == "classification" else "tabswift.regressor")
    check_official_imports(campaign)
    cls = module.TabSwiftClassifier if task == "classification" else module.TabSwiftRegressor
    settings = protocol_settings(campaign, task)
    settings["model_path"] = str(common.verify_file(campaign["weights"]["shared"]))
    estimator = cls(**settings)
    # These native defaults, including PCA100, are deliberately not overridden.
    require(estimator.enable_dim_reduction is True and estimator.pca_dim == 100 and estimator.rp_dim == 32768,
            "Unexpected native dimension-reduction defaults")
    return estimator, processing, settings


def official_preprocess(tx, ys, vx, task, processing):
    """Actual pinned TALENT functions; no test labels accepted by this API."""
    import numpy as np
    import pandas as pd
    require(len(tx) == len(ys) and len(tx) > 0 and len(vx) > 0 and list(tx.columns) == list(vx.columns),
            "Support/test schema or row count mismatch")
    numerical = [name for name in tx if pd.api.types.is_numeric_dtype(tx[name].dtype)]
    categorical = [name for name in tx if name not in numerical]
    num = {"train": tx[numerical].to_numpy(dtype=float), "test": vx[numerical].to_numpy(dtype=float)} if numerical else None
    cat = {"train": tx[categorical].to_numpy(), "test": vx[categorical].to_numpy()} if categorical else None
    # Both keys allow test-only missingness; the official function computes every
    # imputation statistic from N['train'], never test values.
    num, cat, means, _, cat_value = processing.data_nan_process(num, cat, "mean", "new")
    num_train = {"train": num["train"]} if num is not None else None
    num_test = {"test": num["test"]} if num is not None else None
    # A read-only preprocessing view initializes the official mode_values branch.
    # It is NOT a new split and never appends any support rows to model input.
    cat_train = {"train": cat["train"], "val": cat["train"]} if cat is not None else None
    cat_test = {"test": cat["test"]} if cat is not None else None
    num_train, cat_train, encoder, modes, cat_encoder = processing.data_enc_process(num_train, cat_train, "indices")
    num_test, cat_test, _, _, _ = processing.data_enc_process(
        num_test, cat_test, "indices", ord_encoder=encoder, mode_values=modes, cat_encoder=cat_encoder)
    def combine(n, c, key):
        blocks = [part[key] for part in (n, c) if part is not None]
        require(bool(blocks), "No canonical features remain")
        return np.concatenate(blocks, axis=1) if len(blocks) == 2 else blocks[0]
    train, test = combine(num_train, cat_train, "train"), combine(num_test, cat_test, "test")
    require(train.shape == (len(tx), tx.shape[1]) and test.shape == (len(vx), tx.shape[1])
            and np.isfinite(train).all() and np.isfinite(test).all(), "Official preprocessing produced invalid features")
    labels, info, label_encoder = processing.data_label_process({"train": np.asarray(ys)}, task == "regression")
    fit_y = labels["train"]
    require(np.isfinite(fit_y).all() and len(fit_y) == len(ys), "Official label transform is nonfinite (including constant target)")
    if task == "regression":
        require(info["policy"] == "mean_std" and math.isfinite(float(info["std"])) and info["std"] > 0,
                "Official regression standard deviation must be positive")
    audit = {"implementation": "pinned TALENT.model.lib.data functions", "fit_scope": "canonical support only",
        "numerical_columns": [str(n) for n in numerical], "categorical_columns": [str(n) for n in categorical],
        "column_order": "numerical then categorical, original order within each block",
        "numerical_nan_policy": "mean", "categorical_nan_policy": "new", "categorical_policy": "indices",
        "normalization": "none", "num_policy": "none", "imputation_values": data_helper.json_value(means),
        "categorical_missing_value": cat_value, "official_unknown_replacement_values": data_helper.json_value(modes),
        "unknown_replacement_note": "Unmodified official mode_values implementation, including its column[0] branch",
        "support_as_preprocessing_val_view": True, "additional_support_rows": 0,
        "test_labels_used": False, "support_rows": len(train), "test_rows": len(test), "features": train.shape[1],
        "support_features_sha256": data_helper.array_digest(train), "test_features_sha256": data_helper.array_digest(test),
        "fit_targets_sha256": data_helper.array_digest(fit_y), "label_info": data_helper.json_value(info),
        "regression_inverse": "prediction * support_std + support_mean" if task == "regression" else None}
    return train, fit_y, test, info, label_encoder, audit


@contextmanager
def checkpoint_load_guard(torch, identity):
    """Enforce the native safe local load without preloading or changing tensors."""
    original = torch.load
    path = common.verify_file(identity)
    audit = {"loads": 0, "weights_only": True, "map_location": "cpu", "sha256": identity["sha256"]}
    def guarded(filename, *args, **kwargs):
        require(not args and Path(filename).resolve() == path and kwargs.get("weights_only") is True
                and kwargs.get("map_location") == "cpu", "Unexpected/unsafe checkpoint load")
        require(audit["loads"] == 0, "Checkpoint loaded more than once")
        value = original(filename, **kwargs)
        require(isinstance(value, dict) and isinstance(value.get("config"), dict)
                and isinstance(value.get("state_dict"), dict), "Invalid official checkpoint structure")
        audit.update(loads=1, config=data_helper.json_value(value["config"]),
            config_sha256=common.object_digest(data_helper.json_value(value["config"])),
            state_dict_keys_sha256=common.object_digest(sorted(value["state_dict"])),
            state_dict_key_count=len(value["state_dict"]))
        return value
    torch.load = guarded
    try:
        yield audit
    finally:
        torch.load = original
        common.verify_file(identity)


def predict_native(estimator, train, ys, test, task, strict_actual_count, configure):
    import numpy as np
    handle = configure(estimator, task, estimator.n_estimators, strict_actual_count=strict_actual_count)
    try:
        estimator.fit(train, ys)
        require(not estimator.model_.training, "Native model must remain in evaluation mode")
        classes = np.unique(ys) if task == "classification" else None
        if classes is not None:
            limit = int(estimator.model_.max_classes)
            require(limit > 1 and len(classes) <= limit * limit,
                f"Native TabSwift hierarchy cannot represent {len(classes)} classes safely: root ceil(C/{limit}) exceeds head{limit}; no algorithm substitution")
            proba = np.asarray(estimator.predict_proba(test)).astype(np.float32)
            require(proba.shape == (len(test), len(classes)) and np.isfinite(proba).all()
                    and (proba >= 0).all() and np.allclose(proba.sum(axis=1), 1., atol=1e-5),
                    "Invalid full-test native probabilities")
            # Native predict uses its own fitted LabelEncoder, not an assumed column order.
            prediction = estimator.y_encoder_.inverse_transform(proba.argmax(axis=1))
        else:
            proba = None
            prediction = np.asarray(estimator.predict(test)).astype(np.float32)
            require(prediction.shape in ((len(test),), (len(test), 1)), "Wrong full-test regression output shape")
            prediction = prediction.reshape(-1)
        require(prediction.shape == (len(test),) and np.isfinite(prediction).all(), "Incomplete/nonfinite native prediction")
        audit = handle.finish(len(test))
        require(isinstance(audit.get("actual_ensemble_count"), int) and audit["actual_ensemble_count"] > 0,
                "Missing actual native ensemble count")
        require(audit.get("full_test_split") is True, "Missing full-query ensemble evidence")
        if strict_actual_count:
            require(audit["actual_ensemble_count"] == estimator.n_estimators, "Strict actual ensemble budget not met")
        audit.update(native_prediction_sha256=data_helper.array_digest(prediction),
            probability_sha256=data_helper.array_digest(proba) if proba is not None else None,
            official_method_output_cast="float32", native_hierarchy=bool(classes is not None and len(classes) > estimator.model_.max_classes),
            external_hierarchy_added=False, native_pca_dim=estimator.pca_dim)
        return prediction, audit
    finally:
        handle.close()


def evaluate(args, campaign):
    started = time.monotonic()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    require(os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED=0 must be set before launching Python")
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from threadpoolctl import threadpool_limits
    from pfn_mitra_one import gpu_identity
    from tabswift_ensemble import configure
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    gpu = gpu_identity(torch)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats(0)
    loader = data_helper.load_classification if args.task_kind == "classification" else data_helper.load_regression
    with threadpool_limits(limits=args.threads):
        manifest, row, tx, ys, vx, yt, data_audit, raw_files = loader(campaign, args.dataset_index, np, pd)
        require(data_audit["full_test_split"] is True and len(tx) == len(ys) and len(vx) == len(yt), "Frozen split mismatch")
        estimator, processing, settings = import_native(campaign, args.task_kind)
        train, fit_y, test, label_info, label_encoder, preprocessing = official_preprocess(tx, ys, vx, args.task_kind, processing)
        with torch.inference_mode(), checkpoint_load_guard(torch, campaign["weights"]["shared"]) as load_audit:
            prediction, ensemble = predict_native(estimator, train, fit_y, test, args.task_kind,
                campaign["protocol"]["strict_actual_count"], configure)
        require(load_audit["loads"] == 1, "Expected one safe native checkpoint load")
        if args.task_kind == "classification":
            prediction = label_encoder.inverse_transform(prediction)
            metrics = {"accuracy": float(np.mean(prediction == yt))}
        else:
            prediction = prediction.astype(np.float64) * label_info["std"] + label_info["mean"]
            require(np.isfinite(prediction).all(), "Nonfinite original-unit prediction")
            metrics = {"rmse": float(np.sqrt(mean_squared_error(yt, prediction))),
                "mae": float(mean_absolute_error(yt, prediction)), "r2": float(r2_score(yt, prediction, force_finite=True))}
        torch.cuda.synchronize(0)
        require(all(math.isfinite(v) for v in metrics.values()), "Nonfinite full-test metrics")
        require(data_helper.raw_snapshot(raw_files) == raw_files, "Raw inputs changed during inference")
        if args.task_kind == "classification":
            common.verify_file(row["cache"])
        data_audit.update(data_route="frozen official raw split -> support-only canonicalization -> official TALENT preprocessing -> native TabSwift",
            external_target_transform="official support-only mean/std" if args.task_kind == "regression" else "official LabelEncoder",
            source_taffy_target_transform_applied=False, preprocessing=preprocessing,
            original_unit_prediction_sha256=data_helper.array_digest(prediction), full_test_split=True)
        imports = check_official_imports(campaign)
        # Recheck executable and checkpoint identities before immutable publication.
        for rec in campaign["official_source"]["files"] + campaign["worker_sources"]:
            common.verify_file(rec, allow_empty=True)
        common.verify_file(campaign["weights"]["shared"])
        variant = campaign["protocol"]["variant"]
        return {"complete": True, "status": "complete", "schema_version": 1,
            "model_name": "tabswift_" + variant, "protocol": campaign["protocol"],
            "protocol_description": "official model recipe on frozen standard681 support/test splits with seed42; not paper benchmark reproduction",
            "task_kind": args.task_kind, "dataset_index": args.dataset_index, "dataset": row["dataset"], "suite": row["suite"],
            "manifest_id": campaign["manifest_id"], "data_manifest_id": manifest["manifest_id"],
            "input_fingerprint": row["input_fingerprint"], "worker_source_sha256": data_helper.sha_file(__file__),
            "official_source_commit": COMMIT, "model_sha256": WEIGHTS_SHA256, "weights_revision": WEIGHTS_REVISION,
            "full_test_split": True, "data_audit": data_audit, "physical_gpu": gpu,
            "metrics": metrics, **metrics, "actual_ensemble_count": ensemble["actual_ensemble_count"],
            "requested_ensemble_count": settings["n_estimators"], "ensemble_audit": ensemble,
            "constructor_settings": settings, "checkpoint_load_audit": load_audit, "official_imports": imports,
            "feature_dtypes": {"support": [str(t) for t in tx.dtypes], "test": [str(t) for t in vx.dtypes]},
            "precision": {"native_use_amp": True, "parameter_dtypes": sorted({str(p.dtype) for p in estimator.model_.parameters()}),
                "note": "Native autocast policy; not a uniform FP32 claim"},
            "host": socket.gethostname(), "pid": os.getpid(), "threads": args.threads,
            "seconds": time.monotonic() - started, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--task-kind", choices=("classification", "regression"), required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    require(args.threads == 4, "Campaign freezes four CPU threads per worker")
    campaign = load_campaign(args.campaign)
    output = checked_output(campaign, args.output)
    try:
        result = evaluate(args, campaign)
    except Exception as exc:
        common.publish_new(output, {"complete": False, "status": "error", "manifest_id": campaign["manifest_id"],
            "task_kind": args.task_kind, "dataset_index": args.dataset_index, "model_name": "tabswift_" + campaign["protocol"]["variant"],
            "error": repr(exc), "traceback": traceback.format_exc(), "worker_source_sha256": data_helper.sha_file(__file__)})
        raise
    common.publish_new(output, result)
    print(json.dumps({"status": "complete", "dataset": result["dataset"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
