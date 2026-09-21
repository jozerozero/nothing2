#!/usr/bin/env python3
"""Original Mitra classification: exactly 32 contributing members per tree node.

Uses the frozen historical457 numerical cache and unchanged support-centroid
hierarchy for >10 classes. Seed0, BF16, support8192/query1024, no fine-tuning.
Native member logits, class slicing, softmax, and probability averaging are
preserved. A many-class problem has 32 members PER fitted hierarchy node.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import socket
import time

import pfn_mitra_one as historical

common = historical.common
require, digest_file = common.require, historical.sha256_file
MEMBERS = 32
DEFAULT_STAGE = Path("/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/mitra_all_classification_regression_20260823_v3")
HISTORICAL_WORKER_SHA256 = "070d4f6f24ac5183654191e5c870a1dfb2a2ec7de57e153f72a33c4dd3fe4500"
HIERARCHY_SHA256 = "12a957d697afb5975d8e826ceebfa420ac85bdfae500721bf81f9fb82060758b"
CLASS_REPO = "autogluon/mitra-classifier"
CLASS_REVISION = "c425e9fa0910a6be1c494321792e7ba2a1367b1a"
CLASS_SHA256 = "e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19"
RUNTIME_PINS = {
    "autogluon.tabular.models.mitra.sklearn_interface": "bf283c9249982b310945689e844a84b2065e382576d2d9d901969a6899300ee0",
    "autogluon.tabular.models.mitra._internal.core.trainer_finetune": "3dc3f65b7dcb6a4525a35c7557e686c05b752984e424dda7febb1d8421587e1b",
    "autogluon.tabular.models.mitra._internal.data.dataset_finetune": "018c72435349677a40a7a9f2f98fe1ab0ff856dc96e57d400017217fba55b123",
    "autogluon.tabular.models.mitra._internal.data.preprocessor": "cf1da95aba436864926b03da8fb89e46314eb5cb7f17287c2d88e8fd8b65346d",
}
PROTOCOL = {"experiment": "original_mitra_classification457_actual32_seed0_v1",
    "n_estimators": MEMBERS, "seed": 0, "fine_tune": False, "fine_tune_steps": 0,
    "max_epochs": 0, "precision": "bfloat16", "native_support_cap": 8192,
    "native_query_cap": 1024, "native_head_classes": 10,
    "support_initialization_validation": "identical full support; no test labels",
    "data": "unchanged frozen historical457 numeric cache; full official test",
    "hierarchy": "unchanged historical support-centroid PCA balanced hierarchy",
    "actual_member_scope": "32 successful contributing members per test row PER fitted hierarchy node",
    "query_oom_policy": "historical external chunk backoff1024->64; discard failed chunk attempt",
    "native_support_oom_downgrade": False, "equal_compute_claim": False,
    "historical_only_requested_change": "one native estimator ->32; seed0 retained"}


def array_digest(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}, sort_keys=True).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def rng_digest(rng):
    name, keys, position, cached, value = rng.get_state()
    return common.object_digest([name, keys.tolist(), int(position), int(cached), float(value)])


def load_plan(path, index):
    plan = json.loads(Path(path).read_text())
    require(plan.get("manifest_id") == common.object_digest({k: v for k, v in plan.items() if k != "manifest_id"}),
            "Classification plan content identity mismatch")
    rows = plan.get("rows", [])
    require(plan.get("membership_count") == len(rows) == 457
            and len({r["dataset"] for r in rows}) == 457
            and [r["dataset_index"] for r in rows] == list(range(457)), "Expected exact indexed457 classification plan")
    require(plan.get("n_estimators", MEMBERS) == MEMBERS and plan.get("seed", 0) == 0,
            "Classification plan must request32 estimators/seed0")
    require(0 <= index < len(rows), "dataset-index outside classification457")
    row = rows[index]
    require(row["input_fingerprint"] == common.object_digest(row["cache"]), "Cache input fingerprint mismatch")
    record = plan.get("classification_benchmark_manifest", plan.get("class_benchmark_manifest"))
    require(record is not None, "Historical classification benchmark manifest missing")
    benchmark = json.loads(common.verify_file(record).read_text())
    names = [r["dataset"] for r in benchmark["rows"]]
    require(benchmark.get("complete") is True and len(names) == len(set(names)) == 457
            and set(names) == {r["dataset"] for r in rows}, "Classification membership differs from historical457")
    old = next(r for r in benchmark["rows"] if r["dataset"] == row["dataset"])
    require(names[index] == row["dataset"], "Classification dataset index differs from historical ordering")
    if "cache_path" in old:
        require(Path(row["cache"]["path"]).resolve() == Path(old["cache_path"]).resolve(),
                "Selected cache path differs from the historical classification membership")
    if "suite" in old:
        require(old["suite"] == row["suite"], "Classification suite changed")
    common.verify_file(plan["weights_manifest"])
    common.verify_file(row["cache"])
    return plan, row


def load_cache(row, np):
    path = common.verify_file(row["cache"])
    with np.load(path, allow_pickle=False) as archive:
        raw = {key: np.asarray(archive[key]) for key in ("X_train", "y_train", "X_test", "y_test")}
    require(all(value.dtype.kind in "biuf" for value in raw.values()), "Only numeric nonobject cache arrays permitted")
    require(raw["X_train"].ndim == raw["X_test"].ndim == 2
            and raw["X_train"].shape[1] == raw["X_test"].shape[1] > 0, "Cached feature schema invalid")
    require(all(raw[key].ndim == 1 or (raw[key].ndim == 2 and raw[key].shape[1] == 1)
                for key in ("y_train", "y_test")), "Cached labels must be vectors or singleton columns")
    require(all(np.isfinite(value).all() for value in raw.values()), "Nonfinite frozen cache")
    xs, ys = raw["X_train"].astype(np.float32), raw["y_train"].astype(np.int64).reshape(-1)
    xt, yt = raw["X_test"].astype(np.float32), raw["y_test"].astype(np.int64).reshape(-1)
    require(np.array_equal(raw["y_train"].reshape(-1), ys) and np.array_equal(raw["y_test"].reshape(-1), yt),
            "Nonintegral cached labels")
    require(len(xs) == len(ys) >= 2 and len(xt) == len(yt) >= 1, "Cached row counts invalid")
    classes = np.unique(ys)
    require(len(classes) >= 2 and np.array_equal(classes, np.arange(len(classes)))
            and set(np.unique(yt)) <= set(classes), "Invalid dense support/test labels")
    require(np.isfinite(xs).all() and np.isfinite(xt).all(), "Float32 conversion produced nonfinite features")
    for key, actual in (("train_rows", len(ys)), ("test_rows", len(yt)), ("features", xs.shape[1]), ("classes", len(classes))):
        if key in row:
            require(row[key] == actual, f"Frozen cache metadata mismatch: {key}")
    common.verify_file(row["cache"])
    return xs, ys, xt, yt, {"cache": row["cache"], "input_fingerprint": row["input_fingerprint"],
        "original_arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_digest(value)}
                            for key, value in raw.items()},
        "support_rows": len(ys), "test_rows": len(yt), "features": xs.shape[1], "classes": len(classes),
        "full_test_split": True, "test_rows_filtered": 0, "support_subsampling_in_loader": False,
        "test_labels_used_for_fit_or_routing": False, "numerical_conversion": "same historical float32 features/int64 labels"}


def check_cfg(cfg):
    expected = {"n_ensembles": MEMBERS, "max_epochs": 0, "max_samples_support": 8192,
                "max_samples_query": 1024, "precision": "bfloat16", "dim_output": 10,
                "grad_scaler_enabled": False, "shuffle_classes": False, "shuffle_features": False,
                "use_random_transforms": False, "random_mirror_x": True}
    require(cfg.seed == 0, "Historical classification seed0 changed")
    for name, value in expected.items():
        require(cfg.hyperparams.get(name) == value, f"Classifier32 native config changed: {name}")
    return expected


def guard_trainer(original):
    class GuardedClassifierTrainer(original):
        def __init__(self, cfg, model, *args, **kwargs):
            check_cfg(cfg)
            require(model.dim_output == 10 and not model.use_flash_attn, "Expected original10-class ROCm model")
            require(kwargs.get("rng") is not None, "Native shared ensemble RNG missing")
            self.audit_rng_entry = rng_digest(kwargs["rng"])
            super().__init__(cfg, model, *args, **kwargs)
            self.audit_optimizer_steps, self.audit_fit_calls = 0, 0
            def forbidden(*_args, **_kwargs):
                self.audit_optimizer_steps += 1
                raise RuntimeError("Optimizer update forbidden in classifier32 inference")
            self.optimizer.step = forbidden

        def train(self, xs, ys, xv, yv):
            import numpy as np
            check_cfg(self.cfg)
            require(xs is xv and ys is yv, "Classifier validation must be the identical supplied support")
            self.audit_fit_calls += 1
            require(self.audit_fit_calls == 1, "Unexpected classifier trainer refit")
            self.audit_support_rows = len(ys)
            self.audit_preprocess_rng = rng_digest(np.random)
            result = super().train(xs, ys, xv, yv)
            self.audit_feature_mirror = self.preprocessor.mirror.tolist()
            self.audit_rng_after_fit = rng_digest(self.rng)
            require(self.audit_optimizer_steps == 0, "Unexpected classifier optimizer update")
            return result
    return GuardedClassifierTrainer


def load_runtime(plan):
    stage = Path(plan.get("mitra_stage", DEFAULT_STAGE))
    require(importlib.metadata.version("autogluon.tabular") == "1.5.0", "Requires historical AutoGluon1.5.0")
    require(digest_file(stage / "mitra_common.py") == historical.HISTORICAL_ADAPTER_SHA256, "Historical Mitra helper changed")
    require(digest_file(stage / "hierarchical_mitra.py") == HIERARCHY_SHA256, "Historical class hierarchy changed")
    adapter = common.import_path("_class32_historical_mitra", stage / "mitra_common.py")
    hierarchy = common.import_path("_class32_historical_hierarchy", stage / "hierarchical_mitra.py")
    weight = adapter.validate_weights(common.verify_file(plan["weights_manifest"]), "classifier")
    require((weight["repo_id"], weight["revision"], weight["sha256"]) ==
            (CLASS_REPO, CLASS_REVISION, CLASS_SHA256), "Original classifier weight identity mismatch")
    require(Path(weight["offline_main_ref"]).read_text() == CLASS_REVISION,
            "Classifier offline main ref must equal pinned revision exactly, with no newline")
    from huggingface_hub import hf_hub_download
    for filename, expected in (("model.safetensors", weight["sha256"]), ("config.json", weight["config_sha256"])):
        path = hf_hub_download(repo_id=CLASS_REPO, filename=filename, revision="main", local_files_only=True)
        require(digest_file(path) == expected, f"Offline main differs from pinned classifier {filename}")
    modules, sources = {}, {}
    for name, expected in RUNTIME_PINS.items():
        module = importlib.import_module(name)
        require(digest_file(module.__file__) == expected, f"Native runtime source changed: {name}")
        modules[name], sources[name] = module, {"path": str(Path(module.__file__).resolve()), "sha256": expected}
    si = modules["autogluon.tabular.models.mitra.sklearn_interface"]
    original = modules["autogluon.tabular.models.mitra._internal.core.trainer_finetune"].TrainerFinetune
    require(si.TrainerFinetune is original, "Unexpected pre-existing classifier trainer monkeypatch")
    si.TrainerFinetune = guard_trainer(original)
    adapter.reset_seed(0)
    estimator = si.MitraClassifier(n_estimators=MEMBERS, fine_tune=False, fine_tune_steps=0,
        hf_model=CLASS_REPO, device="cuda", seed=0, verbose=False)
    return estimator, hierarchy, weight, sources


def fitted_member_records(estimator, np):
    require(estimator.n_estimators == MEMBERS and estimator.seed == 0
            and estimator.fine_tune is False and estimator.fine_tune_steps == 0, "Classifier32 constructor changed")
    trainers = estimator.trainers
    require(len(trainers) == len({id(t) for t in trainers}) == len({id(t.model) for t in trainers}) == MEMBERS,
            "Exactly32 distinct native classifier trainers/models required")
    require(len({id(t.rng) for t in trainers}) == 1, "Members must share native advancing RNG")
    require(len({t.audit_rng_entry for t in trainers}) == MEMBERS, "Member support RNG repeatedly reset")
    require(len({t.audit_preprocess_rng for t in trainers}) == MEMBERS, "Member preprocessing RNG repeatedly reset")
    records = []
    for index, trainer in enumerate(trainers):
        require(trainer.audit_fit_calls == 1 and trainer.audit_support_rows == len(estimator.y)
                and trainer.audit_optimizer_steps == 0, "Classifier member fit/no-update contract changed")
        records.append({"member_index": index, "config": check_cfg(trainer.cfg),
            "native_rng_entry_sha256": trainer.audit_rng_entry, "native_rng_after_fit_sha256": trainer.audit_rng_after_fit,
            "preprocessor_rng_sha256": trainer.audit_preprocess_rng, "feature_mirror": trainer.audit_feature_mirror,
            "optimizer_steps": 0})
    return records


def audited_probability_chunk(estimator, query, np):
    """Observe, never replace, each native class-logit -> softmax -> mean step."""
    rows, local_classes = len(query), len(np.unique(estimator.y))
    require(0 < rows <= 1024 and 2 <= local_classes <= 10, "Invalid native classification chunk/classes")
    trainers = estimator.trainers
    require(len(trainers) == MEMBERS, "Missing classifier ensemble members")
    xs_hash, ys_hash, query_hash = map(array_digest, (estimator.X, estimator.y, query))
    probabilities, calls, forwards = [None] * MEMBERS, [0] * MEMBERS, [0] * MEMBERS
    records, originals, hooks = [], [], []
    for member, trainer in enumerate(trainers):
        record = {"member_index": member, "native_rng_before_predict_sha256": None}
        records.append(record)
        original = trainer.predict
        originals.append((trainer, original))
        def capture(xs, ys, xt, *, member=member, trainer=trainer, original=original, record=record):
            require((array_digest(xs), array_digest(ys), array_digest(xt)) == (xs_hash, ys_hash, query_hash),
                    "Native classifier member received different inputs")
            calls[member] += 1
            require(calls[member] == 1, "Duplicate classifier member prediction")
            record["native_rng_before_predict_sha256"] = rng_digest(trainer.rng)
            raw = original(xs, ys, xt)
            logits = np.asarray(raw)
            require(logits.shape == (rows, 10) and np.isfinite(logits).all(), "Invalid native10-class logits")
            selected = logits[:, :local_classes]
            probabilities[member] = np.exp(selected) / np.exp(selected).sum(axis=1, keepdims=True)
            record.update(logits_shape=list(logits.shape), logits_sha256=array_digest(logits),
                          optimizer_steps=trainer.audit_optimizer_steps)
            require(record["optimizer_steps"] == 0, "Classifier optimizer updated during prediction")
            return raw
        def forward_check(module, inputs, output, *, member=member, record=record):
            require(len(inputs) == 6 and module.dim_output == 10, "Unexpected classifier forward signature/head")
            xs, ys, xt, _features, support_padding, query_padding = inputs
            require(tuple(xs.shape[:2]) == (1, min(8192, len(estimator.y)))
                    and tuple(xt.shape[:2]) == (1, rows) and tuple(output.shape) == (1, rows, 10),
                    "Native classifier context/query/output shape changed")
            require(not bool(support_padding.any()) and not bool(query_padding.any()), "Unexpected padded classifier rows")
            forwards[member] += 1
            require(forwards[member] == 1, "Hidden duplicate classifier forward")
            record.update(model_output_shape=list(output.shape), actual_support_rows=int(xs.shape[1]),
                          actual_query_rows=rows)
        trainer.predict = capture
        hooks.append(trainer.model.register_forward_hook(forward_check))
    try:
        probability = estimator.predict_proba(query)
    finally:
        for hook in hooks:
            hook.remove()
        for trainer, original in originals:
            trainer.predict = original
    require(calls == forwards == [1] * MEMBERS and all(p is not None for p in probabilities),
            "Every class query row requires32 independently executed native members")
    expected = sum(probabilities) / MEMBERS
    require(np.asarray(probability).shape == (rows, local_classes) and np.isfinite(probability).all()
            and np.array_equal(probability, expected), "Native softmax/member arithmetic mean changed")
    for trainer in trainers:
        check_cfg(trainer.cfg)
    return np.asarray(probability, dtype=np.float64), records


class AuditedClassifier:
    def __init__(self, estimator, np, torch):
        self.native, self.np, self.torch = estimator, np, torch
        self.ensemble_audits = []
        self.current = None

    def fit(self, xs, ys):
        classes = self.np.unique(ys)
        require(2 <= len(classes) <= 10 and self.np.array_equal(classes, self.np.arange(len(classes))),
                "Hierarchy node labels must be dense2..10 classes")
        # Historical hierarchy refits the same estimator per node. Release all
        # preceding member/model cycles before loading32 fresh frozen copies.
        self.native.trainers.clear()
        gc.collect()
        self.torch.cuda.empty_cache()
        self.native.fit(xs, ys, X_val=xs, y_val=ys)
        records = fitted_member_records(self.native, self.np)
        self.current = {"node_index": len(self.ensemble_audits), "support_rows": len(ys),
            "node_classes": len(classes), "node_support_sha256": array_digest(xs),
            "node_target_sha256": array_digest(ys), "actual_ensemble_count": MEMBERS,
            "member_audits": records, "successful_prediction_chunks": [], "discarded_oom_attempts": []}
        self.ensemble_audits.append(self.current)
        return self

    def predict_full(self, query):
        require(self.current is not None and "actual32_verified" not in self.current, "Each node must predict full test once")
        np, torch = self.np, self.torch
        chunk = max(64, min(1024, len(query)))
        position, outputs = 0, []
        coverage = np.zeros(len(query), dtype=np.uint8)
        while position < len(query):
            stop = min(position + chunk, len(query))
            try:
                probability, records = audited_probability_chunk(self.native, query[position:stop], np)
            except torch.OutOfMemoryError:
                self.current["discarded_oom_attempts"].append({"start": position, "stop": stop, "chunk": chunk})
                if chunk <= 64:
                    raise
                chunk = max(64, chunk // 2)
                torch.cuda.empty_cache()
                continue
            require(probability.shape == (stop - position, self.current["node_classes"]), "Wrong node probability shape")
            outputs.append(probability)
            coverage[position:stop] += MEMBERS
            self.current["successful_prediction_chunks"].append({"start": position, "stop": stop,
                "actual_ensemble_count": MEMBERS, "member_audits": records})
            position = stop
        probability = np.concatenate(outputs, axis=0)
        require(np.isfinite(probability).all() and (probability >= 0).all(), "Invalid native probabilities")
        total = probability.sum(axis=1, keepdims=True)
        require((total > 0).all() and bool((coverage == MEMBERS).all()), "Incomplete32-member/full-test coverage")
        self.current.update(actual32_verified=True, all_test_rows_covered=True, test_rows=len(query),
            minimum_contributions_per_test_row=int(coverage.min()), maximum_contributions_per_test_row=int(coverage.max()),
            coverage_sha256=array_digest(coverage), minimum_test_chunk_used=chunk,
            total_successful_member_forwards=MEMBERS * len(self.current["successful_prediction_chunks"]))
        return probability / total, chunk


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    require(1 <= args.threads <= 64, "Invalid CPU threads")
    require(not args.output.exists() and not args.output.is_symlink(), "Refusing to overwrite any result")
    require(digest_file(historical.__file__) == HISTORICAL_WORKER_SHA256, "Frozen HIP/shared helper changed")
    require(digest_file(common.__file__) == historical.RAW_LOADER_SHA256, "Frozen publication helper changed")
    plan, row = load_plan(args.plan, args.dataset_index)
    stage = Path(plan.get("mitra_stage", DEFAULT_STAGE))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ.update(HF_HOME=str(stage / "hf_cache"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    started = time.monotonic()
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    gpu = historical.gpu_identity(torch)
    torch.cuda.reset_peak_memory_stats(0)
    with threadpool_limits(limits=args.threads):
        xs, ys, xt, yt, data_audit = load_cache(row, np)
        native, hierarchy_helper, weight, sources = load_runtime(plan)
        estimator = AuditedClassifier(native, np, torch)
        classes = len(np.unique(ys))
        if classes > 10:
            probability, hierarchy = hierarchy_helper.hierarchical_predict_proba(
                estimator, xs, ys, xt, branch_factor=10, predict_fn=lambda fitted, query: fitted.predict_full(query))
            require(hierarchy["nodes_fitted"] == len(estimator.ensemble_audits), "Hierarchy node audit incomplete")
        else:
            estimator.fit(xs, ys)
            probability, _chunk = estimator.predict_full(xt)
            hierarchy = None
        require(probability.shape == (len(yt), classes) and np.isfinite(probability).all(), "Invalid final classification probabilities")
        require(all(a["actual32_verified"] and a["test_rows"] == len(yt) for a in estimator.ensemble_audits),
                "Not every hierarchy node verified32 full-test members")
        accuracy = float(np.mean(probability.argmax(axis=1) == yt))
        require(math.isfinite(accuracy), "Nonfinite accuracy")
        common.verify_file(row["cache"])
        torch.cuda.synchronize(0)
        result = {"schema_version": 1, "complete": True, "protocol_validation": True,
            "model_name": "mitra", "task_kind": "classification", "task": "classification",
            "dataset_index": args.dataset_index, "dataset": row["dataset"], "suite": row["suite"],
            "manifest_id": plan["manifest_id"], "input_fingerprint": row["input_fingerprint"],
            "accuracy": accuracy, "metrics": {"accuracy": accuracy}, "model_repo": weight["repo_id"],
            "model_revision": weight["revision"], "model_sha256": weight["sha256"],
            "n_estimators": MEMBERS, "actual_ensemble_count": MEMBERS, "actual32_verified": True,
            "seed": 0, "fine_tune": False, "fine_tune_steps": 0, "protocol": PROTOCOL,
            "protocol_fingerprint": common.object_digest(PROTOCOL), "data_audit": data_audit,
            "full_test_split": True,
            "support_rows": len(ys), "test_rows": len(yt), "features": xs.shape[1], "classes": classes,
            "hierarchical": classes > 10, "hierarchical_protocol": hierarchy,
            "node_count": len(estimator.ensemble_audits), "ensemble_audits": estimator.ensemble_audits,
            "probability_sha256": array_digest(probability), "runtime_sources": sources,
            "hierarchy_source_sha256": HIERARCHY_SHA256, "historical_worker_sha256": HISTORICAL_WORKER_SHA256,
            "worker_source_sha256": digest_file(__file__), "gpu": gpu, "node": socket.gethostname(), "pid": os.getpid(),
            "parent_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "elapsed_seconds": time.monotonic() - started, "cpu_threads": torch.get_num_threads(),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0), "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__), "autogluon.tabular": "1.5.0"}}
        common.publish_new(args.output, result)
    print(json.dumps({"complete": True, "model_name": "mitra", "task_kind": "classification",
        "dataset_index": args.dataset_index, "accuracy": accuracy, "actual_ensemble_count": MEMBERS,
        "node_count": result["node_count"], "output": str(args.output)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
