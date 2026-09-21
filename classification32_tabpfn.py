"""Pinned TabPFN classification adapter with 32 observed contributors per node.

Public API: ``predict(model_key, arrays, model_config) -> (probabilities, audit)``.
``arrays`` supplies X_train, y_train, X_test; test labels are not consumed.
``model_config`` supplies source_root, checkpoint_path, checkpoint_sha256,
source_commit, hierarchy_helper_path and hierarchy_helper_sha256. Legacy aliases
model_path/model_sha256/hierarchy_path are accepted. test_chunk, when supplied,
must retain the historical4096.

All numerical prediction, class permutation, temperature scaling, softmax,
averaging, and hierarchy remain in the pinned native implementation. Scoped
observers prove that32 actual member outputs reach the native32-way reduction.
The old8/default workers, checkpoints, model sources, and data are never edited.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


MEMBERS = 32
SOURCE_COMMIT = "38f574987deb5f313a832e2b42108ee3ef190e85"
MODEL_VERSIONS = {"tabpfn2": ("V2", 10), "tabpfn25": ("V2_5", 10), "tabpfn3": ("V3", 160)}
SOURCE_PINS = {
    "tabpfn.classifier": "115002989cebc78e2d964877d0cf9f8083f5449157b4e7d4367c1f3f15397467",
    "tabpfn.inference": "04aba891f8c23b29b4b6fcf726c9f187bb6fa9cf5ed8503bdbcd2f3afb199f7e",
    "tabpfn.preprocessing.ensemble": "973ad1808c0e8f185d4c4faa2969ce54275418e7c23038db32fb661065f7b9a2",
}
HIERARCHY_SHA256 = "83ef503a4ecdc1dcb1d0f55f41e028c819494ee3fc4880f0e13cd3fab008e85c"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    raise RuntimeError(f"Unsupported native ensemble configuration value: {type(value)}")


def as_array(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def array_digest(value):
    import numpy as np
    array = np.ascontiguousarray(as_array(value))
    digest = hashlib.sha256(json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def validate_inputs(arrays):
    import numpy as np
    # No access to y_test, including when it is present in a runner's mapping.
    xs, ys, xt = (np.asarray(arrays[key]) for key in ("X_train", "y_train", "X_test"))
    require(xs.ndim == xt.ndim == 2 and ys.ndim == 1 and len(xs) == len(ys), "Invalid cached array dimensions")
    require(xs.shape[1] == xt.shape[1] > 0 and len(xt) > 0 and len(xs) >= 2, "Invalid support/query schema")
    require(xs.dtype == xt.dtype == np.dtype("float32") and ys.dtype == np.dtype("int64"),
            "Runner must supply historical float32 features/int64 labels without further conversion")
    require(np.isfinite(xs).all() and np.isfinite(xt).all(), "Nonfinite cached features")
    classes = np.unique(ys)
    require(len(classes) >= 2 and np.array_equal(classes, np.arange(len(classes))), "Support labels must be dense0..K-1")
    return xs, ys, xt


def normalize_config(model_config):
    config = dict(model_config)
    for primary, alias in (("checkpoint_path", "model_path"),
                           ("checkpoint_sha256", "model_sha256"),
                           ("hierarchy_helper_path", "hierarchy_path")):
        if primary in config and alias in config:
            require(str(config[primary]) == str(config[alias]), f"Conflicting model configuration aliases: {primary}")
        require(primary in config or alias in config, f"Missing model configuration: {primary}")
        config[alias] = config[primary] if primary in config else config[alias]
    require(config.get("hierarchy_helper_sha256", HIERARCHY_SHA256) == HIERARCHY_SHA256,
            "Unexpected requested hierarchy source pin")
    return config


def load_runtime(model_key, model_config):
    model_config = normalize_config(model_config)
    require(model_key in MODEL_VERSIONS, "Unknown TabPFN classification model")
    require(model_config["source_commit"] == SOURCE_COMMIT, "Wrong historical TabPFN source pin")
    require(model_config.get("test_chunk", 4096) == 4096, "Retain historical initial query chunk4096")
    source = Path(model_config["source_root"]).resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    require(head == SOURCE_COMMIT, "TabPFN source HEAD changed")
    checkpoint = Path(model_config["model_path"]).resolve(strict=True)
    before = checkpoint.stat()
    require(len(model_config["model_sha256"]) == 64 and file_digest(checkpoint) == model_config["model_sha256"],
            "Pinned TabPFN classification checkpoint changed")
    after = checkpoint.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), "Checkpoint changed while hashing")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source / "src"))
    modules, sources = {}, {}
    for name, expected in SOURCE_PINS.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(source), f"Unexpected imported TabPFN source: {name}")
        require(file_digest(path) == expected, f"Audited TabPFN implementation changed: {name}")
        modules[name], sources[name] = module, {"path": str(path), "sha256": expected}
    hierarchy_path = Path(model_config["hierarchy_path"]).resolve(strict=True)
    require(file_digest(hierarchy_path) == HIERARCHY_SHA256, "Historical TabPFN hierarchy changed")
    spec = importlib.util.spec_from_file_location("_classification32_pinned_tabpfn_hierarchy", hierarchy_path)
    require(spec is not None and spec.loader is not None, "Cannot load historical TabPFN hierarchy")
    hierarchy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hierarchy)
    import torch
    from tabpfn.constants import ModelVersion
    version_name, class_limit = MODEL_VERSIONS[model_key]
    estimator = modules["tabpfn.classifier"].TabPFNClassifier.create_default_for_version(
        getattr(ModelVersion, version_name), model_path=checkpoint, n_estimators=MEMBERS,
        device="cuda:0", ignore_pretraining_limits=True, inference_precision=torch.float32,
        fit_mode="fit_preprocessors", memory_saving_mode=True, random_state=0,
        n_preprocessing_jobs=1, show_progress_bar=False)
    require(estimator.get_inference_config().MAX_NUMBER_OF_CLASSES == class_limit, "Native class limit changed")
    sources["hierarchical_tabpfn"] = {"path": str(hierarchy_path), "sha256": HIERARCHY_SHA256}
    return estimator, hierarchy, torch, sources


class Native32Audit:
    """One unchanged classifier, refitted at each original hierarchy node."""
    def __init__(self, estimator, torch):
        self.estimator, self.torch = estimator, torch
        self.nodes = []
        self.current = None

    def fit(self, xs, ys):
        import numpy as np
        if self.current is not None:
            require(self.current.get("actual32_verified") is True, "Previous hierarchy node was not fully predicted")
        est = self.estimator
        require(est.n_estimators == MEMBERS and est.random_state == 0 and est.fit_mode == "fit_preprocessors",
                "TabPFN32 constructor recipe changed")
        est.fit(xs, ys)
        require(est.n_estimators_ == MEMBERS and len(est.ensemble_configs_) == MEMBERS,
                "Native fit did not generate32 member configurations")
        require(est.use_autocast_ is False and est.forced_inference_dtype_ == self.torch.float32,
                "Historical FP32/no-AMP recipe changed")
        members = est.executor_.ensemble_members
        seeds = est.executor_.ensemble_preprocessor.pipeline_seeds
        require(len(members) == len(seeds) == MEMBERS and len({id(m) for m in members}) == MEMBERS,
                "Native fitted executor does not hold32 distinct members")
        require(len({id(m.config) for m in members}) == MEMBERS and
                {id(m.config) for m in members} == {id(c) for c in est.ensemble_configs_},
                "Executor member configurations differ from generated native configurations")
        require(np.array_equal(est.classes_, np.arange(len(np.unique(ys)))), "Native class order changed")
        records = []
        self.member_by_config_identity = {}
        for index, (member, seed) in enumerate(zip(members, seeds, strict=True)):
            identity = {"configuration": json_value(member.config), "preprocessing_seed": int(seed)}
            record = {"member_index": index, "member_id": object_digest(identity), **identity}
            records.append(record)
            self.member_by_config_identity[id(member.config)] = record
        require(len({record["member_id"] for record in records}) == MEMBERS,
                "Native member configuration/seeds are not32 distinct identities")
        self.current = {"node_index": len(self.nodes), "support_rows": len(ys), "classes": len(est.classes_),
            "support_sha256": array_digest(xs), "support_labels_sha256": array_digest(ys),
            "actual_ensemble_count": MEMBERS, "member_audits": records,
            "successful_prediction_chunks": [], "discarded_oom_attempts": [],
            "postprocessing": {"softmax_temperature": float(getattr(est, "softmax_temperature_", est.softmax_temperature)),
                "average_before_softmax": bool(est.average_before_softmax),
                "balance_probabilities": bool(est.balance_probabilities),
                "native_class_order": est.classes_.tolist()}}
        self.nodes.append(self.current)
        return self

    def predict_chunk(self, query):
        import numpy as np
        est, rows = self.estimator, len(query)
        classes = self.current["classes"]
        observed, processed = [], []
        state = {"generator_exhausted": False, "probability_calls": 0, "mean_calls": 0}
        executor = est.executor_
        original_iter = executor.iter_outputs
        original_probability = est.logits_to_probabilities
        original_mean = est._average_across_estimators

        def iter_outputs(X, *args, **kwargs):
            require(len(X) == rows and kwargs.get("task_type") == "multiclass" and kwargs.get("autocast") is False,
                    "Native classifier executor query/task/precision changed")
            for output, config in original_iter(X, *args, **kwargs):
                require(len(observed) < MEMBERS, "Native executor emitted more than32 members")
                identity = self.member_by_config_identity.get(id(config))
                require(identity is not None, "Executor yielded an unknown fitted member configuration")
                require(identity["member_id"] not in {item["member_id"] for item in observed},
                        "Executor repeated an ensemble member")
                logits = as_array(output)
                require(logits.ndim == 2 and logits.shape[0] == rows and np.isfinite(logits).all(),
                        "Native member output is invalid or misses query rows")
                permutation = config.class_permutation
                if permutation is None:
                    selected = logits[:, :classes]
                else:
                    use_perm = np.asarray(permutation)
                    if len(use_perm) != classes:
                        require(len(use_perm) <= classes, "Native class permutation exceeds local classes")
                        use_perm = np.arange(classes)
                        use_perm[:len(permutation)] = permutation
                    selected = logits[:, use_perm]
                require(selected.shape == (rows, classes), "Native member class-aligned logits shape changed")
                # Only a CPU hash is retained, not a second32-member GPU tensor.
                prepared_hash = array_digest(np.asarray(selected, dtype=np.float32))
                processed.append(prepared_hash)
                observed.append({"member_index": identity["member_index"], "member_id": identity["member_id"],
                    "query_rows": rows, "raw_output_shape": list(logits.shape), "raw_output_sha256": array_digest(logits),
                    "class_aligned_logits_sha256": prepared_hash})
                yield output, config
            state["generator_exhausted"] = True

        def probabilities(raw_logits, *args, **kwargs):
            require(state["generator_exhausted"] and len(observed) == MEMBERS,
                    "Native aggregation began without32 consumed member outputs")
            require(tuple(raw_logits.shape) == (MEMBERS, rows, 1, classes),
                    "Native aggregation must receive all32 full-query class-aligned outputs")
            for index, expected in enumerate(processed):
                require(array_digest(raw_logits[index, :, 0, :]) == expected,
                        "A member was dropped, reordered, or replaced before native aggregation")
            state["probability_calls"] += 1
            require(state["probability_calls"] == 1, "Native probability processing repeated")
            return original_probability(raw_logits, *args, **kwargs)

        def average(tensors):
            require(state["probability_calls"] == 1 and tuple(tensors.shape) == (MEMBERS, rows, 1, classes),
                    "Native estimator mean did not receive32 full-query member tensors")
            state["mean_calls"] += 1
            require(state["mean_calls"] == 1, "Native estimator mean repeated")
            return original_mean(tensors)

        executor.iter_outputs = iter_outputs
        est.logits_to_probabilities = probabilities
        est._average_across_estimators = average
        try:
            result = np.asarray(est.predict_proba(query))
        finally:
            executor.iter_outputs = original_iter
            est.logits_to_probabilities = original_probability
            est._average_across_estimators = original_mean
        require(state == {"generator_exhausted": True, "probability_calls": 1, "mean_calls": 1},
                "Native32-way aggregation was not completed")
        require(len(observed) == MEMBERS and result.shape == (rows, classes) and np.isfinite(result).all()
                and (result >= 0).all() and (result.sum(axis=1) > 0).all(), "Invalid32-member probability output")
        return result, {"members": observed, "actual_ensemble_count": MEMBERS,
                        "native_aggregation_verified": True, "probability_sha256": array_digest(result)}

    def predict_full(self, query):
        import numpy as np
        require(self.current is not None and "actual32_verified" not in self.current,
                "Every node must predict the full test split exactly once")
        chunk, position, outputs = max(128, min(4096, len(query))), 0, []
        while position < len(query):
            stop = min(position + chunk, len(query))
            try:
                probability, record = self.predict_chunk(query[position:stop])
            except self.torch.OutOfMemoryError:
                self.current["discarded_oom_attempts"].append({"start": position, "stop": stop,
                                                              "included_in_final": False})
                if chunk <= 128:
                    raise
                chunk = max(128, chunk // 2)
                self.torch.cuda.empty_cache()
                continue
            outputs.append(probability)
            self.current["successful_prediction_chunks"].append({"start": position, "stop": stop, **record})
            position = stop
        result = np.concatenate(outputs, axis=0)
        require(result.shape == (len(query), self.current["classes"]), "Full-test probability coverage changed")
        self.current.update(actual32_verified=True, all_test_rows_covered=True, test_rows=len(query),
            minimum_contributions_per_test_row=MEMBERS, maximum_contributions_per_test_row=MEMBERS,
            minimum_test_chunk_used=chunk, full_test_sha256=array_digest(query))
        return result, chunk


def predict(model_key, arrays, model_config):
    import numpy as np
    model_config = normalize_config(model_config)
    xs, ys, xt = validate_inputs(arrays)
    before = tuple(array_digest(value) for value in (xs, ys, xt))
    estimator, hierarchy, torch, sources = load_runtime(model_key, model_config)
    audited = Native32Audit(estimator, torch)
    class_limit = MODEL_VERSIONS[model_key][1]
    if len(np.unique(ys)) > class_limit:
        probability, hierarchy_record = hierarchy.hierarchical_predict_proba(
            audited, xs, ys, xt, branch_factor=class_limit,
            predict_fn=lambda fitted, query: fitted.predict_full(query))
        require(hierarchy_record["nodes_fitted"] == len(audited.nodes), "Hierarchy node count changed")
    else:
        audited.fit(xs, ys)
        probability, _ = audited.predict_full(xt)
        hierarchy_record = None
    require(before == tuple(array_digest(value) for value in (xs, ys, xt)), "Adapter changed frozen input arrays")
    require(probability.shape == (len(xt), len(np.unique(ys))) and np.isfinite(probability).all(),
            "Invalid final full-test classification probabilities")
    require(audited.nodes and all(node.get("actual32_verified") is True and node["test_rows"] == len(xt)
                                 for node in audited.nodes), "Missing actual32 node coverage")
    audit = {"model_key": model_key, "actual32_verified": True, "actual_ensemble_count": MEMBERS,
        "n_estimators": MEMBERS, "actual_members_per_test_row": MEMBERS,
        "actual_member_scope": "per test row per fitted hierarchy node",
        "node_count": len(audited.nodes), "ensemble_audits": audited.nodes,
        "hierarchical_protocol": hierarchy_record, "full_test_split": True, "test_rows": len(xt),
        "seed": 0, "precision": "FP32", "amp": False, "training_performed": False,
        "query_chunk_policy": "historical4096 initial; nativeOOM backoff to128; no support cap changes",
        "aggregation": "unchanged pinned native class permutation, temperature/softmax,32-way mean and probability normalization; hierarchy products unchanged",
        "source_commit": SOURCE_COMMIT, "runtime_sources": sources,
        "adapter_source_sha256": file_digest(__file__), "model_sha256": model_config["model_sha256"],
        "probability_sha256": array_digest(probability)}
    return probability, audit
