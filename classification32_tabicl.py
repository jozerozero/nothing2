"""Actual32 adapter for frozen TabICLv1/v2 and native Taffy Loop3/4.

Public API: predict(model_key, {X_train, y_train, X_test}, model_config).
The JSON config requires source_root (containing src/tabicl), checkpoint_path,
checkpoint_sha256; Taffy additionally requires loop and checkpoint_step. Optional
source_files accepts relative-path SHA mappings or lists of absolute file identity
records. runtime_sources accepts absolute-path SHA mappings.
One model/source per subprocess is required. No downloads, fitting of weights,
query-label access, support subsampling, or query chunking is performed here.

Native 32-member generators are unchanged. Short native ensembles retain all
their original configurations, supplemented by deterministic support-row order
transformations. Each supplemental member makes a real forward contribution;
equal predictions remain possible and are NOT claimed statistically independent.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import sys


PROTOCOL = "classification-actual32-native-prefix-support-row-permutation-v1"
CLASSIFIER_SHA256 = "ea7c7a27e5d650958373876ffac16d46385ed4d49d99797c0ba0ce63d0276be1"
PREPROCESSING_SHA256 = "9213210d7cc94bd98ac2ec9c405d6ca5e7c9b7685080a9c4a05afcb1ed3f0198"
ALIASES = {"loop3": "taffy_loop3", "loop4": "taffy_loop4",
           "loop3-step19650": "taffy_loop3", "loop4-step22400": "taffy_loop4"}
MODELS = {"tabiclv1", "tabiclv2", "taffy_loop3", "taffy_loop4"}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), f"File changed during read: {path}")
    return h.hexdigest()


def array_sha256(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    require(a.dtype.kind != "O", "Object-array hashes are not stable")
    h = hashlib.sha256(json.dumps([a.dtype.str, list(a.shape)]).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _key(model_key):
    key = ALIASES.get(model_key, model_key)
    require(key in MODELS, f"Unsupported model: {model_key}")
    return key


def _check_taffy_config(config, loop):
    required = {"shared_depth_icl_enabled": True,
                "shared_depth_icl_dataset_conditioned": True,
                "shared_depth_icl_num_passes": loop,
                "shared_depth_icl_rho": 1.0, "icl_num_blocks": 12}
    require(all(config.get(k) == v for k, v in required.items()),
            f"Native Taffy Loop{loop} checkpoint config mismatch")
    require(config.get("max_classes", 0) > 0, "Regression checkpoint is not permitted")


def _verify_source_pins(root, config):
    root = Path(root).resolve(strict=True)
    pins = config.get("source_files", {})
    entries = ((record["path"], record) for record in pins) if isinstance(pins, list) else pins.items()
    entries = list(entries) + list(config.get("runtime_sources", {}).items())
    for relative, expected in entries:
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root), "Source pin escapes source_root")
        if isinstance(expected, dict):
            stat = path.stat()
            size = expected.get("size_bytes", expected.get("size"))
            require(size is None or stat.st_size == size, f"Source size mismatch: {path}")
            require("mtime_ns" not in expected or stat.st_mtime_ns == expected["mtime_ns"],
                    f"Source timestamp mismatch: {path}")
            expected = expected["sha256"]
        require(file_sha256(path) == expected, f"Source pin mismatch: {relative}")


def build_estimator(model_key, model_config):
    """Recreate the original frozen classification constructor, without downloads.

    The caller must set PYTHONHASHSEED=0 before starting Python, just as the
    original campaign did. This matters because native norm-group ordering uses
    a set; changing it changes floating-point accumulation order.
    """
    key = _key(model_key)
    require(os.environ.get("PYTHONHASHSEED") == "0", "Start worker with PYTHONHASHSEED=0")
    root = Path(model_config["source_root"]).resolve(strict=True)
    package = root / "src" / "tabicl"
    require(package.is_dir(), "source_root must contain src/tabicl")
    _verify_source_pins(root, model_config)
    checkpoint = Path(model_config["checkpoint_path"]).resolve(strict=True)
    require(file_sha256(checkpoint) == model_config["checkpoint_sha256"], "Checkpoint SHA256 mismatch")
    if key.startswith("taffy_"):
        loop = 3 if key == "taffy_loop3" else 4
        step = {3: 19650, 4: 22400}[loop]
        require(model_config.get("loop") == loop and model_config.get("checkpoint_step") == step,
                "Only native Loop3 step19650 and Loop4 step22400 are authorized")
        require(checkpoint.name == f"step-{step}.ckpt", "Taffy checkpoint filename/step mismatch")
        os.environ["CROSS_TABLE_ARM"] = "E4"
    sys.path.insert(0, str(root / "src"))
    module = importlib.import_module("tabicl")
    require(Path(module.__file__).resolve().is_relative_to(package), "Wrong imported tabicl source")
    classifier = module.TabICLClassifier
    classifier_file = Path(inspect.getfile(classifier)).resolve()
    require(classifier_file.is_relative_to(package) and file_sha256(classifier_file) == CLASSIFIER_SHA256,
            "Classifier differs from audited deployed runtime")
    generator_class = importlib.import_module(classifier.__module__).EnsembleGenerator
    preprocessing_file = Path(inspect.getfile(generator_class)).resolve()
    require(preprocessing_file.is_relative_to(package)
            and file_sha256(preprocessing_file) == PREPROCESSING_SHA256,
            "Preprocessing differs from audited deployed runtime")
    # These explicit values equal the deployed evaluator and classifier defaults.
    kwargs = dict(n_estimators=32, norm_methods=["none", "power"], batch_size=8,
                  n_jobs=1, device="cuda", use_amp=False, use_fa3=False,
                  kv_cache=False, allow_auto_download=False, model_path=str(checkpoint),
                  verbose=False, random_state=42, feat_shuffle_method="latin",
                  class_shuffle_method="shift", outlier_threshold=4.0,
                  average_logits=True, softmax_temperature=0.9,
                  support_prior_match_alpha=0.0, support_many_classes=True,
                  use_pseudo_ssmax_thinking=False, offload_mode="auto")
    estimator = classifier(**kwargs)
    estimator._actual32_factory_audit = {
        "model_key": key, "source_root": str(root), "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": model_config["checkpoint_sha256"],
        "classifier_sha256": CLASSIFIER_SHA256, "preprocessing_sha256": PREPROCESSING_SHA256,
        "constructor": kwargs, "checkpoint_step": model_config.get("checkpoint_step"),
        "loop": model_config.get("loop"), "PYTHONHASHSEED": "0"}
    return estimator


def _replace(instance, name, value, restoration):
    """Temporarily replace an instance method without changing its class/source."""
    existed = name in instance.__dict__
    old = instance.__dict__.get(name)
    restoration.append((instance, name, existed, old))
    setattr(instance, name, value)


def _restore(restoration):
    for instance, name, existed, old in reversed(restoration):
        if existed:
            setattr(instance, name, old)
        else:
            delattr(instance, name)


def _configuration(norm, feature_order, class_order, support_order):
    return {"normalization": str(norm), "feature_order": [int(x) for x in feature_order],
            "class_order": [int(x) for x in class_order],
            "support_row_order_sha256": array_sha256(support_order)}


def prepare_members(generator, seed=42):
    """Plan support-only supplements; no generator mutation and no query access.

    Returns replacement config maps, ordered member records, and row-index maps.
    Original native configs are retained in their original normalization groups.
    """
    import numpy as np
    n_support = len(generator.y_)
    require(n_support >= 2 and generator.classification, "Expected classification support")
    original = generator.ensemble_configs_
    native_count = sum(len(configs) for configs in original.values())
    require(1 <= native_count <= 32, "Native ensemble count outside 1..32")
    identity = np.arange(n_support, dtype=np.int64)
    configs = OrderedDict((norm, list(values)) for norm, values in original.items())
    records, orders, seen = OrderedDict(), OrderedDict(), set()
    bases = []
    for norm, values in configs.items():
        records[norm], orders[norm] = [], []
        for feature_order, class_order in values:
            config = _configuration(norm, feature_order, class_order, identity)
            h = digest(config)
            require(h not in seen, "Duplicate native ensemble configuration")
            seen.add(h)
            record = {"configuration": config, "configuration_sha256": h,
                      "origin": "native", "support_row_permutation": "identity"}
            records[norm].append(record)
            orders[norm].append(identity)
            bases.append((norm, feature_order, class_order))
    if native_count == 32:
        return native_count, configs, records, orders
    rng_seed = int(digest({"protocol": PROTOCOL, "seed": seed,
                          "support_rows": n_support, "native": sorted(seen)})[:16], 16)
    rng = random.Random(rng_seed)
    # The float32 support view is exactly what native _batch_forward consumes.
    # Reject permutations whose resulting support inputs AND labels are unchanged.
    support_views = {}
    for norm, feature_order, class_order in bases:
        base_key = digest([str(norm), list(feature_order), list(class_order)])
        if base_key not in support_views:
            X = np.asarray(generator.preprocessors_[norm].X_transformed_[:, feature_order], dtype=np.float32)
            y = np.asarray(class_order, dtype=np.float32)[np.asarray(generator.y_, dtype=int)]
            support_views[base_key] = (X, y, {(array_sha256(X), array_sha256(y))})
    draws = 0
    for extra in range(32 - native_count):
        norm, feature_order, class_order = bases[extra % native_count]
        base_key = digest([str(norm), list(feature_order), list(class_order)])
        X, y, input_hashes = support_views[base_key]
        for _ in range(4096):
            draws += 1
            order = list(range(n_support))
            rng.shuffle(order)
            order = np.asarray(order, dtype=np.int64)
            config = _configuration(norm, feature_order, class_order, order)
            h = digest(config)
            if h in seen:
                continue
            input_hash = (array_sha256(X[order]), array_sha256(y[order]))
            if input_hash in input_hashes:
                continue
            seen.add(h)
            input_hashes.add(input_hash)
            configs[norm].append((feature_order, class_order))
            orders[norm].append(order)
            records[norm].append({"configuration": config, "configuration_sha256": h,
                "origin": "supplemental", "support_row_permutation": "deterministic_random",
                "permutation_rng_seed": rng_seed, "permutation_draw": draws,
                "support_features_sha256": input_hash[0], "support_labels_sha256": input_hash[1]})
            break
        else:
            raise RuntimeError("Cannot realize 32 distinct support transformations; no padding permitted")
    require(len(seen) == 32, "Expected 32 unique configurations")
    return native_count, configs, records, orders


def _loop_hooks(estimator, loop, handles, calls):
    if loop is None:
        return
    _check_taffy_config(estimator.model_config_, loop)
    encoder = estimator.model_.icl_predictor.tf_icl
    require(encoder.shared_depth_enabled and encoder.shared_depth_dataset_conditioned
            and encoder.shared_depth_num_passes == loop and encoder.shared_depth_rho == 1.0
            and len(encoder.blocks) == 12, "Wrong realized Taffy encoder")
    counts = [0] * 12
    def before(*_):
        counts[:] = [0] * 12
    def after(_module, _args, output):
        require(counts == [loop] * 12, f"Wrong realized Loop{loop} block calls: {counts}")
        require(str(output.dtype) == "torch.float32", "Taffy encoder output is not FP32")
        calls.append(list(counts))
    handles.extend([encoder.register_forward_pre_hook(before), encoder.register_forward_hook(after)])
    for index, block in enumerate(encoder.blocks):
        def count(*_, index=index):
            counts[index] += 1
        handles.append(block.register_forward_hook(count))


def _memory_cap(model_key, config, restoration):
    level = int(config.get("historical_memory_level", 0))
    require(level in (0, 1), "Unsupported historical memory level")
    if not level:
        return
    require(model_key.startswith("taffy_") and config.get("dataset") == "talent__walking-activity",
            "Historical level1 caps apply only to Taffy walking-activity")
    from tabicl._model.inference import InferenceManager
    original = InferenceManager.estimate_safe_batch_size
    def estimate(self, seq_len, include_inputs=True, in_dim=None, max_bs=50000):
        gpu_mb, adaptive = original(self, seq_len, include_inputs, in_dim, max_bs)
        cap = {"tf_col": 8, "tf_row": 7500, "tf_icl": 1}[self.enc_name]
        require(self.min_batch_size <= cap, "Historical cap below native minimum")
        return gpu_mb, max(self.min_batch_size, min(int(adaptive), int(max_bs), cap))
    _replace(InferenceManager, "estimate_safe_batch_size", estimate, restoration)


def predict_estimator(estimator, arrays, *, model_key, model_config=None):
    """Audited fit/predict primitive; factory-independent to allow CPU contract tests."""
    import numpy as np
    key, config = _key(model_key), model_config or {}
    require(set(arrays) == {"X_train", "y_train", "X_test"}, "Only support labels may enter this adapter")
    X_train, y_train, X_test = (arrays[k] for k in ("X_train", "y_train", "X_test"))
    n_support, n_test = len(y_train), len(X_test)
    require(len(X_train) == n_support and n_test > 0, "Invalid support/query sizes")
    require(estimator.n_estimators == 32 and not estimator.kv_cache
            and not getattr(estimator, "use_pseudo_ssmax_thinking", False),
            "Expected native est32, no cache, no pseudo-label inference")
    restoration, handles, loop_calls, forward_batches = [], [], [], []
    aggregate_calls, batch_cursor = [], [0]
    loop = {"taffy_loop3": 3, "taffy_loop4": 4}.get(key)
    try:
        _memory_cap(key, config, restoration)
        estimator.fit(X_train, y_train)
        require(getattr(estimator, "model_kv_cache_", None) is None, "Unexpected fitted KV cache")
        _loop_hooks(estimator, loop, handles, loop_calls)
        generator = estimator.ensemble_generator_
        native_count, configs, grouped, orders = prepare_members(generator)
        members = [record for records in grouped.values() for record in records]
        for index, record in enumerate(members):
            record.update(member_index=index, forward_contributions=0, aggregation_contributions=0,
                          test_rows=n_test, full_test_coverage=False)
        require(len(members) == 32, "Member plan must contain exactly32")
        old_transform = generator.transform
        if native_count < 32:
            _replace(generator, "ensemble_configs_", configs, restoration)
            _replace(generator, "feature_shuffles_", OrderedDict(
                (norm, [pair[0] for pair in values]) for norm, values in configs.items()), restoration)
            _replace(generator, "class_shuffles_", OrderedDict(
                (norm, [pair[1] for pair in values]) for norm, values in configs.items()), restoration)
            def transformed(*args, **kwargs):
                require(kwargs.get("mode", "both") == "both", "Only full native support+query transform permitted")
                data = old_transform(*args, **kwargs)
                for norm, (Xs, ys) in data.items():
                    require(Xs.shape[:2] == (len(grouped[norm]), n_support + n_test)
                            and ys.shape == (len(grouped[norm]), n_support), "Transformed support/query size changed")
                    for i, record in enumerate(grouped[norm]):
                        if record["origin"] == "supplemental":
                            order = orders[norm][i]
                            Xs[i, :n_support] = Xs[i, :n_support][order]
                            ys[i] = ys[i][order]
                return data
            _replace(generator, "transform", transformed, restoration)
        # Monitor model calls, including batched members: one call is NOT one member.
        state = {"active": None, "offset": 0, "pending": None}
        def model_before(_module, args, kwargs):
            require(state["active"] is not None and state["pending"] is None, "Unscoped/nested top-level forward")
            X = kwargs.get("X", args[0] if args else None)
            y = kwargs.get("y_train", args[1] if len(args) > 1 else None)
            require(X is not None and y is not None, "Unrecognized native forward signature")
            size = int(X.shape[0])
            start = state["offset"]
            selected = state["active"][start:start + size]
            require(len(selected) == size and size > 0 and tuple(y.shape) == (size, n_support)
                    and X.shape[1] == n_support + n_test, "Forward omitted members/support/query rows")
            state["pending"] = selected
            state["offset"] += size
        def model_after(_module, _args, _kwargs, output):
            selected = state["pending"]
            require(selected is not None and tuple(output.shape) ==
                    (len(selected), n_test, len(estimator.classes_)), "Forward output coverage/class count mismatch")
            for record in selected:
                record["forward_contributions"] += 1
                record["full_test_coverage"] = True
            forward_batches.append({"member_indices": [m["member_index"] for m in selected],
                "batch_members": len(selected), "support_rows": n_support, "test_rows": n_test,
                "output_shape": [int(x) for x in output.shape]})
            state["pending"] = None
        handles.extend([estimator.model_.register_forward_pre_hook(model_before, with_kwargs=True),
                        estimator.model_.register_forward_hook(model_after, with_kwargs=True)])
        native_batch = estimator._batch_forward
        native_aggregate = estimator._aggregate_ensemble_outputs
        norms = list(grouped)
        def batch_forward(Xs, ys, feature_shuffles=None):
            pos = batch_cursor[0]
            require(pos < len(norms), "Unexpected additional prediction pass")
            selected = grouped[norms[pos]]
            require(Xs.shape[0] == len(selected), "Batch member count mismatch")
            state.update(active=selected, offset=0, pending=None)
            outputs = native_batch(Xs, ys, feature_shuffles)
            require(state["pending"] is None and state["offset"] == len(selected), "Not all members actually forwarded")
            require(outputs.shape == (len(selected), n_test, len(estimator.classes_))
                    and np.isfinite(outputs).all(), "Invalid native member outputs")
            for record, output in zip(selected, outputs):
                record["raw_output_sha256"] = array_sha256(output)
            state["active"] = None
            batch_cursor[0] += 1
            return outputs
        def aggregate(outputs, class_shuffles):
            require(not aggregate_calls and outputs.shape == (32, n_test, len(estimator.classes_))
                    and len(class_shuffles) == 32, "Aggregation does not contain exactly32 members")
            for record, output, shuffle in zip(members, outputs, class_shuffles):
                require(record["forward_contributions"] == 1 and record["raw_output_sha256"] == array_sha256(output),
                        "Aggregated member did not come from its verified native forward")
                require(record["configuration"]["class_order"] == [int(i) for i in shuffle],
                        "Aggregation class alignment changed")
                record["aggregation_contributions"] += 1
            aggregate_calls.append(32)
            return native_aggregate(outputs, class_shuffles)
        _replace(estimator, "_batch_forward", batch_forward, restoration)
        _replace(estimator, "_aggregate_ensemble_outputs", aggregate, restoration)
        # Native feature-mask handling can modify its argument; never mutate caller data.
        probabilities = estimator.predict_proba(X_test.copy())
        require(batch_cursor[0] == len(norms) and aggregate_calls == [32]
                and all(m["forward_contributions"] == m["aggregation_contributions"] == 1
                        and m["full_test_coverage"] for m in members), "Actual32 participation verification failed")
        require(probabilities.shape == (n_test, len(estimator.classes_))
                and np.isfinite(probabilities).all() and (probabilities >= 0).all()
                and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5), "Invalid full-query probabilities")
        require(loop is None or loop_calls, "No verified native Taffy loop calls")
        audit = {"protocol": PROTOCOL, "model_key": key, "configured_n_estimators": 32,
            "native_ensemble_count": native_count, "supplemental_ensemble_count": 32 - native_count,
            "native_generator_unchanged": native_count == 32, "native_configurations_retained": True,
            "actual_ensemble_count": 32, "actual32_verified": True,
            "actual_members_per_test_row": 32,
            "unique_configuration_count": len({m["configuration_sha256"] for m in members}),
            "support_rows": n_support, "test_rows": n_test, "full_test_split": True,
            "all_test_rows_covered": True, "minimum_contributions_per_test_row": 32,
            "maximum_contributions_per_test_row": 32, "test_labels_used": False,
            "query_order_unchanged": True, "support_row_membership_unchanged": True,
            "aggregation": "unchanged native class-aligned mean logits then temperature softmax"
                           if estimator.average_logits else "unchanged native class-aligned mean probabilities",
            "statistically_independent_predictions_claimed": False,
            "equal_prediction_members_allowed": True, "prediction_padding_used": False,
            "member_audits": members, "forward_batches": forward_batches,
            "realized_loop_block_calls": loop_calls,
            "precision": {"dtype": "float32", "amp": False, "fa3": False},
            "historical_memory_level": int(config.get("historical_memory_level", 0)),
            "factory": getattr(estimator, "_actual32_factory_audit", None),
            "evidence": "top-level model pre/post hooks count batched members; native aggregation verifies each output hash"}
        return probabilities, audit
    finally:
        for handle in reversed(handles):
            handle.remove()
        _restore(restoration)


def predict(model_key, arrays, model_config):
    """Build one frozen estimator, predict all query rows, and return JSON-safe audit."""
    model_config = dict(model_config)
    historical_level = int(_key(model_key).startswith("taffy_")
                           and model_config.get("dataset") == "talent__walking-activity")
    require(int(model_config.get("historical_memory_level", historical_level)) == historical_level,
            "Historical Taffy walking-activity memory policy must remain unchanged")
    model_config["historical_memory_level"] = historical_level
    estimator = build_estimator(model_key, model_config)
    output, audit = predict_estimator(estimator, arrays, model_key=model_key, model_config=model_config)
    require(file_sha256(model_config["checkpoint_path"]) == model_config["checkpoint_sha256"],
            "Checkpoint changed during inference")
    audit["classes"] = estimator.classes_.tolist()
    return output, audit
