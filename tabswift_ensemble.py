"""Per-estimator TabSwift ensemble auditing and strictly bounded count extension.

Official16 runs its native fit/configuration/prediction unchanged. Strict32/8
first runs the same native fit, then repeats existing configurations only when
native low-dimensional deduction generated too few. No new RNG calls, source
patches, model weights, preprocessing, labels, row subsets or inference settings.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from copy import deepcopy
import hashlib
import json


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def configuration_records(generator, task_kind):
    records = []
    for method, configs in generator.ensemble_configs_.items():
        require(method in generator.feature_shuffle_patterns_ and method in generator.class_shift_offsets_,
                "Incomplete native generator configuration maps")
        patterns = generator.feature_shuffle_patterns_[method]
        offsets = generator.class_shift_offsets_[method]
        require(len(configs) == len(patterns) == len(offsets) > 0, "Native configuration map lengths differ")
        for index, (pattern, offset) in enumerate(configs):
            order = [int(i) for i in pattern]
            require(order == [int(i) for i in patterns[index]] and int(offset) == int(offsets[index]),
                    "Native configuration maps disagree")
            require(sorted(order) == list(range(generator.n_features_in_)), "Invalid native feature permutation")
            record = {"normalization": str(method), "feature_order": order, "class_shift": int(offset)}
            effective = dict(record, class_shift=int(offset) if task_kind == "classification" else 0)
            record["configuration_sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            record["effective_configuration_sha256"] = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
            records.append(record)
    require(records, "Native ensemble is empty")
    return records


def extend_native_configs(generator, target, task_kind):
    """Preserve every native per-normalization prefix, cycling only native configs.

Appended members are balanced across the original normalization groups. The
native grouped traversal order is preserved, but extending an earlier group can
insert repeated entries before a later group's original entries. This is audited
explicitly; repeated configurations are executions, not unique/independent models.
"""
    before = configuration_records(generator, task_kind)
    require(0 < len(before) <= target, "Cannot shrink or replace a native ensemble")
    if len(before) == target:
        return {"extended": False, "native_count": len(before), "added_member_count": 0,
                "native_configuration_records": before, "native_group_prefixes_preserved": True}
    base = OrderedDict((method, deepcopy(configs)) for method, configs in generator.ensemble_configs_.items())
    expanded = OrderedDict((method, deepcopy(configs)) for method, configs in base.items())
    methods = list(base)
    added = Counter()
    while sum(map(len, expanded.values())) < target:
        method = min(methods, key=lambda name: (len(expanded[name]), methods.index(name)))
        expanded[method].append(deepcopy(base[method][added[method] % len(base[method])]))
        added[method] += 1
    generator.ensemble_configs_ = expanded
    generator.feature_shuffle_patterns_ = OrderedDict((method, [c[0] for c in configs]) for method, configs in expanded.items())
    generator.class_shift_offsets_ = OrderedDict((method, [c[1] for c in configs]) for method, configs in expanded.items())
    after = configuration_records(generator, task_kind)
    require(len(after) == target, "Strict extension failed its actual configuration count")
    return {"extended": True, "native_count": len(before), "added_member_count": target - len(before),
            "added_by_normalization": dict(added), "native_configuration_records": before,
            "native_group_prefixes_preserved": True,
            "global_flattened_prefix_preserved": after[:len(before)] == before,
            "extension": "cycle existing configs within native normalization groups; balance group sizes; no RNG draws"}


def _hash_array(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    require(value.dtype.kind != "O", "Object prediction cannot be hashed")
    digest = hashlib.sha256(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}, sort_keys=True).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


class EnsembleAudit:
    def __init__(self, estimator, task_kind, requested_count, strict_actual_count):
        require(task_kind in ("classification", "regression"), "Unknown TabSwift task kind")
        require(type(requested_count) is int and requested_count > 0, "Invalid requested member count")
        require(type(strict_actual_count) is bool, "strict_actual_count must be a bool")
        target = 32 if task_kind == "classification" else 8
        require(requested_count == (target if strict_actual_count else 16), "Only official16 and strict32/8 protocols are supported")
        require(estimator.n_estimators == requested_count, "Estimator requested count differs from protocol")
        require(not hasattr(estimator, "ensemble_generator_"), "Configure audit before native fit")
        require(not hasattr(estimator, "_tabswift_ensemble_audit"), "Estimator already instrumented")
        if task_kind == "regression":
            require(estimator.class_shift is False, "Official regression recipe requires class_shift=False")
        self.estimator = estimator
        self.task_kind = task_kind
        self.requested_count = requested_count
        self.strict = strict_actual_count
        self.patches = []
        self.hooks = []
        self.transforms = []
        self.model_calls = []
        self.icl_calls = []
        self.members = []
        self.batches = []
        self.fit_complete = False
        self.prediction_complete = False
        self.in_prediction = False
        self.expected_sum = None
        self.cursor = 0
        self.report = None
        self._install()

    def _patch(self, owner, name, replacement):
        existed = name in owner.__dict__
        previous = owner.__dict__.get(name)
        self.patches.append((owner, name, existed, previous))
        setattr(owner, name, replacement)

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        for owner, name, existed, previous in reversed(self.patches):
            if existed:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self.patches.clear()

    def _install(self):
        estimator = self.estimator
        original_fit = estimator.fit
        original_predict = estimator.predict_proba
        original_transform = estimator._apply_dimensionality_transform

        def transform(X, *args, **kwargs):
            input_shape = tuple(X.shape)
            result = original_transform(X, *args, **kwargs)
            require(len(result.shape) == 2 and result.shape[0] == input_shape[0],
                    "Native PCA/dimensionality transform changed row count")
            fitting = kwargs.get("is_fitting", args[0] if args else False)
            self.transforms.append({"phase": "fit" if fitting else "predict", "input_shape": list(input_shape),
                                    "output_shape": list(result.shape), "row_count_unchanged": True})
            return result

        def fit(X, y, *args, **kwargs):
            require(not self.fit_complete, "One fresh estimator fit per audit is required")
            self.support_rows = len(X)
            self.original_features = int(X.shape[1])
            require(len(y) == self.support_rows > 0, "Invalid original support rows")
            result = original_fit(X, y, *args, **kwargs)
            generator = estimator.ensemble_generator_
            require(len(generator.X_) == len(generator.y_) == self.support_rows,
                    "Native fit dropped/truncated support rows")
            native = configuration_records(generator, self.task_kind)
            self.extension = {"extended": False, "native_count": len(native), "added_member_count": 0,
                              "native_configuration_records": native, "native_group_prefixes_preserved": True}
            if self.strict:
                self.extension = extend_native_configs(generator, self.requested_count, self.task_kind)
            self.configurations = configuration_records(generator, self.task_kind)
            require(0 < len(self.configurations) <= self.requested_count, "Native ensemble exceeds requested count")
            if self.strict:
                require(len(self.configurations) == self.requested_count, "Strict ensemble count not achieved")
            self._instrument_model()
            self.fit_complete = True
            return result

        def predict_proba(X, *args, **kwargs):
            import numpy as np
            require(self.fit_complete and not self.prediction_complete and not self.in_prediction,
                    "Audit requires exactly one native full-test prediction after fit")
            self.test_rows = len(X)
            require(self.test_rows > 0, "Empty test split")
            self.in_prediction = True
            try:
                result = original_predict(X, *args, **kwargs)
            finally:
                self.in_prediction = False
            observed = np.asarray(result)
            count = len(self.configurations)
            expected_width = estimator.n_classes_ if self.task_kind == "classification" else 1
            require(observed.shape == (self.test_rows, expected_width) and np.isfinite(observed).all(),
                    "Native prediction output shape/values do not cover the full test split")
            require(self.cursor == count == len(self.members), "Some generated ensemble members were not aggregated")
            require(sum(call["members"] for call in self.model_calls) == count,
                    "Actual successful top-level model forward members differ from aggregation")
            require(self.expected_sum is not None, "No member predictions captured")
            self.expected_sum /= count
            if self.task_kind == "classification" and estimator.average_logits:
                self.expected_sum = estimator.softmax(self.expected_sum, axis=-1, temperature=estimator.softmax_temperature)
            require(np.array_equal(observed, self.expected_sum), "Native final aggregation differs from audited equal-weight members")
            self.aggregate_max_abs_error = float(np.max(np.abs(observed - self.expected_sum)))
            self.prediction_sha256 = _hash_array(observed)
            self.expected_sum = None
            self.prediction_complete = True
            return result

        self._patch(estimator, "fit", fit)
        self._patch(estimator, "predict_proba", predict_proba)
        self._patch(estimator, "_apply_dimensionality_transform", transform)
        self._patch(estimator, "_tabswift_ensemble_audit", self)

    def _instrument_model(self):
        estimator = self.estimator
        model = estimator.model_
        require(hasattr(model, "register_forward_hook") and hasattr(model, "icl_predictor"),
                "Expected native TabSwift model and ICL predictor")
        original_batch = estimator._batch_forward

        def model_hook(_module, args, kwargs, output):
            require(self.in_prediction, "Unexpected model forward outside native full-test predict")
            X = args[0] if args else kwargs["X"]
            y = args[1] if len(args) > 1 else kwargs["y_train"]
            require(len(X.shape) == 3 and len(y.shape) == 2 and X.shape[0] == y.shape[0]
                    and y.shape[1] == self.support_rows and X.shape[1] == self.support_rows + self.test_rows,
                    "Native top-level forward changed support/test row coverage")
            require(len(output.shape) == 3 and output.shape[0] == X.shape[0] and output.shape[1] == self.test_rows,
                    "Native top-level model output lost member/test rows")
            self.model_calls.append({"members": int(X.shape[0]), "support_rows": int(y.shape[1]),
                                     "test_rows": self.test_rows, "features": int(X.shape[2]),
                                     "output_shape": list(output.shape)})

        self.hooks.append(model.register_forward_hook(model_hook, with_kwargs=True))
        predictor = model.icl_predictor
        for method in ("_icl_predictions", "_icl_predictions_reg"):
            require(hasattr(predictor, method), "Native ICL prediction hook missing: " + method)
            self._instrument_icl(predictor, method)

        def batch(Xs, ys, *args, **kwargs):
            import numpy as np
            require(self.in_prediction and Xs.ndim == 3 and ys.ndim == 2,
                    "Unexpected native batch prediction input")
            count, total, _features = Xs.shape
            require(ys.shape == (count, self.support_rows) and total == self.support_rows + self.test_rows,
                    "Native ensemble batch dropped support/query rows")
            output = original_batch(Xs, ys, *args, **kwargs)
            output_array = np.asarray(output)
            width = estimator.n_classes_ if self.task_kind == "classification" else 1
            require(output_array.shape == (count, self.test_rows, width) and np.isfinite(output_array).all(),
                    "Native batch member predictions incomplete/nonfinite")
            require(self.cursor + count <= len(self.configurations), "More member outputs than generated configs")
            self.batches.append({"start_member": self.cursor, "members": count,
                                 "support_rows": self.support_rows, "test_rows": self.test_rows})
            for raw in output_array:
                config = self.configurations[self.cursor]
                offset = config["class_shift"]
                corrected = np.concatenate([raw[..., offset:], raw[..., :offset]], axis=-1)
                if self.expected_sum is None:
                    self.expected_sum = corrected
                else:
                    self.expected_sum += corrected
                self.members.append({"member_index": self.cursor, "configuration_sha256": config["configuration_sha256"],
                                     "test_rows": self.test_rows, "aggregation_weight": 1. / len(self.configurations),
                                     "raw_prediction_sha256": _hash_array(raw)})
                self.cursor += 1
            return output

        self._patch(estimator, "_batch_forward", batch)

    def _instrument_icl(self, predictor, name):
        original = getattr(predictor, name)
        def inner(*args, **kwargs):
            require(self.in_prediction, "ICL forward occurred outside native prediction")
            R = args[0] if args else kwargs["R"]
            y = args[1] if len(args) > 1 else kwargs["y_train"]
            require(len(R.shape) == 3 and y.shape[0] == R.shape[0]
                    and R.shape[1] - y.shape[1] == self.test_rows,
                    "Native ICL node/batch dropped or truncated query rows")
            require(0 < y.shape[1] <= self.support_rows, "Invalid hierarchy support subset")
            result = original(*args, **kwargs)
            registers = int(predictor.register_tokens)
            require(result.shape[0] == R.shape[0] and result.shape[1] == R.shape[1] + registers,
                    "Native ICL register/support/query output shape changed")
            self.icl_calls.append({"method": name, "member_node_evaluations": int(R.shape[0]),
                                   "node_support_rows": int(y.shape[1]), "test_rows": self.test_rows,
                                   "register_tokens": registers, "output_shape": list(result.shape)})
            return result
        self._patch(predictor, name, inner)

    def finish(self, test_rows):
        require(self.prediction_complete and test_rows == self.test_rows, "No complete audited prediction for this test split")
        require(self.icl_calls, "No actual inner ICL execution observed")
        count = len(self.members)
        if self.strict:
            require(count == self.requested_count, "Strict actual ensemble count failed")
        unique = len({r["effective_configuration_sha256"] for r in self.configurations})
        model = self.estimator.model_
        hierarchical = self.task_kind == "classification" and self.estimator.n_classes_ > model.max_classes
        inner_count = sum(r["member_node_evaluations"] for r in self.icl_calls)
        require(inner_count >= count, "Inner ICL execution coverage smaller than top-level ensembles")
        report = {"protocol": "strict_actual32_classification_actual8_regression" if self.strict else "official16_native_deduction",
                  "requested_n_estimators": self.requested_count, "actual_ensemble_count": count,
                  "requested_ensemble_count": self.requested_count, "strict_actual_count": self.strict,
                  "actual_members_per_test_row": count, "actual_member_forward_verified": True,
                  "actual_count_verified": True, "strict_actual_count_verified": self.strict,
                  "unique_configuration_count": unique, "duplicate_configuration_member_count": count - unique,
                  "configuration_identity_scope": "normalization+feature_permutation+effective_class_shift; not independent checkpoints",
                  "same_native_checkpoint_all_members": True, "extension_audit": self.extension,
                  "configurations": self.configurations, "members": self.members, "native_batches": self.batches,
                  "successful_model_forward_calls": self.model_calls, "icl_execution_calls": self.icl_calls,
                  "icl_member_node_evaluations": inner_count, "native_hierarchy_used": hierarchical,
                  "icl_count_includes_hierarchy_and_possible_native_OOM_recomputation": True,
                  "ensemble_count_scope": "top-level ensemble members contributing to every final test-row prediction",
                  "support_rows": self.support_rows, "test_rows": self.test_rows, "full_test_split": True,
                  "full_original_support_at_top_level": True, "external_support_or_query_subsampling": False,
                  "dimensionality_transforms": self.transforms, "native_PCA_unchanged": True,
                  "original_features": self.original_features,
                  "effective_generator_features": int(self.estimator.ensemble_generator_.n_features_in_),
                  "aggregation": "native class-shift correction, equal member weights, native logit/probability choice",
                  "aggregation_verified_exact": True, "aggregation_max_abs_error": self.aggregate_max_abs_error,
                  "prediction_sha256": self.prediction_sha256, "random_state": self.estimator.random_state,
                  "helper_consumed_no_random_numbers": True, "batch_size": self.estimator.batch_size}
        self.report = report
        return report


def configure(estimator, task_kind, requested_count, strict_actual_count=False):
    """Call before fit; use native fit/predict; finish(test_rows) returns JSON audit."""
    return EnsembleAudit(estimator, task_kind, requested_count, strict_actual_count)
