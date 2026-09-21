"""TabICLv2 balanced-label hierarchy with native-default TabFM at every node.

Adapted from TabICL 2.1.1, commit a81a2a278e32dfeec15d616be9036afd0e151d27,
src/tabicl/_model/learning.py: _grouping, _fit_node, _predict_hierarchical.
The grouping and probability-chain rule are identical. The predictor is NOT
native TabICL: each node runs the full native TabFMClassifier preprocessing and
32-member aggregation, then node probabilities are multiplied along the tree.
TabICL's learned-row-representation reuse / per-member hierarchy is not claimed.

Only training labels determine the tree. All test rows visit every node. One
already-loaded, eval-mode model is shared; estimators are discarded sequentially.
No feature/context/seed/dtype override, fine-tuning, or test labels are accepted.

Adapted hierarchy portions: BSD 3-Clause License.
Copyright (c) 2025, Soda team @ Inria

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from typing import Any, Callable

import numpy as np


REFERENCE = {
    "package": "TabICL", "version": "2.1.1",
    "commit": "a81a2a278e32dfeec15d616be9036afd0e151d27",
    "file": "src/tabicl/_model/learning.py",
    "sha256": "93e1a0c9e6f9a0c8f3d6114dec414ec859d34d6cf4a97010241d2d704473a81f",
    "methods": ["_grouping", "_fit_node", "_predict_hierarchical"],
}
PROTOCOL = "tabicl-v2-balanced-label-tree-tabfm-native-defaults-per-node-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def balanced_groups(num_classes: int, max_classes: int) -> tuple[np.ndarray, int]:
    """NumPy translation of TabICLv2 ICLearning._grouping, without randomness."""
    _require(num_classes > 0 and max_classes >= 2, "Invalid hierarchy class capacity")
    assignments = np.zeros(num_classes, dtype=np.int64)
    if num_classes <= max_classes:
        return assignments, 1
    num_groups = min(math.ceil(num_classes / max_classes), max_classes)
    current_pos = 0
    remaining_classes, remaining_groups = num_classes, num_groups
    for group in range(num_groups):
        group_size = math.ceil(remaining_classes / remaining_groups)
        assignments[current_pos:current_pos + group_size] = group
        current_pos += group_size
        remaining_classes -= group_size
        remaining_groups -= 1
    return assignments, num_groups


def _row_subset(X: Any, rows: np.ndarray) -> Any:
    if hasattr(X, "iloc"):
        return X.iloc[rows].copy(deep=True)
    return np.asarray(X)[rows].copy()


def _copy_input(X: Any) -> Any:
    return X.copy(deep=True) if hasattr(X, "iloc") else np.asarray(X).copy()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parameter_identity(model: Any) -> tuple:
    """Detect replaced / in-place-written parameters without hashing GPU weights."""
    return tuple((id(p), int(p.data_ptr()), int(p._version), tuple(p.shape), str(p.dtype))
                 for p in model.parameters())


def _native_factory(model: Any) -> Any:
    from tabfm import TabFMClassifier
    return TabFMClassifier(model=model)


def _check_defaults(estimator: Any, model: Any) -> None:
    _require(estimator.model is model, "Hierarchy must share one checkpoint model")
    for name, parameter in inspect.signature(type(estimator).__init__).parameters.items():
        if name in ("self", "model"):
            continue
        _require(parameter.default is not inspect.Parameter.empty,
                 f"Unexpected required native classifier parameter: {name}")
        _require(hasattr(estimator, name) and getattr(estimator, name) == parameter.default,
                 f"Hierarchy changed native default: {name}")
    # Assert the pinned version's essential defaults; never override them.
    for name, expected in {"n_estimators": 32, "random_state": 42,
                           "max_num_rows": None, "max_num_features": 500,
                           "batch_size": 1, "cache_context": False,
                           "enable_nnls": False, "class_shift": True,
                           "average_logits": True, "softmax_temperature": 0.9}.items():
        _require(getattr(estimator, name, object()) == expected,
                 f"Unexpected native default {name}; refusing to alter the protocol")


def _default_node_predictor(estimator: Any, X: Any, y: np.ndarray,
                            X_test: Any, node_info: dict) -> tuple[np.ndarray, dict]:
    """Default implementation; the campaign may supply its stronger native audit."""
    del node_info
    calls: list[dict] = []
    phase = ["fit"]

    def observe(_module, args, output):
        _require(phase[0] == "predict", "Default native fit unexpectedly ran a model forward")
        _require(len(args) >= 3, "Unexpected native forward signature")
        x, labels, train_sizes = args[:3]
        batch, total = int(x.shape[0]), int(x.shape[1])
        sizes = np.asarray(train_sizes.detach().cpu().numpy()).reshape(-1)
        _require(batch == 1 and len(sizes) == batch and np.all(sizes == len(y)),
                 "Native default support size/member batch changed")
        _require(total == len(y) + len(X_test) and tuple(labels.shape) == (batch, total),
                 "Native node lost support/query rows")
        _require(tuple(output.shape) == (batch, total, int(estimator.model.max_classes)),
                 "Native classification output shape changed")
        calls.append({"member_count": batch, "support_rows": len(y),
                      "test_rows": len(X_test), "output_shape": list(output.shape)})

    handle = estimator.model.register_forward_hook(observe)
    try:
        estimator.fit(X, y)
        configs = estimator.ensemble_generator_.ensemble_configs_
        generated = sum(len(group) for group in configs.values())
        _require(generated == 32, "Native node did not generate its default 32 members")
        phase[0] = "predict"
        probabilities = np.asarray(estimator.predict_proba(_copy_input(X_test)))
        count = sum(call["member_count"] for call in calls)
        _require(count == generated, "Generated and executed native members differ")
        return probabilities, {
            "actual_ensemble_count": count, "native_generated_configurations": generated,
            "native_member_forward_verified": True, "full_test_split": True,
            "support_rows": len(y), "test_rows": len(X_test), "forward_calls": calls,
            "forced_estimator_count": False, "external_support_subsampling": False,
            "external_query_chunking": False,
        }
    finally:
        handle.remove()


def _checked_probabilities(probabilities: Any, rows: int, classes: int) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    _require(probabilities.shape == (rows, classes), "Wrong hierarchy node probability shape")
    _require(np.isfinite(probabilities).all() and (probabilities >= 0).all(),
             "Non-finite or negative hierarchy probability")
    _require(np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-5),
             "Hierarchy node probabilities are not normalized")
    return probabilities


def hierarchical_predict_proba(
    model: Any, X_train: Any, y_train: Any, X_test: Any, *,
    classifier_factory: Callable | None = None,
    node_predictor: Callable | None = None,
) -> tuple[np.ndarray, dict]:
    """Return full-class probabilities and node audits using training-only hierarchy.

    ``node_predictor(estimator, X_node, y_node, X_test, node_info)`` must fit the
    estimator and return ``(probabilities, audit)`` without extra model calls.
    The callback's probability columns must follow ``estimator.classes_``.
    Every callback must prove its 32 native members covered all test rows.
    Output columns follow sorted ``np.unique(y_train)`` (also audit['classes']).
    The helper is deliberately restricted to problems exceeding the native head.
    """
    y = np.asarray(y_train)
    _require(y.ndim == 1 and len(y) == len(X_train) and len(y) > 0,
             "Hierarchy expects one training label per support row")
    _require(len(X_test) > 0 and len(X_train.shape) == len(X_test.shape) == 2
             and X_train.shape[1] == X_test.shape[1], "Invalid train/test feature dimensions")
    _require(not any(v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v))
                     for v in y), "Missing/nonfinite training labels are not allowed")
    classes, encoded = np.unique(y, return_inverse=True)
    capacity = int(model.max_classes)
    _require(capacity >= 2 and len(classes) > capacity,
             "Hierarchy is only for more classes than the native model supports")
    _require(not model.training, "Shared checkpoint must already be in eval mode")
    parameters_before = _parameter_identity(model)
    _require(parameters_before, "Shared checkpoint has no observable parameters")
    factory = classifier_factory or _native_factory
    predictor = node_predictor or _default_node_predictor
    node_audits: list[dict] = []
    tree: list[dict] = []

    def process(rows: np.ndarray, depth: int, parent: int | None) -> np.ndarray:
        node_classes, local_labels = np.unique(encoded[rows], return_inverse=True)
        leaf = len(node_classes) <= capacity
        assignments, groups = balanced_groups(len(node_classes), capacity)
        node_targets = local_labels if leaf else assignments[local_labels]
        outputs = len(node_classes) if leaf else groups
        node_id = len(tree)
        descriptor = {"node_id": node_id, "parent_node_id": parent, "depth": depth,
                      "is_leaf": leaf, "global_class_indices": node_classes.tolist(),
                      "class_count": len(node_classes), "local_class_count": outputs,
                      "support_rows": len(rows), "test_rows": len(X_test),
                      "support_row_indices_sha256": hashlib.sha256(
                          np.ascontiguousarray(rows, dtype=np.int64).tobytes()).hexdigest(),
                      "child_node_ids": []}
        if not leaf:
            descriptor["class_group_assignments"] = assignments.tolist()
        tree.append(descriptor)
        estimator = factory(model)
        _check_defaults(estimator, model)
        probabilities, audit = predictor(estimator, _row_subset(X_train, rows),
                                          node_targets.copy(), X_test, dict(descriptor))
        _require(estimator.model is model and _parameter_identity(model) == parameters_before,
                 "Hierarchy changed or replaced the shared checkpoint parameters")
        _require(not model.training, "A hierarchy node switched the checkpoint to training")
        probabilities = _checked_probabilities(probabilities, len(X_test), outputs)
        actual_classes = np.asarray(estimator.classes_)
        _require(actual_classes.shape == (outputs,) and np.array_equal(np.sort(actual_classes),
                     np.arange(outputs)), "Node probability labels are not a complete local vocabulary")
        # Explicit mapping avoids assuming any estimator's class-column ordering.
        probabilities = probabilities[:, np.argsort(actual_classes)]
        _require(isinstance(audit, dict) and audit.get("actual_ensemble_count") == 32
                 and audit.get("native_member_forward_verified") is True
                 and audit.get("full_test_split") is True
                 and audit.get("support_rows") == len(rows)
                 and audit.get("test_rows") == len(X_test),
                 "Every hierarchy node must prove default32 full-test native execution")
        node_audits.append({**descriptor, "native_audit": audit,
                            "actual_ensemble_count": 32, "native_member_forward_verified": True,
                            "full_test_split": True})
        del estimator
        result = np.zeros((len(X_test), len(classes)), dtype=probabilities.dtype)
        if leaf:
            result[:, node_classes] = probabilities
            return result
        for group in range(groups):
            child_rows = rows[node_targets == group]
            _require(len(child_rows) > 0, "Hierarchy created an empty child")
            descriptor["child_node_ids"].append(len(tree))
            child_probabilities = process(child_rows, depth + 1, node_id)
            # Identical probability-chain combination to TabICLv2.
            result += child_probabilities * probabilities[:, group:group + 1]
        return result

    result = process(np.arange(len(y), dtype=np.int64), 0, None)
    _checked_probabilities(result, len(X_test), len(classes))
    # No posthoc renormalization, clipping, class pruning, or probability padding.
    tree_digest = hashlib.sha256(json.dumps(tree, sort_keys=True, allow_nan=False).encode()).hexdigest()
    audit = {
        "protocol": PROTOCOL, "reference": dict(REFERENCE), "hierarchy_used": True,
        "tree_rule": "sorted classes; min(ceil(C / capacity), capacity) balanced contiguous groups",
        "combination": "product of native node-ensemble probabilities along each path",
        "native_tabicl_representation_reuse": False,
        "adaptation": "full native TabFM preprocessing and ensemble per node, one shared checkpoint",
        "classes": [_json_value(v) for v in classes], "native_max_classes": capacity,
        "node_count": len(tree), "tree": tree, "tree_sha256": tree_digest,
        "ensemble_audits": node_audits, "native_n_estimators_per_node": 32,
        "actual_ensemble_count": 32, "actual_ensemble_count_scope": "per hierarchy node",
        "native_member_forward_verified": True, "hierarchy_added": True,
        "actual_ensemble_count_per_node": 32, "all_nodes_native_member_forward_verified": True,
        "total_member_forwards": 32 * len(tree), "total_member_forward_count": 32 * len(tree),
        "forced_estimator_count": False,
        "shared_checkpoint_model": True, "checkpoint_parameters_unchanged": True,
        "optimizer_updates": 0, "test_labels_used": False, "full_test_split": True,
        "test_rows": len(X_test), "support_rows": len(y), "probability_shape": list(result.shape),
        "external_support_subsampling": False, "external_query_chunking": False,
        "posthoc_probability_renormalization": False,
    }
    return result, audit
