"""Independent LimiX classification32 adapter; historical sources stay untouched.

API: predict(model_key, {X_train, y_train, X_test}, model_config) -> (probs, audit).
The frozen historical four-pipeline recipe is expanded in order eight times.
One native predictor creates all32 independently seeded preprocessing pipelines.
Native retrieval, class permutation, softmax, averaging and hierarchy are kept.
Every hierarchy node must execute every query through each of its32 pipelines.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

MEMBERS = 32
PROTOCOL = "historical-limix-classification-native32-seed0-v1"
REQUIRED_RUNTIME = (
    "inference/predictor.py", "inference/inference_method.py", "inference/preprocess.py",
    "utils/data_utils.py", "utils/retrieval_utils.py", "utils/loading.py",
    "model/layer.py", "model/transformer.py",
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    result = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)]).encode())
    result.update(value.tobytes())
    return result.hexdigest()


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value)


def validate_arrays(arrays):
    require(isinstance(arrays, dict) and set(arrays) == {"X_train", "y_train", "X_test"},
            "Adapter accepts exactly support features/labels and query features; no test labels")
    raw = {k: np.asarray(v) for k, v in arrays.items()}
    require(all(v.dtype.kind in "biuf" and np.isfinite(v).all() for v in raw.values()),
            "Historical classification cache must contain finite numeric arrays")
    xs, ys, xt = raw["X_train"], raw["y_train"], raw["X_test"]
    require(xs.ndim == xt.ndim == 2 and xs.shape[1] == xt.shape[1] > 0
            and len(xs) >= 2 and len(xt) > 0, "Invalid classification feature dimensions")
    require(ys.ndim == 1 or (ys.ndim == 2 and ys.shape[1] == 1), "Invalid support label dimensions")
    yi = ys.astype(np.int64).reshape(-1)
    require(len(yi) == len(xs) and np.array_equal(yi, ys.reshape(-1)), "Invalid integral support labels")
    classes = np.unique(yi)
    require(len(classes) >= 2 and np.array_equal(classes, np.arange(len(classes))), "Support labels must be dense0..K-1")
    xs, xt = xs.astype(np.float32, copy=True), xt.astype(np.float32, copy=True)
    require(np.isfinite(xs).all() and np.isfinite(xt).all(), "Float32 feature conversion overflow")
    return xs, yi.copy(), xt


def expand_config(base):
    require(isinstance(base, list) and len(base) == 4, "Expected historical4 pipeline recipe")
    for item in base:
        retrieval = item.get("retrieval_config", {})
        require(retrieval.get("use_retrieval") is True and retrieval.get("use_cluster") is True
                and retrieval.get("subsample_type") == "sample"
                and retrieval.get("retrieval_before_preprocessing") is False,
                "Only audited historical clustered sample-retrieval recipes supported")
        require(item.get("FeatureShuffler", {}).get("mode") == "shuffle", "Historical feature shuffle changed")
    return [copy.deepcopy(base[i % 4]) for i in range(MEMBERS)]


def pipeline_records(predictor):
    require(predictor.n_estimators == MEMBERS and len(predictor.preprocess_pipelines) == MEMBERS,
            "Native predictor did not create32 pipelines")
    require(len({id(p) for p in predictor.preprocess_pipelines}) == MEMBERS, "Repeated pipeline objects")
    require(predictor.seed == 0 and predictor.mix_precision is True
            and predictor.softmax_temperature == 0.9 and not predictor.mask_prediction
            and not predictor.inference_with_DDP, "Historical LimiX inference settings changed")
    width = predictor.preprocess_num
    require(width == 10 and len(predictor.seeds) == MEMBERS * width, "Native preprocessing seed schedule changed")
    shifts = [int(x) for x in predictor.all_shifts]
    require(len(shifts) == len(set(shifts)) == MEMBERS, "Native32 feature-shuffle offsets missing")
    schedules = [tuple(predictor.seeds[i * width:(i + 1) * width]) for i in range(MEMBERS)]
    require(len(set(schedules)) == MEMBERS, "Native pipeline seed schedules repeated")
    return [{"member_index": i, "historical_config_index": i % 4,
             "feature_shuffle_offset": shifts[i], "preprocessing_seeds": list(schedules[i])}
            for i in range(MEMBERS)]


class QueryCoverage:
    """Follow original positional cluster indices, never deduplicate feature rows."""
    def __init__(self, query):
        self.query = as_numpy(query)
        self.order = None
        self.position = 0
        self.forward_shapes = []
        self.support_rows_per_forward = []

    def set_clusters(self, mapping):
        require(self.order is None and isinstance(mapping, dict), "Unexpected repeated/missing query clustering")
        order = np.concatenate([as_numpy(v).reshape(-1) for v in mapping.values()])
        require(order.dtype.kind in "iu" and len(order) == len(self.query)
                and np.array_equal(np.sort(order), np.arange(len(self.query))),
                "Native cluster query indices do not cover each test row exactly once")
        self.order = order.astype(np.int64)

    def observe(self, full_x, eval_pos, output):
        x, out = as_numpy(full_x), as_numpy(output)
        require(self.order is not None, "Forward before audited query clustering")
        require(x.ndim == 3 and x.shape[0] == 1 and 0 < eval_pos < x.shape[1], "Unexpected retrieval forward shape")
        count = x.shape[1] - eval_pos
        stop = self.position + count
        require(stop <= len(self.query), "Extra/repeated query forward rows")
        expected = self.query[self.order[self.position:stop]]
        require(np.array_equal(x[0, eval_pos:], expected), "Retrieval model query rows/order differ from cluster mapping")
        require((out.ndim == 3 and out.shape[:2] == (1, count))
                or (out.ndim == 2 and out.shape[0] == count), "Invalid native classifier output shape")
        require(out.shape[-1] >= 2 and np.isfinite(out).all(), "Invalid native classifier output")
        self.position = stop
        self.forward_shapes.append(list(out.shape))
        self.support_rows_per_forward.append(int(eval_pos))

    def finish(self):
        require(self.order is not None and self.position == len(self.query) and self.forward_shapes,
                "Missing actual model computations for some test rows")
        return {"all_test_rows_covered": True, "actual_query_rows": len(self.query),
                "query_index_order_sha256": array_digest(self.order), "actual_model_forwards": len(self.forward_shapes),
                "observed_forward_shapes": self.forward_shapes,
                "retrieved_support_rows_per_forward": self.support_rows_per_forward,
                "minimum_contributions_per_test_row": 1, "maximum_contributions_per_test_row": 1}


def audited_node(predictor, predictor_module, inference_module, xs, ys, xt, torch):
    """Wrap a single unchanged native32 predict call and verify its aggregation."""
    from sklearn.preprocessing import MinMaxScaler
    # Exactly the numerical preprocessing in the historical inference_dataset.
    scaler = MinMaxScaler()
    support = np.asarray(scaler.fit_transform(xs), dtype=np.float32)
    query = np.asarray(scaler.transform(xt), dtype=np.float32)
    classes = len(np.unique(ys))
    require(2 <= classes <= 10, "Hierarchy node must contain2..10 classes")
    member_records = pipeline_records(predictor)
    member_probs, executed = [], []
    original_factory = predictor_module.InferenceResultWithRetrieval
    original_cluster = inference_module.cluster_test_data
    require(original_factory is inference_module.InferenceResultWithRetrieval, "Unexpected retrieval factory replacement")
    active = [None]

    def cluster(*args, **kwargs):
        result = original_cluster(*args, **kwargs)
        require(active[0] is not None and isinstance(result, tuple) and len(result) == 2,
                "Unexpected native cluster result")
        active[0].set_clusters(result[1])
        return result

    class AuditedRetrieval(original_factory):
        def inference(self, *args, **kwargs):
            index = len(executed)
            require(index < MEMBERS and self.model is predictor.model and self.sample_selection_type == "AM",
                    "Unexpected/extra retrieval member")
            require(len(args) >= 3 and kwargs.get("task_type") == "cls", "Unexpected native retrieval signature")
            require(len(args[2]) == len(query) and len(args[0]) == len(support), "Native retrieval row count changed")
            cover = QueryCoverage(args[2])
            active[0] = cover

            def forward(_model, positional, named, output):
                require(not torch.is_grad_enabled() and not positional and named.get("task_type") == "cls",
                        "Unexpected model-forward/gradient contract")
                cover.observe(named["x"], int(named["eval_pos"]), output)

            hook = self.model.register_forward_hook(forward, with_kwargs=True)
            try:
                raw = super().inference(*args, **kwargs)
            finally:
                hook.remove()
                active[0] = None
            require(torch.is_tensor(raw) and tuple(raw.shape)[0] == len(query) and raw.ndim == 2
                    and raw.shape[1] >= classes and bool(torch.isfinite(raw).all()), "Invalid retrieval logits")
            permutation = np.asarray(predictor.class_permutations[index])
            require(np.array_equal(np.sort(permutation), np.arange(classes)), "Native class permutation invalid")
            # Preserve native ordering: temperature, inverse class mapping, softmax.
            logits = raw[:, :classes].float() / predictor.softmax_temperature
            logits = logits[..., permutation]
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            member_probs.append(probabilities)
            record = dict(member_records[index], **cover.finish(), class_permutation=permutation.tolist(),
                          prediction_sha256=array_digest(as_numpy(probabilities)),
                          transformed_query_sha256=array_digest(as_numpy(args[2])))
            executed.append(record)
            return raw

    predictor_module.InferenceResultWithRetrieval = AuditedRetrieval
    inference_module.cluster_test_data = cluster
    try:
        probability = predictor.predict(support, ys, query, task_type="Classification")
    finally:
        predictor_module.InferenceResultWithRetrieval = original_factory
        inference_module.cluster_test_data = original_cluster
    require(len(executed) == len(member_probs) == MEMBERS, "Fewer than32 executed retrieval members")
    expected = torch.stack(member_probs).mean(dim=0).float().cpu().numpy()
    expected /= expected.sum(axis=1, keepdims=True)
    probability = np.asarray(probability)
    require(probability.shape == (len(query), classes) and np.isfinite(probability).all()
            and np.array_equal(probability, expected), "Native class reorder/softmax/member mean changed")
    return probability.astype(np.float64), {
        "actual_ensemble_count": MEMBERS, "actual32_verified": True, "actual_members_per_test_row": MEMBERS,
        "all_test_rows_covered": True, "test_rows": len(query), "support_rows": len(support),
        "node_classes": classes, "member_audits": executed,
        "minimum_contributions_per_test_row": MEMBERS, "maximum_contributions_per_test_row": MEMBERS,
        "distinct_realized_query_contexts": len({r["transformed_query_sha256"] for r in executed}),
        "external_preprocessing": "historical support-fit MinMaxScaler then float32",
    }


def verify_sources(config):
    root = Path(config["source_root"]).resolve(strict=True)
    sources = config.get("runtime_sources", {})
    require(isinstance(sources, dict) and sources, "Immutable runtime source hashes required")
    resolved = {}
    for key, expected in sources.items():
        path = Path(key) if Path(key).is_absolute() else root / key
        path = path.resolve(strict=True)
        require(isinstance(expected, str) and len(expected) == 64 and digest_file(path) == expected,
                f"Runtime source identity changed: {path}")
        resolved[str(path)] = expected
    for relative in REQUIRED_RUNTIME:
        require(str((root / relative).resolve(strict=True)) in resolved, f"Missing runtime pin: {relative}")
    for prefix in ("checkpoint", "config", "hierarchy_helper"):
        path = Path(config[prefix + "_path"]).resolve(strict=True)
        require(digest_file(path) == config[prefix + "_sha256"], f"Frozen {prefix} changed")
    return root, resolved


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot import frozen helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict(model_key, arrays, model_config):
    require(model_key in ("limix2m", "limix16m"), "Wrong model for LimiX32 adapter")
    require(model_config.get("n_estimators", MEMBERS) == MEMBERS and model_config.get("seed", 0) == 0,
            "LimiX classification32 preserves seed0")
    xs, ys, xt = validate_arrays(arrays)
    original_inputs = [array_digest(a) for a in (xs, ys, xt)]
    root, sources = verify_sources(model_config)
    base = json.loads(Path(model_config["config_path"]).read_text())
    configs = expand_config(base)
    sys.path.insert(0, str(root))
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one bound GPU required")
    pm = importlib.import_module("inference.predictor")
    im = importlib.import_module("inference.inference_method")
    require(Path(pm.__file__).resolve() == (root / "inference/predictor.py").resolve()
            and Path(im.__file__).resolve() == (root / "inference/inference_method.py").resolve(),
            "Imported LimiX runtime differs from pinned source root")
    hierarchy = import_file("_limix32_historical_hierarchy", model_config["hierarchy_helper_path"])
    predictor = pm.LimiXPredictor(device=torch.device("cuda"), model_path=str(model_config["checkpoint_path"]),
        inference_config=configs, seed=0)
    require(not predictor.model.training, "Native LimiX model must be in eval mode")
    predictor.model.requires_grad_(False)
    pipeline_records(predictor)
    audits = []
    def node(node_x, node_y, query_x):
        probability, audit = audited_node(predictor, pm, im, node_x, node_y, query_x, torch)
        audit["node_index"] = len(audits)
        audits.append(audit)
        return probability
    probs, tree = hierarchy.hierarchical_predict(xs, ys, xt, node)
    require(tree["nodes"] == len(audits) and all(a["actual32_verified"] and a["test_rows"] == len(xt) for a in audits),
            "Not every historical hierarchy node executed32 members/full test")
    require(probs.shape == (len(xt), len(np.unique(ys))) and np.isfinite(probs).all()
            and (probs >= 0).all() and np.allclose(probs.sum(axis=1), 1), "Invalid final hierarchy probabilities")
    require(original_inputs == [array_digest(a) for a in (xs, ys, xt)], "Adapter mutated frozen input arrays")
    require(all(p.grad is None and not p.requires_grad for p in predictor.model.parameters()), "Unexpected LimiX gradients")
    return probs, {"protocol": PROTOCOL, "model_key": model_key, "n_estimators": MEMBERS,
        "actual_ensemble_count": MEMBERS, "actual32_verified": True, "actual_members_per_test_row": MEMBERS,
        "actual_member_scope": "per hierarchy node per full test row", "ensemble_audits": audits,
        "node_count": len(audits), "hierarchy": tree, "seed": 0, "fine_tune": False,
        "full_test_split": True, "test_rows": len(xt), "test_labels_received": False,
        "historical_config_count": 4, "expanded_config_count": MEMBERS,
        "config_expansion": "four historical configs in original order repeated8, with native32 seed schedules",
        "precision": "historical native mixed precision; retrieval autocast enabled",
        "runtime_sources": sources, "checkpoint_sha256": model_config["checkpoint_sha256"],
        "adapter_source_sha256": digest_file(__file__), "probability_sha256": array_digest(probs),
        "equal_compute_claim": False}
