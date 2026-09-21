"""Mitra2 classification with32 audited native members per hierarchy node.

Preserves the older dual-standard681 protocol: official classifier2 checkpoint,
seed0, BF16,8192/1024 native caps, support-centroid hierarchy and a tiny support-
only initialization validation subset. No fine-tuning or historical mutation.
API: predict(model_key, {X_train,y_train,X_test}, model_config)->(probs,audit).
"""
from __future__ import annotations

import gc
import importlib
import importlib.metadata
from pathlib import Path

import numpy as np

import mitra_class32_one as class_audit
from classification32_limix import (require, digest_file, validate_arrays, import_file)

MEMBERS = 32
PROTOCOL = "mitra2-classification457-native32-seed0-noft-v1"
LEGACY_HELPER_SHA256 = "dd820dc8c448576d78d875c8a187879af06894e705d304aa2ef4ac620c169246"
CHECKPOINT_SHA256 = "5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09"
CONFIG_SHA256 = "2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df"
REPO_ID = "autogluon/mitra-classifier-2"
REVISION = "edada0d20759c58ada8c8605c25f22f6e98ea5f0"
RUNTIME_MODULES = (
    "autogluon.tabular.models.mitra.sklearn_interface",
    "autogluon.tabular.models.mitra._internal.core.trainer_finetune",
    "autogluon.tabular.models.mitra._internal.data.dataset_finetune",
    "autogluon.tabular.models.mitra._internal.data.preprocessor",
    "autogluon.tabular.models.mitra._internal.models.tab2d",
)


def validation_indices(y):
    """Unchanged old dual-standard681 validation subset, never query rows."""
    classes = np.unique(y)
    require(2 <= len(classes) <= 10 and np.array_equal(classes, np.arange(len(classes))),
            "Mitra2 hierarchy node labels must be dense2..10 classes")
    return np.unique(np.r_[np.arange(min(8, len(y))), [np.flatnonzero(y == c)[0] for c in classes]])


def guard_trainer(original, torch):
    class GuardedTrainer32(original):
        def __init__(self, cfg, model, *args, **kwargs):
            class_audit.check_cfg(cfg)
            require(model.dim_output == 10 and not model.use_flash_attn, "Mitra2 native head/attention changed")
            require(kwargs.get("rng") is not None, "Native advancing member RNG missing")
            self.audit_rng_entry = class_audit.rng_digest(kwargs["rng"])
            model.requires_grad_(False)
            def finite_forward(_module, _inputs, output):
                require(torch.is_tensor(output) and bool(torch.isfinite(output).all()), "Nonfinite Mitra2 raw logits")
            model.register_forward_hook(finite_forward)
            super().__init__(cfg, model, *args, **kwargs)
            self.audit_optimizer_steps, self.audit_fit_calls = 0, 0
            def forbidden_step(*_args, **_kwargs):
                self.audit_optimizer_steps += 1
                raise RuntimeError("Optimizer updates forbidden in frozen Mitra2 classification32")
            self.optimizer.step = forbidden_step

        def train(self, xs, ys, xv, yv):
            class_audit.check_cfg(self.cfg)
            selected = validation_indices(ys)
            require(np.array_equal(xv, xs[selected]) and np.array_equal(yv, ys[selected]),
                    "Mitra2 initialization validation differs from historical support-only subset")
            self.audit_fit_calls += 1
            require(self.audit_fit_calls == 1, "Unexpected member refit")
            self.audit_support_rows = len(ys)
            self.audit_validation_indices = selected.tolist()
            self.audit_preprocess_rng = class_audit.rng_digest(np.random)
            with torch.no_grad():
                result = super().train(xs, ys, xv, yv)
            self.audit_feature_mirror = self.preprocessor.mirror.tolist()
            self.audit_rng_after_fit = class_audit.rng_digest(self.rng)
            require(self.audit_optimizer_steps == 0 and all(p.grad is None and not p.requires_grad
                    for p in self.model.parameters()), "Unexpected Mitra2 parameter gradients/updates")
            return result
    return GuardedTrainer32


def audited_full_probability(estimator, query):
    """Keep historical full-query native call and member-major RNG traversal.

    Pinned DatasetFinetune enumerates contiguous query chunks of1024. Observe
    each member's positional chunk coverage, including legitimate duplicate X
    rows; never infer coverage from unique feature values or cached predictions.
    """
    rows, classes = len(query), len(np.unique(estimator.y))
    require(rows > 0 and 2 <= classes <= 10, "Invalid native classification rows/classes")
    trainers = estimator.trainers
    require(len(trainers) == len({id(t) for t in trainers}) == len({id(t.model) for t in trainers}) == MEMBERS,
            "Exactly32 distinct native classifier trainers/models required")
    input_hashes = tuple(map(class_audit.array_digest, (estimator.X, estimator.y, query)))
    probabilities, calls, offsets = [None] * MEMBERS, [0] * MEMBERS, [0] * MEMBERS
    records, originals, hooks = [], [], []
    active, execution_order = [None], []
    try:
        for member, trainer in enumerate(trainers):
            class_audit.check_cfg(trainer.cfg)
            record = {"member_index": member, "forward_chunks": []}
            records.append(record)
            original = trainer.predict
            originals.append((trainer, original))
            def capture(xs, ys, xt, *, member=member, trainer=trainer, original=original, record=record):
                require(tuple(map(class_audit.array_digest, (xs, ys, xt))) == input_hashes,
                        "Native Mitra2 member received different full-query inputs")
                calls[member] += 1
                require(calls[member] == 1 and active[0] is None, "Duplicate/interleaved Mitra2 member prediction")
                active[0] = member
                execution_order.append(member)
                record["native_rng_before_predict_sha256"] = class_audit.rng_digest(trainer.rng)
                try:
                    raw = original(xs, ys, xt)
                finally:
                    active[0] = None
                logits = np.asarray(raw)
                require(logits.shape == (rows, 10) and np.isfinite(logits).all(), "Invalid native Mitra2 logits")
                require(offsets[member] == rows, "Mitra2 member omitted query rows")
                selected = logits[:, :classes]
                probabilities[member] = np.exp(selected) / np.exp(selected).sum(axis=1, keepdims=True)
                record.update(logits_shape=list(logits.shape), logits_sha256=class_audit.array_digest(logits),
                    native_rng_after_predict_sha256=class_audit.rng_digest(trainer.rng),
                    optimizer_steps=trainer.audit_optimizer_steps, actual_query_rows=rows,
                    all_test_rows_covered=True, model_forward_calls=len(record["forward_chunks"]))
                require(trainer.audit_optimizer_steps == 0, "Mitra2 optimizer updated during prediction")
                return raw  # Preserve native trainer return shape and dtype.
            def forward_check(module, inputs, output, *, member=member, record=record):
                require(active[0] == member and len(inputs) == 6 and module.dim_output == 10,
                        "Unexpected Mitra2 forward member/signature/head")
                xs, _ys, xt, _features, support_padding, query_padding = inputs
                start, count = offsets[member], min(1024, rows - offsets[member])
                require(count > 0 and tuple(xs.shape[:2]) == (1, min(8192, len(estimator.y)))
                        and tuple(xt.shape[:2]) == (1, count) and tuple(output.shape) == (1, count, 10),
                        "Native Mitra2 context/query/output shape changed")
                require(not bool(support_padding.any()) and not bool(query_padding.any()),
                        "Unexpected padded Mitra2 query/support rows")
                offsets[member] += count
                record["forward_chunks"].append({"start": start, "stop": start + count,
                    "actual_support_rows": int(xs.shape[1]), "model_output_shape": list(output.shape)})
            trainer.predict = capture
            hooks.append(trainer.model.register_forward_hook(forward_check))
        probability = estimator.predict_proba(query)  # One full native call, no external chunking/backoff.
    finally:
        for hook in hooks:
            hook.remove()
        for trainer, original in originals:
            trainer.predict = original
    require(calls == [1] * MEMBERS and offsets == [rows] * MEMBERS
            and execution_order == list(range(MEMBERS)) and all(p is not None for p in probabilities),
            "Every query row requires32 independently executed native Mitra2 members")
    expected = sum(probabilities) / MEMBERS
    require(np.asarray(probability).shape == (rows, classes) and np.isfinite(probability).all()
            and np.array_equal(probability, expected), "Native Mitra2 softmax/member arithmetic mean changed")
    for trainer in trainers:
        class_audit.check_cfg(trainer.cfg)
    return np.asarray(probability), records


class AuditedMitra2(class_audit.AuditedClassifier):
    def fit(self, xs, ys):
        selected = validation_indices(ys)
        self.native.trainers.clear()
        gc.collect()
        self.torch.cuda.empty_cache()
        self.native.fit(xs, ys, X_val=xs[selected], y_val=ys[selected])
        records = class_audit.fitted_member_records(self.native, np)
        for trainer, record in zip(self.native.trainers, records):
            require(trainer.audit_validation_indices == selected.tolist(), "Member validation subset changed")
            require(all(not p.requires_grad and p.grad is None for p in trainer.model.parameters()),
                    "Mitra2 parameters must remain frozen")
            record.update(initialization_validation_indices=selected.tolist(), all_parameters_frozen=True)
        self.current = {"node_index": len(self.ensemble_audits), "support_rows": len(ys),
            "node_classes": len(np.unique(ys)), "node_support_sha256": class_audit.array_digest(xs),
            "node_target_sha256": class_audit.array_digest(ys), "actual_ensemble_count": MEMBERS,
            "initialization_validation_source": "historical tiny support-only subset; never test labels",
            "initialization_validation_rows": len(selected), "member_audits": records,
            "successful_prediction_chunks": [], "discarded_oom_attempts": []}
        self.ensemble_audits.append(self.current)
        return self

    def predict_full(self, query):
        require(self.current is not None and "actual32_verified" not in self.current,
                "Each Mitra2 node must predict the full test once")
        probability, records = audited_full_probability(self.native, query)
        require(probability.shape == (len(query), self.current["node_classes"]), "Wrong Mitra2 node probability shape")
        self.current.update(actual32_verified=True, actual_members_per_test_row=MEMBERS,
            all_test_rows_covered=True, test_rows=len(query), minimum_contributions_per_test_row=MEMBERS,
            maximum_contributions_per_test_row=MEMBERS, full_query_native_calls=1,
            native_query_chunk_size=1024, native_member_major_query_traversal=True,
            successful_prediction_chunks=[{"start": 0, "stop": len(query), "actual_ensemble_count": MEMBERS,
                                           "member_audits": records}],
            total_model_forward_calls=sum(record["model_forward_calls"] for record in records))
        for fitted, predicted in zip(self.current["member_audits"], records):
            fitted.update(predicted)
        return probability, 1024


def load_runtime(config):
    require(config["legacy_helper_sha256"] == LEGACY_HELPER_SHA256
            and digest_file(config["legacy_helper_path"]) == LEGACY_HELPER_SHA256, "Historical Mitra2 helper changed")
    require(config["checkpoint_sha256"] == CHECKPOINT_SHA256 and config["config_sha256"] == CONFIG_SHA256,
            "Mitra2 official classifier2 weight/config identity changed")
    require(config["hierarchy_helper_sha256"] == class_audit.HIERARCHY_SHA256
            and digest_file(config["hierarchy_helper_path"]) == class_audit.HIERARCHY_SHA256,
            "Historical support-centroid hierarchy changed")
    helper = import_file("_mitra2_class32_legacy_weight_verifier", config["legacy_helper_path"])
    checkpoint = Path(config["checkpoint_path"]).resolve(strict=True)
    require(checkpoint.name == "model.safetensors" and Path(config["config_path"]).resolve(strict=True)
            == checkpoint.parent / "config.json", "Mitra2 weight directory layout changed")
    # This helper is used ONLY for file/revision verification, not its locked1 runtime.
    identity = helper.verify_weight_config("mitra2", "classification", {
        "repo_id": REPO_ID, "revision": REVISION, "local_dir": str(checkpoint.parent),
        "weights_sha256": CHECKPOINT_SHA256, "config_sha256": CONFIG_SHA256,
        "patches_path": config["patches_path"], "patches_sha256": config["patches_sha256"], "device": "cuda"})
    pins = config.get("runtime_sources", {})
    require(isinstance(pins, dict) and pins, "Native Mitra2 runtime source hashes required")
    resolved = {}
    for path, expected in pins.items():
        path = Path(path).resolve(strict=True)
        require(isinstance(expected, str) and len(expected) == 64 and digest_file(path) == expected,
                f"Mitra2 runtime file changed: {path}")
        resolved[str(path)] = expected
    versions = {name: importlib.metadata.version("autogluon." + name) for name in ("common", "core", "features", "tabular")}
    require(len(set(versions.values())) == 1 and versions["tabular"] in ("1.6.0", "1.6.1"),
            "Requires historical matching AutoGluon1.6.0/1.6.1 quartet")
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one bound GPU required")
    modules = {name: importlib.import_module(name) for name in RUNTIME_MODULES}
    for module in modules.values():
        require(str(Path(module.__file__).resolve()) in resolved, f"Unpinned imported native module: {module.__name__}")
    si, tm, tab2d = modules[RUNTIME_MODULES[0]], modules[RUNTIME_MODULES[1]], modules[RUNTIME_MODULES[-1]]
    require(si.TrainerFinetune is tm.TrainerFinetune
            and not getattr(si, "_mitra_finetune_view_installed", None)
            and not getattr(si, "_mitra_finetune_reg_ce_patched", False), "Unexpected native training/regression patch")
    patches = import_file("_mitra2_class32_official_patches", identity["patches_path"])
    patches.install_use_hf_patch()
    tab2d.FLASH_ATTN_AVAILABLE = False
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    original = si.TrainerFinetune
    si.TrainerFinetune = guard_trainer(original, torch)
    native = si.MitraClassifier(hf_model=identity["local_dir"], device="cuda", fine_tune=False,
        fine_tune_steps=0, n_estimators=MEMBERS, seed=0, verbose=False)
    hierarchy = import_file("_mitra2_class32_hierarchy", config["hierarchy_helper_path"])
    return native, hierarchy, identity, resolved, versions, torch, si, original


def predict(model_key, arrays, model_config):
    require(model_key == "mitra2", "Wrong model for Mitra2 classification32 adapter")
    require(model_config.get("n_estimators", MEMBERS) == MEMBERS and model_config.get("seed", 0) == 0,
            "Mitra2 classification32 preserves seed0")
    xs, ys, xt = validate_arrays(arrays)
    before = [class_audit.array_digest(v) for v in (xs, ys, xt)]
    native, hierarchy, identity, sources, versions, torch, si, original = load_runtime(model_config)
    estimator = AuditedMitra2(native, np, torch)
    classes = len(np.unique(ys))
    try:
        if classes > 10:
            probs, tree = hierarchy.hierarchical_predict_proba(estimator, xs, ys, xt,
                branch_factor=10, predict_fn=lambda fitted, query: fitted.predict_full(query))
            require(tree["nodes_fitted"] == len(estimator.ensemble_audits), "Missing hierarchy node audits")
        else:
            estimator.fit(xs, ys)
            probs, _ = estimator.predict_full(xt)
            tree = None
    finally:
        si.TrainerFinetune = original
    require(probs.shape == (len(xt), classes) and np.isfinite(probs).all()
            and (probs >= 0).all() and np.allclose(probs.sum(axis=1), 1), "Invalid Mitra2 probabilities")
    for node in estimator.ensemble_audits:
        require(node["actual32_verified"] and node["test_rows"] == len(xt), "Incomplete native32/full test node")
        node["actual_members_per_test_row"] = MEMBERS
    require(before == [class_audit.array_digest(v) for v in (xs, ys, xt)], "Mitra2 adapter mutated input arrays")
    return probs, {"protocol": PROTOCOL, "model_key": model_key, "n_estimators": MEMBERS,
        "actual_ensemble_count": MEMBERS, "actual32_verified": True, "actual_members_per_test_row": MEMBERS,
        "actual_member_scope": "per hierarchy node per full test row", "ensemble_audits": estimator.ensemble_audits,
        "node_count": len(estimator.ensemble_audits), "hierarchy": tree, "full_test_split": True,
        "test_rows": len(xt), "test_labels_received": False, "seed": 0, "fine_tune": False, "fine_tune_steps": 0,
        "precision": "bfloat16", "native_support_cap": 8192, "native_query_cap": 1024,
        "initialization_validation": "historical tiny support-only subset",
        "weight_identity": identity, "runtime_sources": sources, "autogluon_versions": versions,
        "adapter_source_sha256": digest_file(__file__), "probability_sha256": class_audit.array_digest(probs),
        "equal_compute_claim": False}
