#!/usr/bin/env python3
"""Isolated, train-only adaptation of a native G5SC Loop3 regression model.

Manifest: {"datasets": [{"id": "name", "split": "train",
 "numeric_path": "N_train.npy", "categorical_path": "C_train.npy",
 "target_path": "y_train.npy"}]}. At least one feature path is required.
Alternatively use train_csv, target_column, and optional categorical_columns
and numeric_columns. Paths are relative to the manifest, not the work directory.
Input filenames must explicitly identify the train split. Object/pickled NPY
arrays are forbidden. CSV categoricals must be explicitly declared.

--inspect audits all train data and strictly loads weights on CPU, without a
forward pass or optimizer. Normal execution performs a tiny forward/backward
Loop3 probe (no update), then exactly 50 updates with four episodes each.
There is deliberately no resume, test-set evaluation, or checkpoint selection.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import uuid

SOURCE_STEP = 22175
UPDATES = 50
ACCUMULATION = 4
np = None
torch = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_new(path, writer):
    """Publish a complete file without ever replacing an existing filename."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic publication; EEXIST is a fatal error.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    atomic_new(path, lambda handle: handle.write(payload))


def train_filename(name):
    tokens = set(re.split(r"[^a-z0-9]+", str(name).lower()))
    require("train" in tokens, f"Input filename must explicitly identify train split: {name}")
    require(not tokens.intersection({"test", "val", "valid", "validation", "holdout"}),
            f"Non-training input forbidden: {name}")


def resolve_train_path(value, manifest_dir, extension):
    supplied = Path(value).expanduser()
    train_filename(supplied.name)
    resolved = (manifest_dir / supplied).resolve() if not supplied.is_absolute() else supplied.resolve()
    train_filename(resolved.name)
    require(resolved.is_file() and resolved.suffix.lower() == extension,
            f"Missing/incorrect input file: {resolved}")
    return resolved


def read_manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    entries = payload.get("datasets")
    require(isinstance(entries, list) and entries, "manifest.datasets must be a nonempty list")
    specs, seen = [], set()
    aliases = {
        "numeric_path": ("numeric_path", "numeric_npy", "train_features_npy"),
        "categorical_path": ("categorical_path", "categorical_npy"),
        "target_path": ("target_path", "target_npy", "train_targets_npy"),
    }
    for entry in entries:
        require(isinstance(entry, dict) and entry.get("split") == "train",
                "Every dataset must explicitly declare split='train'")
        dataset_id = entry.get("id", entry.get("dataset"))
        require(isinstance(dataset_id, str) and dataset_id.strip() and dataset_id not in seen,
                f"Invalid/duplicate dataset id: {dataset_id}")
        seen.add(dataset_id)
        spec = {"id": dataset_id, "split": "train"}
        if "train_csv" in entry:
            require(not any(key in entry for keys in aliases.values() for key in keys),
                    "Specify CSV or NPY arrays, never both")
            spec["train_csv"] = resolve_train_path(entry["train_csv"], path.parent, ".csv")
            require(isinstance(entry.get("target_column"), str), "CSV requires target_column")
            spec["target_column"] = entry["target_column"]
            for key in ("categorical_columns", "numeric_columns"):
                if key in entry:
                    require(isinstance(entry[key], list) and all(isinstance(x, str) for x in entry[key]),
                            f"{key} must be a list of column names")
                    spec[key] = entry[key]
        else:
            for canonical, names in aliases.items():
                provided = [key for key in names if key in entry]
                require(len(provided) <= 1, f"Ambiguous aliases for {canonical}")
                if provided:
                    spec[canonical] = resolve_train_path(entry[provided[0]], path.parent, ".npy")
            require("target_path" in spec and ("numeric_path" in spec or "categorical_path" in spec),
                    "NPY requires target_path and numeric_path and/or categorical_path")
        specs.append(spec)
    return specs


def load_train_data(spec):
    numeric, categorical = None, None
    if "train_csv" in spec:
        with spec["train_csv"].open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            names = reader.fieldnames or []
            require(len(names) == len(set(names)), "Duplicate CSV header")
            require(spec["target_column"] in names, "CSV target column absent")
            cat_columns = spec.get("categorical_columns", [])
            num_columns = spec.get("numeric_columns", [x for x in names
                if x != spec["target_column"] and x not in cat_columns])
            selected = num_columns + cat_columns + [spec["target_column"]]
            require(len(selected) == len(set(selected)) and set(selected).issubset(names),
                    "CSV feature/target columns overlap, repeat, or are missing")
            rows = list(reader)
        def numeric_value(value):
            return float(value) if value and value.strip().lower() not in {"nan", "na", "null", "none", "?"} else math.nan
        y = np.asarray([numeric_value(row[spec["target_column"]]) for row in rows], dtype=np.float64)
        if num_columns:
            numeric = np.asarray([[numeric_value(row[c]) for c in num_columns] for row in rows], dtype=np.float64)
        if cat_columns:
            categorical = np.asarray([[row[c] or "__MISSING__" for c in cat_columns] for row in rows], dtype=str)
    else:
        y = np.load(spec["target_path"], allow_pickle=False)
        require(y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1), "Target must be [N] or [N,1]")
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if "numeric_path" in spec:
            numeric = np.asarray(np.load(spec["numeric_path"], allow_pickle=False), dtype=np.float64)
        if "categorical_path" in spec:
            categorical = np.load(spec["categorical_path"], allow_pickle=False)
            require(categorical.dtype.kind in {"U", "S", "i", "u", "b"},
                    "Categorical NPY must contain non-object strings or integers")
            categorical = categorical.astype(str)
    require(len(y) >= 3 and np.isfinite(y).all(), f"{spec['id']}: need >=3 rows and finite labels")
    for name, values in (("numeric", numeric), ("categorical", categorical)):
        if values is not None:
            require(values.ndim == 2 and values.shape[0] == len(y), f"{spec['id']}: bad {name} shape")
    if numeric is not None:
        # Missing numerical values are imputed from context; infinities are missing too.
        numeric = np.where(np.isfinite(numeric), numeric, np.nan)
    features = sum(x.shape[1] for x in (numeric, categorical) if x is not None)
    require(features > 0, f"{spec['id']}: no features")
    return numeric, categorical, y


def audit_data(specs):
    result = []
    for spec in specs:
        numeric, categorical, y = load_train_data(spec)
        paths = [value for value in spec.values() if isinstance(value, Path)]
        result.append({"id": spec["id"], "split": "train", "rows": len(y),
            "numeric_features": 0 if numeric is None else numeric.shape[1],
            "categorical_features": 0 if categorical is None else categorical.shape[1],
            "nonfinite_numeric_count": 0 if numeric is None else int(np.isnan(numeric).sum()),
            "target_constant": bool(np.ptp(y) == 0),
            "inputs": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in paths]})
    return result


class TrainCache:
    def __init__(self, specs, audits, limit=2):
        self.specs, self.limit, self.cache = specs, limit, OrderedDict()
        self.identities = {item["path"]: (item["bytes"], item["sha256"])
                           for audit in audits for item in audit["inputs"]}

    def get(self, index):
        if index not in self.cache:
            for path in (value for value in self.specs[index].values() if isinstance(value, Path)):
                expected_size, expected_sha = self.identities[str(path)]
                require(path.stat().st_size == expected_size and sha256_file(path) == expected_sha,
                        f"Training input changed after audit: {path}")
            self.cache[index] = load_train_data(self.specs[index])
        self.cache.move_to_end(index)
        while len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return self.cache[index]


def make_episode(data, rng, support_max=256, query_max=64, feature_limit=None):
    numeric, categorical, y = data
    total = min(len(y), support_max + query_max)
    query = min(query_max, max(1, total // 5))
    support = min(support_max, total - query)
    indices = rng.choice(len(y), size=support + query, replace=False)
    require(support >= 2 and query >= 1, "Episode requires >=2 context rows and >=1 query row")
    matrices, numeric_stats, categories, unknown = [], [], [], 0
    if numeric is not None and numeric.shape[1]:
        selected = numeric[indices]
        context = selected[:support]
        medians = np.asarray([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
                              for col in context.T], dtype=np.float64)
        filled = np.where(np.isfinite(selected), selected, medians[None, :])
        means = filled[:support].mean(axis=0)
        scales = filled[:support].std(axis=0)
        scales = np.where(scales > 1e-8, scales, 1.0)
        matrices.append((filled - means) / scales)
        numeric_stats = {"imputation": "context_median", "scaling": "context_mean_std_ddof0",
                         "all_missing_context_columns": int((~np.isfinite(context).any(axis=0)).sum())}
    if categorical is not None and categorical.shape[1]:
        selected = categorical[indices]
        encoded = np.empty(selected.shape, dtype=np.float64)
        for col in range(selected.shape[1]):
            vocabulary = {value: idx for idx, value in enumerate(sorted(set(selected[:support, col])))}
            encoded[:, col] = [vocabulary.get(value, -1) for value in selected[:, col]]
            categories.append(len(vocabulary))
        unknown = int((encoded[support:] == -1).sum())
        matrices.append(encoded)
    X = np.concatenate(matrices, axis=1)
    if feature_limit is not None:
        X = X[:, :feature_limit]
    labels = y[indices]
    mean, std = float(labels[:support].mean()), float(labels[:support].std())
    scale = std if std > 1e-8 else 1.0
    labels = (labels - mean) / scale
    X, labels = X.astype(np.float32), labels.astype(np.float32)
    require(np.isfinite(X).all() and np.isfinite(labels).all(), "Nonfinite episode after context-only preprocessing")
    record = {"support_rows": support, "query_rows": query, "features": X.shape[1],
        "row_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "context_only": True, "numeric": numeric_stats, "categorical_cardinalities": categories,
        "query_unknown_categories": unknown, "target_context_mean": mean,
        "target_context_std_ddof0": std, "target_scale": scale,
        "target_zero_variance_fallback": bool(std <= 1e-8)}
    return X, labels[:support], labels[support:], record


def load_model(source, checkpoint):
    source = Path(source).resolve()
    candidates = [source, source / "src", source / "source" / "src"]
    source_root = next((p for p in candidates if (p / "tabicl" / "_model" / "tabicl.py").is_file()), None)
    require(source_root is not None, "--source must contain the audited native tabicl source")
    sys.path.insert(0, str(source_root))
    module = importlib.import_module("tabicl._model.tabicl")
    require(Path(module.__file__).resolve() == source_root / "tabicl" / "_model" / "tabicl.py",
            "Imported tabicl does not match requested --source")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(isinstance(checkpoint_data, dict), "Expected checkpoint dictionary")
    config = checkpoint_data["config"]
    for key, expected in {"max_classes": 0, "num_quantiles": 999,
        "shared_depth_icl_enabled": True, "shared_depth_icl_dataset_conditioned": True,
        "shared_depth_icl_num_passes": 3, "shared_depth_icl_rho": 1.0,
        "bias_free_ln": False, "icl_num_blocks": 12}.items():
        require(config.get(key) == expected, f"Checkpoint config mismatch: {key} must equal {expected}")
    require(checkpoint_data.get("curr_step") == SOURCE_STEP, "Source checkpoint must be step 22175")
    model = module.TabICL(**config).float()
    model.load_state_dict(checkpoint_data["state_dict"], strict=True)
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite source parameters")
    stack = model.icl_predictor.tf_icl
    require(len(stack.blocks) == 12 and stack.shared_depth_num_passes == 3, "Realized model is not native Loop3")
    require(stack.shared_depth_condition_weight.numel() == 51, "Expected 51-dimensional native gate")
    source_hashes = {str(p.relative_to(source_root)): sha256_file(p)
                     for p in sorted((source_root / "tabicl" / "_model").rglob("*.py"))}
    provenance = {"source_root": str(source_root), "source_model_sha256": source_hashes,
        "source_training_contract": checkpoint_data.get("training_contract"),
        "strict_state_load": True, "pickle_load_weights_only": True}
    del checkpoint_data
    return model, config, provenance


def episode_loss(model, episode, device):
    X, y_context, y_query, record = episode
    X = torch.from_numpy(X).unsqueeze(0).to(device)
    y_context = torch.from_numpy(y_context).unsqueeze(0).to(device)
    y_query = torch.from_numpy(y_query).unsqueeze(0).to(device)
    prediction = model(X, y_context)
    require(tuple(prediction.shape) == (1, record["query_rows"], 999), "Wrong regression output shape")
    require(bool(torch.isfinite(prediction).all()), "Nonfinite model output")
    alpha = torch.arange(1, 1000, device=device, dtype=torch.float32).view(1, 1, -1) / 1000
    error = y_query.unsqueeze(-1) - prediction.float()
    loss = torch.maximum(alpha * error, (alpha - 1) * error).mean()
    require(bool(torch.isfinite(loss)), "Nonfinite pinball loss")
    return loss


def assert_finite_gradients(model):
    seen = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            require(bool(torch.isfinite(parameter.grad).all()), f"Nonfinite gradient: {name}")
            seen += 1
    require(seen > 0, "Backward produced no gradients")


def gate_state(model):
    stack = model.icl_predictor.tf_icl
    return {"a": float(stack.shared_depth_gate.detach().cpu()),
            "w": stack.shared_depth_condition_weight.detach().cpu().tolist()}


def probe(model, data, device, seed):
    rng = np.random.default_rng(seed)
    tiny = make_episode(data, rng, support_max=8, query_max=4, feature_limit=12)
    counts = [0] * 12
    hooks = []
    for index, block in enumerate(model.icl_predictor.tf_icl.blocks):
        def hook(_module, _args, _output, index=index):
            counts[index] += 1
        hooks.append(block.register_forward_hook(hook))
    before = gate_state(model)
    model.zero_grad(set_to_none=True)
    try:
        loss = episode_loss(model, tiny, device)
        forward_counts = list(counts)
        require(forward_counts == [3] * 12, f"Expected three actual forward passes: {forward_counts}")
        loss.backward()
        assert_finite_gradients(model)
        stack = model.icl_predictor.tf_icl
        require(stack.shared_depth_gate.grad is not None and stack.shared_depth_condition_weight.grad is not None,
                "Native gate disconnected from training graph")
        result = {"forward_block_calls": forward_counts, "forward_and_backward_block_calls": counts,
            "finite_loss": float(loss.detach().cpu()), "finite_gradients": True,
            "optimizer_steps": 0, "episode": tiny[3],
            "gate_gradient": float(stack.shared_depth_gate.grad.detach().cpu()),
            "condition_gradient_norm": float(stack.shared_depth_condition_weight.grad.norm().detach().cpu())}
        require(before == gate_state(model), "Probe unexpectedly mutated gate weights")
        return result
    finally:
        for handle in hooks:
            handle.remove()
        model.zero_grad(set_to_none=True)


def rng_state(rng):
    legacy = np.random.get_state()
    return {"python": random.getstate(), "numpy_generator": rng.bit_generator.state,
            "numpy_legacy": (legacy[0], legacy[1].tolist(), legacy[2], legacy[3], legacy[4]),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def memory_record(device):
    if device.type != "cuda":
        return {"device": str(device), "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    torch.cuda.synchronize(device)
    return {"device": str(device), "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("source", "checkpoint", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--inspect", action="store_true", help="CPU audit and strict load only; no updates")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--support-max", type=int, default=128, help="Context rows per episode, bounded by 256")
    parser.add_argument("--query-max", type=int, default=32, help="Query rows per episode, bounded by 64")
    args = parser.parse_args(argv)
    require(math.isfinite(args.learning_rate) and 0 < args.learning_rate <= 1e-5, "Require a low positive LR <=1e-5")
    require(math.isfinite(args.weight_decay) and 0 <= args.weight_decay <= 1, "Invalid weight decay")
    require(2 <= args.support_max <= 256 and 1 <= args.query_max <= 64, "Episode row limits exceed contract")
    require(args.checkpoint.is_file(), "Checkpoint does not exist")
    require(not args.output.exists(), "Output directory already exists; refusing overwrite/resume")
    specs = read_manifest(args.manifest)
    global np, torch
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    started = time.monotonic()
    device = torch.device("cpu" if args.inspect else args.device)
    gpu_identity = None
    if not args.inspect:
        require(device.type == "cuda" and torch.cuda.is_available(), "Training requires an allocated CUDA/ROCm GPU")
        require(torch.cuda.device_count() == 1, "Training must expose exactly one GPU through the runtime")
        require(device.index in (None, 0), "The single allocated GPU must be cuda:0")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        gpu_identity = {"runtime_device_count": torch.cuda.device_count(), "name": torch.cuda.get_device_name(0),
            "hip_version": torch.version.hip, "cuda_version": torch.version.cuda,
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES")}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.inspect:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(args.seed)
    audits = audit_data(specs)
    model, config, provenance = load_model(args.source, args.checkpoint)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    contract = {"schema_version": 1, "source_step": SOURCE_STEP, "optimizer_updates": UPDATES,
        "shared_model": True, "episodes_per_update": ACCUMULATION, "microbatch_size": 1,
        "max_support": args.support_max, "max_query": args.query_max, "all_features_used_for_training": True,
        "dataset_sampling": "uniform balanced shuffled cycles", "optimizer": "fresh AdamW",
        "learning_rate": args.learning_rate, "scheduler": "constant", "weight_decay": args.weight_decay,
        "gradient_clip_norm": 1.0, "amp": False, "parameter_dtype": "float32", "tf32": False,
        "seed": args.seed, "target_normalization": "context mean/std ddof0; tiny std fallback1; no clipping",
        "feature_preprocessing": "context-only numeric median/mean/std; context category vocabulary, unknown=-1",
        "loss": "mean unsorted 999-quantile pinball alpha=0.001,...,0.999",
        "training_split": "official train only", "validation_or_test_loaded": False,
        "checkpoint_source": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest_source": str(args.manifest.resolve()), "manifest_sha256": sha256_file(args.manifest),
        "model_config": config, "datasets": audits, "inspect_only": args.inspect,
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "gpu_identity": gpu_identity, "parameter_bytes": parameter_bytes,
        "estimated_checkpoint_storage_bytes": int(parameter_bytes * 3 * UPDATES * 1.15), **provenance}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.inspect:
        required_space = int(parameter_bytes * 3 * (UPDATES + 1) * 1.15) + (1 << 30)
        require(shutil.disk_usage(args.output.parent).free >= required_space,
                f"Insufficient free space for 50 optimizer-bearing checkpoints (require {required_space} bytes)")
    args.output.mkdir(exist_ok=False)
    save_json(args.output / "contract.json", contract)
    if args.inspect:
        save_json(args.output / "inspection.json", {"complete": True, "strict_load": True,
            "datasets": len(audits), "optimizer_updates": 0, "elapsed_seconds": time.monotonic() - started})
        print(json.dumps({"event": "inspection_complete", "output": str(args.output), "datasets": len(audits)}), flush=True)
        return
    model.to(device).train()
    cache = TrainCache(specs, audits)
    probe_started = time.monotonic()
    probe_result = probe(model, cache.get(0), device, args.seed + 1)
    probe_result["elapsed_seconds"] = time.monotonic() - probe_started
    probe_result["memory"] = memory_record(device)
    save_json(args.output / "probe.json", probe_result)
    print(json.dumps({"event": "probe_pass", **probe_result}, allow_nan=False), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    initial_gate = gate_state(model)
    deck, deck_position = [], 0
    training_started = time.monotonic()
    event_path = args.output / "events.jsonl"
    with event_path.open("x", encoding="utf-8") as events:
        for update in range(1, UPDATES + 1):
            update_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            episode_records, losses = [], []
            before = gate_state(model)
            for _ in range(ACCUMULATION):
                if deck_position >= len(deck):
                    deck = [int(x) for x in rng.permutation(len(specs))]
                    deck_position = 0
                index = deck[deck_position]
                deck_position += 1
                episode = make_episode(cache.get(index), rng, support_max=args.support_max, query_max=args.query_max)
                loss = episode_loss(model, episode, device)
                (loss / ACCUMULATION).backward()
                losses.append(float(loss.detach().cpu()))
                episode_records.append({"dataset_id": specs[index]["id"], **episode[3]})
                del loss, episode
            assert_finite_gradients(model)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            stack = model.icl_predictor.tf_icl
            require(stack.shared_depth_gate.grad is not None and stack.shared_depth_condition_weight.grad is not None,
                    "Missing native gate gradients")
            gate_gradient = float(stack.shared_depth_gate.grad.detach().cpu())
            w_gradient_norm = float(stack.shared_depth_condition_weight.grad.norm().detach().cpu())
            optimizer.step()  # Exactly one successful optimizer call per saved update.
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite updated parameters")
            after = gate_state(model)
            save_started = time.monotonic()
            checkpoint_path = args.output / f"step-{SOURCE_STEP + update}.ckpt"
            saved = {"config": config,
                "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                "curr_step": SOURCE_STEP + update, "source_step": SOURCE_STEP, "finetune_step": update,
                "optimizer_state": optimizer.state_dict(), "scheduler_state": None, "rng_state": rng_state(rng),
                "dataset_sampler_state": {"deck": deck, "position": deck_position},
                "training_contract": {"kind": "real_train_only_finetune", "contract_sha256": sha256_file(args.output / "contract.json"),
                    "optimizer": "AdamW", "lr": args.learning_rate, "source_step": SOURCE_STEP,
                    "source_checkpoint_sha256": contract["checkpoint_sha256"]}}
            atomic_new(checkpoint_path, lambda handle: torch.save(saved, handle))
            del saved
            event = {"event": "optimizer_update_complete", "finetune_step": update,
                "curr_step": SOURCE_STEP + update, "optimizer_updates_completed": update,
                "loss": sum(losses) / ACCUMULATION, "episode_losses": losses, "finite_loss_and_gradients": True,
                "gradient_norm_preclip": float(grad_norm.detach().cpu()), "gate_gradient_after_clip": gate_gradient,
                "condition_gradient_norm_after_clip": w_gradient_norm, "gate_before": before, "gate_after": after,
                "gate_a_delta": after["a"] - before["a"],
                "gate_w_delta_l2": math.sqrt(sum((a - b) ** 2 for a, b in zip(after["w"], before["w"]))),
                "episodes": episode_records, "learning_rate": optimizer.param_groups[0]["lr"],
                "checkpoint": str(checkpoint_path.resolve()), "checkpoint_bytes": checkpoint_path.stat().st_size,
                "save_seconds": time.monotonic() - save_started, "elapsed_seconds": time.monotonic() - update_started,
                "memory": memory_record(device)}
            encoded = json.dumps(event, sort_keys=True, allow_nan=False)
            events.write(encoded + "\n")
            events.flush()
            os.fsync(events.fileno())
            print(encoded, flush=True)
    checkpoints = sorted(args.output.glob("step-*.ckpt"))
    require(len(checkpoints) == UPDATES, "Not all 50 checkpoints were saved")
    completion = {"complete": True, "source_step": SOURCE_STEP, "optimizer_updates": UPDATES,
        "episodes": UPDATES * ACCUMULATION, "checkpoint_count": len(checkpoints),
        "final_step": SOURCE_STEP + UPDATES, "initial_gate": initial_gate, "final_gate": gate_state(model),
        "training_seconds": time.monotonic() - training_started, "elapsed_seconds": time.monotonic() - started,
        "memory": memory_record(device), "checkpoint_sizes": {p.name: p.stat().st_size for p in checkpoints}}
    save_json(args.output / "complete.json", completion)
    print(json.dumps({"event": "complete", **completion}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
