"""State-preserving all-rank preflight; deploy outside the frozen _model tree."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import socket


def require(condition, name, detail=""):
    if not condition:
        raise AssertionError(f"{name}: {detail}" if detail else name)


def model_hashes(directory):
    directory = Path(directory).resolve()
    result = {
        p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*.py"))
        if not p.name.startswith("._") and "__pycache__" not in p.parts
    }
    require(result and "tabicl.py" in result and "encoders.py" in result,
            "MODEL_TREE_MISSING", str(directory))
    return result


def verify_model_identity(source, reference_source=None):
    source = Path(source).resolve()
    actual = model_hashes(source / "src/tabicl/_model")
    supplied = os.environ.get("T25_G5SC_SOURCE_CONTRACT")
    candidates = ([Path(supplied)] if supplied else []) + [
        source / "source_contract.json", source.parent / "source_contract.json",
    ]
    manifest_path = next((p for p in candidates if p.is_file()), None)
    if reference_source is not None:
        reference_source = Path(reference_source).resolve()
        require(reference_source != source, "REFERENCE_MUST_BE_INDEPENDENT")
        expected = model_hashes(reference_source / "src/tabicl/_model")
        reference = str(reference_source)
    else:
        require(manifest_path is not None, "SOURCE_CONTRACT_MISSING", str(candidates))
        manifest = json.loads(manifest_path.read_text())
        raw = manifest.get("g5sc_model_hashes")
        require(isinstance(raw, dict) and raw, "G5SC_MODEL_HASH_MANIFEST_MISSING")
        expected = {}
        for key, value in raw.items():
            name = key.split("_model/", 1)[-1]
            require(name not in expected, "DUPLICATE_MODEL_HASH_KEY", name)
            expected[name] = value
        reference = manifest.get("g5sc_source", manifest.get("source", "frozen G5SC manifest"))
    require(actual == expected, "G5SC_MODEL_SOURCE_DRIFT", json.dumps({
        "missing": sorted(set(expected) - set(actual)),
        "extra": sorted(set(actual) - set(expected)),
        "changed": sorted(k for k in set(actual) & set(expected) if actual[k] != expected[k]),
    }))
    digest = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
    assembled_manifest = source.parent / "source.sha256"
    assembled_files = None
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text())
        expected_assembled = manifest.get("assembled_python_hashes")
        require(isinstance(expected_assembled, dict) and expected_assembled,
                "ASSEMBLED_PYTHON_HASH_MANIFEST_MISSING")
        package_root = source / "src/tabicl"
        assembled_files = {
            p.relative_to(package_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package_root.rglob("*.py"))
            if not p.name.startswith("._") and "__pycache__" not in p.parts
        }
        require(assembled_files == expected_assembled, "ASSEMBLED_TRAINER_OR_PRIOR_SOURCE_DRIFT")
    return {"source": str(source), "reference_source": str(reference),
            "source_manifest": str(manifest_path) if manifest_path else None,
            "source_contract_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path else None,
            "source_manifest_sha256": hashlib.sha256(assembled_manifest.read_bytes()).hexdigest() if assembled_manifest.is_file() else None,
            "model_identity_sha256": digest, "model_hashes": actual,
            "g5sc_model_hashes": {"_model/" + key: value for key, value in actual.items()},
            "assembled_python_identity_verified": assembled_files is not None,
            "assembled_python_file_count": len(assembled_files) if assembled_files is not None else 0,
            "assembled_python_identity_sha256": hashlib.sha256(json.dumps(assembled_files, sort_keys=True).encode()).hexdigest() if assembled_files is not None else None,
            "checked_model_file_count": len(actual)}


def tensor_state_hash(model):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def probe(model, checkpoint_dir, *, strict_resources=True):
    """Use autograd.grad, private random input, and restore RNG/training flags.

    strict_resources=False is exclusively for CPU tests. Production uses 64
    ranks, eight visible GPUs per node, and LOCAL_RANK as the current device.
    Empty visibility environment variables are allowed and never rewritten.
    """
    import numpy as np
    import torch
    from tabicl._model.tabicl import TabICL
    from tabicl._model.encoders import Encoder
    from tabicl.train._t25_regression_adapter import pinball_loss

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = next(model.parameters()).device
    model_file = Path(inspect.getfile(type(model))).resolve()
    identity = verify_model_identity(model_file.parents[3])
    require(type(model) is TabICL, "MODEL_CLASS_NOT_NATIVE_G5SC", str(type(model)))
    enc = model.icl_predictor.tf_icl
    require(type(enc) is Encoder, "ENCODER_CLASS_NOT_NATIVE_G5SC", str(type(enc)))
    require(Path(inspect.getfile(Encoder)).resolve().parent == model_file.parent, "MIXED_ENCODER_SOURCE")
    require(model.max_classes == 0 and model.num_quantiles == 999, "WRONG_REGRESSION_HEAD")
    require(len(enc.blocks) == 12 and enc.shared_depth_num_passes in (3, 4), "WRONG_LOOP_DEPTH")
    require(enc.shared_depth_enabled and enc.shared_depth_dataset_conditioned, "G5SC_GATE_DISABLED")
    require(enc.shared_depth_rho == 1.0 and not enc.recompute, "G5SC_RHO_OR_RECOMPUTE_DRIFT")
    require(enc.shared_depth_gate.shape == torch.Size([]), "GATE_NOT_SCALAR")
    require(enc.shared_depth_condition_weight.shape == (51,), "CONDITION_WEIGHT_NOT_51")
    require(torch.count_nonzero(enc.shared_depth_gate).item() == 0, "GATE_NOT_INITIAL_ZERO")
    require(torch.count_nonzero(enc.shared_depth_condition_weight).item() == 0, "CONDITION_WEIGHT_NOT_INITIAL_ZERO")
    if strict_resources:
        require(device.type == "cuda", "RUNTIME_DEVICE_NOT_CUDA")
        require(torch.cuda.device_count() == 8, "VISIBLE_DEVICE_COUNT_NOT_8")
        require(device.index == local_rank == torch.cuda.current_device(), "LOCAL_DEVICE_MISMATCH")
        require(world == 64, "WORLD_SIZE_NOT_64", str(world))

    parameters = list(model.parameters())
    original_grads = [(p.grad, p.grad.detach().clone() if p.grad is not None else None) for p in parameters]
    training_flags = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state().clone()
    # Never initialize contexts on the other seven GPUs visible to this rank.
    # Probe inputs use a private CPU generator, not torch.manual_seed().
    cuda_devices = [device.index] if device.type == "cuda" else []
    cuda_rng = {i: torch.cuda.get_rng_state(i).clone() for i in cuda_devices}
    state_before = tensor_state_hash(model)
    counts = [0] * len(enc.blocks)
    handles = []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            generator = torch.Generator(device="cpu").manual_seed(910043)
            x = torch.randn(1, 16, 6, generator=generator).to(device)
            all_y = torch.randn(1, 16, generator=generator).to(device)
            model.train()
            for index, block in enumerate(enc.blocks):
                def count_forward(_module, _args, _output, index=index):
                    counts[index] += 1
                handles.append(block.register_forward_hook(count_forward))
            with torch.enable_grad():
                prediction = model(x, all_y[:, :10])
                for handle in handles:
                    handle.remove()
                handles.clear()
                require(counts == [enc.shared_depth_num_passes] * 12, "LOOP_FORWARD_CALL_COUNT", repr(counts))
                require(prediction.shape == (1, 6, 999), "WRONG_PREDICTION_SHAPE")
                require(prediction.dtype == torch.float32 and torch.isfinite(prediction).all().item(),
                        "NONFINITE_OR_NON_FLOAT32_PREDICTION")
                loss = pinball_loss(prediction, all_y[:, 10:])
                require(loss.ndim == 0 and torch.isfinite(loss).item(), "NONFINITE_PINBALL")
                trainable = [p for p in parameters if p.requires_grad]
                gradients = torch.autograd.grad(loss, trainable, allow_unused=True)
                require(all(g is None or torch.isfinite(g).all().item() for g in gradients), "NONFINITE_PINBALL_GRADIENT")
                by_id = {id(p): g for p, g in zip(trainable, gradients)}
                gate_grad, weight_grad = by_id[id(enc.shared_depth_gate)], by_id[id(enc.shared_depth_condition_weight)]
                require(gate_grad is not None and weight_grad is not None, "GATE_DISCONNECTED")
                gate_norms = {"a_abs": gate_grad.abs().item(), "w_norm": weight_grad.norm().item()}
                loss_value = loss.item()
    finally:
        for handle in handles:
            handle.remove()
        for module, training in training_flags:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    require(torch.equal(cpu_rng, torch.get_rng_state()), "PROBE_CHANGED_CPU_RNG")
    require(all(torch.equal(state, torch.cuda.get_rng_state(i)) for i, state in cuda_rng.items()), "PROBE_CHANGED_GPU_RNG")
    require(tensor_state_hash(model) == state_before, "PROBE_CHANGED_MODEL_TENSORS")
    for parameter, (original, value) in zip(parameters, original_grads):
        require(parameter.grad is original, "PROBE_REPLACED_GRADIENT")
        require(value is None or torch.equal(parameter.grad, value), "PROBE_CHANGED_GRADIENT")
    properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    receipt = dict(identity, status="PASS", rank=rank, local_rank=local_rank, world_size=world,
                   node=socket.gethostname(), device=str(device),
                   visible_devices=torch.cuda.device_count() if device.type == "cuda" else 0,
                   current_device=torch.cuda.current_device() if device.type == "cuda" else None,
                   passes=enc.shared_depth_num_passes, base_blocks=12, block_calls=counts,
                   dtype="float32", head_quantiles=999, max_classes=0,
                   model_class=f"{type(model).__module__}.{type(model).__name__}",
                   encoder_class=f"{type(enc).__module__}.{type(enc).__name__}",
                   model_file=str(model_file), parameter_state_sha256=state_before,
                   pinball_loss=loss_value, gate_gradient_norms=gate_norms,
                   gradients_finite=True, rng_preserved=True, gradients_preserved=True,
                   model_tensors_preserved=True, strict_resources=bool(strict_resources),
                   gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   gpu_uuid=str(getattr(properties, "uuid", "")) or None,
                   cuda_visible=os.getenv("CUDA_VISIBLE_DEVICES"), rocr_visible=os.getenv("ROCR_VISIBLE_DEVICES"),
                   hip_visible=os.getenv("HIP_VISIBLE_DEVICES"))
    destination = Path(checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"preflight-rank-{rank}.json"
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    os.replace(temporary, path)
    print("T25_LOOP_PREFLIGHT " + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt
