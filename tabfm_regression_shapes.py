"""Exact original-receipt dimensions for sidecar eligibility only.

Never modify scientific task rows, worker inputs, manifests, claims or results.
Missing/invalid original receipts are ineligible; changed pinned receipts fail
closed when the separate operational overlay is validated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tabfm_default_dispatch as frozen


def _read(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    blob = path.read_bytes()
    after = path.stat()
    frozen.require(path.is_file() and (before.st_ino, before.st_size, before.st_mtime_ns) ==
                   (after.st_ino, after.st_size, after.st_mtime_ns), "Receipt changed during read")
    identity = {"path": str(path), "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
                "sha256": hashlib.sha256(blob).hexdigest()}
    return json.loads(blob), identity


def _context(man, tasks):
    path = frozen.verify_file(man["regression_manifest"])
    data = frozen.read(path)
    frozen.require(data["manifest_id"] == frozen.digest({k: v for k, v in data.items() if k != "manifest_id"}),
                   "Regression manifest digest mismatch")
    rows = data["rows"]
    selected = {t["dataset_index"]: t for t in tasks if t["task_kind"] == "regression"}
    frozen.require(len(rows) == 224 and set(selected) == set(range(224)), "Expected exact regression224 task scope")
    for index, row in enumerate(rows):
        task = selected[index]
        frozen.require(row["dataset_index"] == index and task["row"] == row
                       and task["dataset"] == row["dataset"] and task["data_manifest_id"] == data["manifest_id"],
                       "Overlay requires unchanged original task rows")
    checkpoints = [c for c in data["checkpoints"] if c.get("step") == 22175 and c.get("kind") == "source_baseline"]
    frozen.require(len(checkpoints) == 1 and checkpoints[0].get("finetune_step") == 0,
                   "Original unfinetuned22175 checkpoint missing/ambiguous")
    return data, checkpoints[0], selected, path.parent / "results" / "step-22175"


def _shape(result, identity, data, checkpoint, task):
    expected = {"complete": True, "checkpoint_step": 22175, "task_kind": "regression",
                "manifest_id": data["manifest_id"], "dataset_index": task["dataset_index"],
                "dataset": task["dataset"], "input_fingerprint": task["row"]["input_fingerprint"]}
    frozen.require(all(result.get(k) == v for k, v in expected.items()), "Original receipt/task binding mismatch")
    frozen.require(result.get("checkpoint") == checkpoint and result.get("strict_checkpoint_load") is True
                   and result.get("checkpoint_load_weights_only") is True, "Not the exact original22175 checkpoint")
    audit = result["data_audit"]
    frozen.require(audit.get("input_fingerprint") == expected["input_fingerprint"]
                   and audit.get("support_subsampling") is False and audit.get("query_chunking") is False
                   and audit.get("test_rows_filtered") == 0 and audit.get("test_targets_masked_or_imputed") is False
                   and audit.get("support_row_order") == audit.get("test_row_order") == "unchanged official order",
                   "Receipt does not prove full original support/test rows")
    fields = {"train_rows": audit["support_rows"], "test_rows": audit["test_rows"], "features": audit["features"]}
    frozen.require(all(type(v) is int and v > 0 for v in fields.values()), "Invalid/nonpositive exact dimensions")
    return dict(fields, receipt=identity, dataset=task["dataset"],
                input_fingerprint=expected["input_fingerprint"], data_manifest_id=data["manifest_id"])


def build(man, tasks):
    """Read frozen FT/eval224 original receipts; publish nothing and mutate nothing."""
    data, checkpoint, selected, root = _context(man, tasks)
    rows, skipped = {}, []
    for index, task in sorted(selected.items()):
        path = root / f"row-{index:03d}.json"
        try:
            result, identity = _read(path)
            rows[str(index)] = _shape(result, identity, data, checkpoint, task)
        except Exception as exc:
            skipped.append({"dataset_index": index, "dataset": task["dataset"],
                            "path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
    return {"schema": 1, "purpose": "eligibility_only_original_task_rows_unchanged",
            "regression_manifest": dict(man["regression_manifest"]), "data_manifest_id": data["manifest_id"],
            "receipts_root": str(root), "rows": rows, "skipped": skipped}


def validate(overlay, man, tasks):
    """Verify every selected receipt again; return str(index)->shape audit mapping."""
    data, checkpoint, selected, root = _context(man, tasks)
    frozen.require(overlay.get("schema") == 1 and overlay.get("purpose") == "eligibility_only_original_task_rows_unchanged"
                   and overlay["regression_manifest"] == man["regression_manifest"]
                   and overlay["data_manifest_id"] == data["manifest_id"]
                   and Path(overlay["receipts_root"]).resolve() == root.resolve(), "Wrong overlay source/protocol")
    verified = {}
    for key, value in overlay["rows"].items():
        frozen.require(key.isdigit() and str(int(key)) == key and int(key) in selected, "Unexpected overlay row")
        index, identity = int(key), value["receipt"]
        frozen.require(Path(identity["path"]).resolve() == (root / f"row-{index:03d}.json").resolve(),
                       "Overlay points to another receipt")
        result, current_identity = _read(identity["path"])
        frozen.require(current_identity == identity, "Pinned shape receipt changed")
        actual = _shape(result, current_identity, data, checkpoint, selected[index])
        frozen.require(actual == value, "Overlay dimensions or binding differ from original receipt")
        verified[key] = actual
    skipped = [r["dataset_index"] for r in overlay["skipped"]]
    frozen.require(len(skipped) == len(set(skipped)) and not set(skipped) & {int(k) for k in verified}
                   and set(skipped) | {int(k) for k in verified} == set(selected), "Overlay coverage/skip audit incomplete")
    return verified
