#!/usr/bin/env python3
"""Derive isolated FP32 TabSwift manifests; never submit or allocate any resource.

The old plan, source files, weights, splits, results and AMP protocols are read
and verified, never edited. Only precision and its worker/metadata are changed.
Run only after tabswift_one_v2.py is final: its exact source identity is frozen.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from eval_one import object_digest, publish_new, require, verify_file
from tabfm_prepare import identity, verify_manifest
from tabswift_bootstrap import BASE, COMMIT, REVISION, STAGE as OLD_STAGE, WEIGHT_SHA

VARIANTS = ("official16", "budget32x8")
NAME = "tabswift_fp32_standard681_20260923_v2"
STAGE = BASE / "stage" / NAME
NEW_SOURCE_NAMES = ("tabswift_prepare_fp32_v2.py", "tabswift_one_v2.py")
PRECISION_NOTE = (
    "User-authorized FP32/use_amp=False variant. The pinned native predictor has an "
    "unconditional autocast context; the isolated v2 worker disables that context "
    "at runtime and verifies FP32 parameters and module inputs/outputs. Native "
    "weights, computation formulas, preprocessing, splits and estimator counts "
    "are unchanged; no equivalence to the old AMP numerical results is claimed."
)
MANIFEST_MUTABLE = {
    "manifest_id", "name", "created_epoch", "output_root", "worker_script",
    "worker_sources", "protocol", "fp32_migration_audit",
}
PLAN_MUTABLE = {
    "plan_id", "name", "created_epoch", "output_root", "campaign_manifests",
    "source_records", "allocation", "legacy_allocation_reference",
    "existing_allocations_only", "new_allocation_submission_allowed", "fp32_migration_audit",
}


def read(path):
    return json.loads(Path(path).read_text())


def verify_plan_document(plan):
    require(plan.get("plan_id") == object_digest({k: v for k, v in plan.items() if k != "plan_id"}),
            "Old plan content identity mismatch")
    require(plan.get("protocol_variants") == list(VARIANTS)
            and plan.get("membership_count") == 1362, "Old plan scope/order mismatch")


def merge_sources(existing, additions):
    """Preserve every original pin; no replacement of an old source is allowed."""
    require(isinstance(existing, list) and existing, "Missing original source pins")
    result = copy.deepcopy(existing)
    by_path = {record["path"]: record for record in result}
    require(len(by_path) == len(result), "Duplicate original source paths")
    for record in additions:
        require(record["path"] not in by_path, "New source collides with an old frozen source")
        require(Path(record["path"]).is_absolute(), "New source identity must be absolute")
        by_path[record["path"]] = record
        result.append(copy.deepcopy(record))
    return result


def transform_manifest(old, old_identity, old_plan_identity, new_root, additions, *, created_epoch):
    """Pure transformation used by tests; caller verifies file pins before this."""
    verify_manifest(old)
    protocol = old["protocol"]
    variant = protocol["variant"]
    require(variant in VARIANTS, "Unknown original variant")
    counts = {"classification": 16, "regression": 16} if variant == "official16" else {
        "classification": 32, "regression": 8}
    require(protocol["n_estimators"] == counts, "Original estimator counts changed")
    require(protocol["strict_actual_count"] is (variant == "budget32x8"), "Original count policy changed")
    require(protocol.get("use_amp") is True and "precision" not in protocol,
            "Expected original immutable AMP protocol, not an already migrated variant")
    require((old["membership_count"], old["classification_count"], old["regression_count"])
            == (681, 457, 224), "Original scope changed")
    root = Path(new_root)
    require(root.is_absolute() and root != Path(old["output_root"]), "New output must be isolated")
    require(root.name == f"tabswift_{variant}_fp32_standard681_20260923_v2", "Unexpected new output name")
    additions_by_name = {Path(record["path"]).name: record for record in additions}
    require(set(additions_by_name) == set(NEW_SOURCE_NAMES), "Exactly the v2 worker and preparer must be pinned")
    new = copy.deepcopy(old)
    new.update(name=root.name, created_epoch=created_epoch, output_root=str(root),
               worker_script=copy.deepcopy(additions_by_name["tabswift_one_v2.py"]),
               worker_sources=merge_sources(old["worker_sources"], additions))
    new["protocol"].update(use_amp=False, precision="fp32", precision_note=PRECISION_NOTE)
    new["fp32_migration_audit"] = {
        "original_manifest": copy.deepcopy(old_identity), "original_manifest_id": old["manifest_id"],
        "original_plan": copy.deepcopy(old_plan_identity),
        "original_worker_script": copy.deepcopy(old["worker_script"]),
        "authorized_change": "FP32/use_amp=False only; separate v2 worker and output roots",
        "native_amp_guard": PRECISION_NOTE,
        "all_other_scientific_fields_unchanged": True, "old_source_pins_preserved": True,
        "old_results_reused": False, "old_results_modified": False,
        "existing_allocations_only": True, "new_allocation_submission_allowed": False,
    }
    new.pop("manifest_id")
    new["manifest_id"] = object_digest(new)
    require({k: v for k, v in new.items() if k not in MANIFEST_MUTABLE}
            == {k: v for k, v in old.items() if k not in MANIFEST_MUTABLE}, "Scientific field changed")
    require({k: v for k, v in new["protocol"].items() if k not in ("use_amp", "precision", "precision_note")}
            == {k: v for k, v in protocol.items() if k != "use_amp"}, "Nonprecision protocol field changed")
    verify_manifest(new)
    return new


def transform_plan(old, old_identity, manifest_records, additions, new_stage, *, created_epoch):
    verify_plan_document(old)
    require(len(manifest_records) == 2 and len({r["path"] for r in manifest_records}) == 2,
            "Exactly two new manifest identities required")
    require(Path(new_stage).is_absolute() and Path(new_stage) != Path(old["output_root"])
            and Path(new_stage).name == NAME, "New plan output must be isolated")
    for variant, record in zip(VARIANTS, manifest_records):
        require(Path(record["path"]).parent.name == f"tabswift_{variant}_fp32_standard681_20260923_v2",
                "New manifest order/name mismatch")
    new = copy.deepcopy(old)
    legacy_allocation = new.pop("allocation", None)
    require(legacy_allocation is not None, "Original allocation reference missing")
    new.update(name="tabswift_fp32_standard681_dual_v2", created_epoch=created_epoch,
        output_root=str(new_stage), campaign_manifests=copy.deepcopy(manifest_records),
        source_records=merge_sources(old["source_records"], additions),
        legacy_allocation_reference=legacy_allocation, existing_allocations_only=True,
        new_allocation_submission_allowed=False,
        fp32_migration_audit={"original_plan": copy.deepcopy(old_identity),
            "original_plan_id": old["plan_id"], "precision_change": PRECISION_NOTE,
            "resource_authority": "Reuse existing authorized allocations only; no sbatch, parent mutation or new allocation",
            "old_results_modified": False, "old_scientific_scope_unchanged": True})
    new.pop("plan_id")
    new["plan_id"] = object_digest(new)
    require({k: v for k, v in new.items() if k not in PLAN_MUTABLE}
            == {k: v for k, v in old.items() if k not in PLAN_MUTABLE}, "Unexpected plan field changed")
    require("allocation" not in new and new["new_allocation_submission_allowed"] is False,
            "New plan must not carry an actionable allocation contract")
    return new


def verify_frozen_bundle(old_plan_path):
    """Fully verify original source pins and scientific manifests before writes."""
    old_plan_path = Path(old_plan_path)
    plan_identity = identity(old_plan_path)
    old_plan = read(verify_file(plan_identity))
    verify_plan_document(old_plan)
    require(old_plan_path == OLD_STAGE / "plan.json", "Only the fixed original plan is allowed")
    for record in old_plan["source_records"]:
        verify_file(record, allow_empty=True)
    verify_file(old_plan["bootstrap_receipt"])
    require(len(old_plan["campaign_manifests"]) == 2, "Two original variants required")
    originals = []
    for variant, record in zip(VARIANTS, old_plan["campaign_manifests"]):
        expected = BASE / "evaluation" / f"tabswift_{variant}_standard681_20260922_v1" / "manifest.json"
        require(Path(record["path"]) == expected, "Original campaign path mismatch")
        man = read(verify_file(record)); verify_manifest(man)
        require(man["protocol"]["variant"] == variant and man["worker_python"] == old_plan["worker_python"],
                "Original variant or environment mismatch")
        require(man["official_source"]["commit"] == COMMIT and man["weight_revision"] == REVISION
                and man["weights"]["shared"]["sha256"] == WEIGHT_SHA, "Official source/weight identity changed")
        for source in man["worker_sources"] + man["official_source"]["files"]:
            verify_file(source, allow_empty=True)
        verify_file(man["worker_script"])
        verify_file(man["weights"]["shared"])
        verify_file(man["standard_split_parent_manifest"])
        verify_file(man["raw_source_manifest"])
        for kind, count in (("classification", 457), ("regression", 224)):
            data = read(verify_file(man[kind + "_manifest"])); verify_manifest(data)
            require(data["membership_count"] == len(data["rows"]) == count, "Frozen membership count changed")
            require([row.get("dataset_index", row.get("position")) for row in data["rows"]] == list(range(count)),
                    "Frozen dataset indices changed")
            require(len({row["dataset"] for row in data["rows"]}) == count, "Duplicate frozen memberships")
        originals.append((record, man))
    require(originals[0][1]["classification_manifest"] == originals[1][1]["classification_manifest"]
            and originals[0][1]["regression_manifest"] == originals[1][1]["regression_manifest"],
            "Variants must share the same frozen data")
    return plan_identity, old_plan, originals


def prepare():
    destinations = [BASE / "evaluation" / f"tabswift_{v}_fp32_standard681_20260923_v2" for v in VARIANTS]
    require(not STAGE.exists() and not STAGE.is_symlink(), "New stage already exists; never overwrite")
    for dest in destinations:
        require(not dest.exists() and not dest.is_symlink(), "New output already exists; never overwrite or mix results")
    old_identity, old_plan, originals = verify_frozen_bundle(OLD_STAGE / "plan.json")
    repo = Path(__file__).resolve().parent
    additions = [identity(repo / name) for name in NEW_SOURCE_NAMES]
    require(Path(old_plan["worker_python"]).is_file(), "Original worker environment disappeared")
    epoch = time.time()
    new_manifests = [transform_manifest(man, rec, old_identity, dest, additions, created_epoch=epoch)
                     for (rec, man), dest in zip(originals, destinations)]
    # All old identities are verified above, all transformations validated, and
    # all new destinations checked before publishing any new document.
    records = []
    for dest, man in zip(destinations, new_manifests):
        publish_new(dest / "manifest.json", man)
        records.append(identity(dest / "manifest.json"))
    plan = transform_plan(old_plan, old_identity, records, additions, STAGE, created_epoch=epoch)
    publish_new(STAGE / "plan.json", plan)
    # Compatibility only: this loader is read-only and cannot submit jobs.
    from tabswift_dispatch import load_plan
    loaded, campaigns = load_plan(STAGE / "plan.json")
    require(loaded["plan_id"] == plan["plan_id"] and len(campaigns) == 2, "New dispatcher compatibility failed")
    summary = {"plan": str(STAGE / "plan.json"), "plan_id": plan["plan_id"],
        "campaigns": [{"manifest": record, "manifest_id": man["manifest_id"],
                       "protocol": man["protocol"]} for record, man in zip(records, new_manifests)],
        "existing_allocations_only": True, "new_allocation_submission_allowed": False,
        "jobs_submitted": 0, "old_results_modified": False, "dispatcher_compatible": True}
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare()
