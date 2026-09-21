"""Pinned runtime-only TMPDIR routing for TabFM; never alter model settings.

Call activate once per actual GPU-lane process, before importing Torch. The
private directory is deliberately retained until node-local cleanup: workers
must never lose temporary files while inference or ROCm initialization is live.
"""
from __future__ import annotations
import os
from pathlib import Path
import stat
import tempfile
import time
import uuid

import tabfm_default_dispatch as frozen

_TMP_RUNTIME_ACTIVE = None


def activate(man, plan, audit_dir):
    """Install a private short TMPDIR and return its immutable JSON audit record.

The caller validates its runtime plan and supplies an audit directory inside
the ORIGINAL output root. The same plan must pin this helper in source_records.
Only TMPDIR changes in the frozen worker environment; every other key is checked.
"""
    global _TMP_RUNTIME_ACTIVE
    frozen.require(_TMP_RUNTIME_ACTIVE is None, "TMPDIR override already installed in this process")
    frozen.require(isinstance(plan.get("plan_id"), str) and plan["plan_id"], "Runtime plan identity missing")
    helper = Path(__file__).resolve()
    records = [record for record in plan.get("source_records", [])
               if Path(record["path"]).resolve() == helper]
    frozen.require(len(records) == 1 and frozen.verify_file(records[0]) == helper,
                   "Runtime plan must uniquely pin the actual short-TMPDIR helper")
    audit_dir = frozen.checked_path(man, Path(audit_dir))
    original_lane_tmpdir = os.environ.get("TMPDIR")
    original_cached_tmpdir = tempfile.tempdir
    directory = Path(tempfile.mkdtemp(prefix="tfm-", dir="/tmp"))
    info = directory.stat()
    frozen.require(directory.parent == Path("/tmp") and directory.name.startswith("tfm-")
                   and not directory.is_symlink() and stat.S_IMODE(info.st_mode) == 0o700
                   and info.st_uid == os.getuid(), "Short TMPDIR is not a private node-local owned directory")
    record = {"manifest_id": man["manifest_id"], "plan_id": plan["plan_id"],
              "helper_source": records[0], "original_lane_TMPDIR": original_lane_tmpdir,
              "original_tempfile_cached_directory": original_cached_tmpdir,
              "new_TMPDIR": str(directory), "resolved_directory": str(directory.resolve()),
              "directory_mode": "0700", "directory_owner_uid": info.st_uid,
              "pid": os.getpid(), "epoch": time.time(), "modified_environment_keys": ["TMPDIR"],
              "model_settings_changed": False, "frozen_source_files_changed": False,
              "cleanup_policy": "retained private node-local directory; no deletion while workers may use it"}
    frozen.atomic(man, audit_dir / "runtime-tmp.json", record)
    original_worker_environment = frozen.worker_environment

    def worker_environment(worker_man, owner):
        native = original_worker_environment(worker_man, owner)
        env = dict(native)
        env["TMPDIR"] = str(directory)
        frozen.require({k: v for k, v in env.items() if k != "TMPDIR"} ==
                       {k: v for k, v in native.items() if k != "TMPDIR"},
                       "Only child TMPDIR may differ from frozen worker environment")
        frozen.atomic(man, audit_dir / ("child-env-" + uuid.uuid4().hex + ".json"),
                      {"manifest_id": man["manifest_id"], "plan_id": plan["plan_id"],
                       "helper_source": records[0], "owner": owner, "epoch": time.time(),
                       "original_worker_TMPDIR": native.get("TMPDIR"), "new_TMPDIR": str(directory),
                       "modified_environment_keys": ["TMPDIR"], "all_other_environment_keys_identical": True})
        return env

    os.environ["TMPDIR"] = str(directory)
    tempfile.tempdir = None  # Do not let Python reuse a previously cached deep path.
    frozen.worker_environment = worker_environment
    _TMP_RUNTIME_ACTIVE = record
    return record
