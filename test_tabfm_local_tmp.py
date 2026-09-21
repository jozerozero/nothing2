"""CPU tests of TMPDIR-only runtime routing and source/audit integrity."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch

import tabfm_local_tmp as helper


class ShortTmpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.man = {"output_root": str(self.root), "manifest_id": "frozen-original"}
        self.auditdir = self.root / "runtime-audit"
        self.identity = {"path": str(Path(helper.__file__).resolve()),
                         "sha256": hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest()}
        self.plan = {"plan_id": "runtime-plan", "source_records": [self.identity]}
        self.native = {"TMPDIR": "/deep/campaign/tmp/parent/rank", "OMP_NUM_THREADS": "4",
                       "ROCR_VISIBLE_DEVICES": "GPU-abc", "HF_HUB_OFFLINE": "1", "unchanged": "test"}
        for patcher in (patch.object(helper, "_TMP_RUNTIME_ACTIVE", None),
                        patch.object(helper.frozen, "worker_environment", return_value=dict(self.native)),
                        patch.dict(os.environ, {"TMPDIR": "/deep/campaign/lane"}),
                        patch.object(tempfile, "tempdir", "/previous/cache")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def activate(self):
        result = helper.activate(self.man, self.plan, self.auditdir)
        directory = Path(result["new_TMPDIR"])
        self.addCleanup(shutil.rmtree, directory)
        return result, directory

    def test_private_short_node_local_directory_and_lane_audit(self):
        audit, directory = self.activate()
        self.assertEqual(directory.parent, Path("/tmp"))
        self.assertTrue(directory.name.startswith("tfm-"))
        self.assertLess(len(str(directory)), 32)
        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
        self.assertEqual(directory.stat().st_uid, os.getuid())
        self.assertEqual(os.environ["TMPDIR"], str(directory))
        self.assertIsNone(tempfile.tempdir)
        self.assertEqual(audit["original_lane_TMPDIR"], "/deep/campaign/lane")
        self.assertEqual(audit["helper_source"], self.identity)
        self.assertEqual(json.loads((self.auditdir / "runtime-tmp.json").read_text()), audit)

    def test_child_environment_changes_only_tmpdir_and_each_call_audited(self):
        _, directory = self.activate()
        for index in range(2):
            env = helper.frozen.worker_environment(self.man, {"job": "parent", "rank": index})
            self.assertEqual({k: v for k, v in env.items() if k != "TMPDIR"},
                             {k: v for k, v in self.native.items() if k != "TMPDIR"})
            self.assertEqual(env["TMPDIR"], str(directory))
        records = list(self.auditdir.glob("child-env-*.json"))
        self.assertEqual(len(records), 2)
        for record in records:
            value = json.loads(record.read_text())
            self.assertEqual(value["original_worker_TMPDIR"], self.native["TMPDIR"])
            self.assertEqual(value["modified_environment_keys"], ["TMPDIR"])
            self.assertTrue(value["all_other_environment_keys_identical"])

    def test_no_source_pin_no_activation(self):
        self.plan["source_records"] = []
        with self.assertRaisesRegex(RuntimeError, "uniquely pin"):
            helper.activate(self.man, self.plan, self.auditdir)
        self.assertEqual(os.environ["TMPDIR"], "/deep/campaign/lane")

    def test_duplicate_source_pin_rejected(self):
        self.plan["source_records"] *= 2
        with self.assertRaisesRegex(RuntimeError, "uniquely pin"):
            helper.activate(self.man, self.plan, self.auditdir)

    def test_changed_helper_hash_rejected(self):
        self.identity["sha256"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "Frozen SHA"):
            helper.activate(self.man, self.plan, self.auditdir)

    def test_no_double_install_or_audit_overwrite(self):
        self.activate()
        before = (self.auditdir / "runtime-tmp.json").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            helper.activate(self.man, self.plan, self.auditdir)
        self.assertEqual((self.auditdir / "runtime-tmp.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
