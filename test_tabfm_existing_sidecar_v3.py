import ast
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import tabfm_existing_sidecar_v3 as v3
import tabfm_sidecar_prepare as prepare
import test_tabfm_sidecar_prepare as original_tests


class V3Tests(unittest.TestCase):
    def test_v1_v2_frozen_sources_untouched(self):
        root = Path(v3.__file__).parent
        for name, expected in {
            "tabfm_existing_sidecar.py": "f284341dc3c5e6d4071faf6c9c78442568f948151bfd72acc5607d4c44f8603b",
            "tabfm_existing_sidecar_v2.py": "aa0a3e4d0b0615057124ec7907462c6f836bc500a82e0be532db08d5d7db49f5",
            "tabfm_default_dispatch.py": "7f3c7672ff0100247edf0b150fa7cf683d5543e8ab28284105470177d20ab30b",
        }.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)

    def test_only_lane_tmpdir_integration_differs_from_v2(self):
        root = Path(v3.__file__).parent
        old = ast.parse((root / "tabfm_existing_sidecar_v2.py").read_text())
        new = ast.parse(Path(v3.__file__).read_text())
        old_functions = {n.name: ast.dump(n, include_attributes=False) for n in old.body if isinstance(n, ast.FunctionDef)}
        new_functions = {n.name: ast.dump(n, include_attributes=False) for n in new.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(old_functions.keys(), new_functions.keys())
        self.assertEqual([name for name in old_functions if old_functions[name] != new_functions[name]], ["lane"])
        source = Path(v3.__file__).read_text()
        self.assertLess(source.index("tmpdir_audit = activate("), source.index("    import torch\n"))

    def test_preparer_v3_pins_helper_and_keeps_original_model_budget(self):
        fixture = original_tests.PrepareTests("test_complete_plan_has_operational_limits")
        with patch.object(prepare, "SIDECAR", "existing196092-single-20260922-v3"), \
                patch.object(prepare, "SIDECAR_SCRIPT", "tabfm_existing_sidecar_v3.py"):
            plan = fixture.make_plan(original_tests.snapshot(1790025913), original_tests.snapshot(1790025943), 1790025950)
        self.assertEqual(plan["inherited_cpu_count"], 64)
        self.assertEqual(plan["cpu_count"], 4)
        self.assertEqual(plan["mem_gib"], 40)
        self.assertEqual(len(plan["source_records"]), 1)
        self.assertIn("only TMPDIR changes", plan["runtime_TMPDIR_override"])
        self.assertEqual(plan["plan_id"], prepare.digest({k: v for k, v in plan.items() if k != "plan_id"}))


if __name__ == "__main__":
    unittest.main()
