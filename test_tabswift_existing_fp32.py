"""Local control-flow tests; no GPU, video actions, Slurm, or remote access."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import tabswift_existing_fp32 as entry


def task(index, kind="classification"):
    return {"task_kind": kind, "dataset_index": index, "dataset": f"data-{index}",
            "row": {"work_size": index+1}, "data_manifest_id": "data"}


def result():
    return {"protocol": {"precision": "fp32"}, "constructor_settings": {"use_amp": False},
            "precision": {"native_use_amp": False, "requested_precision": "fp32",
                "parameter_dtypes": ["torch.float32"], "autocast_enabled_observations": 0,
                "forward_hook_calls": 1, "checked_float_tensor_inputs": 2,
                "checked_float_tensor_outputs": 1},
            "ensemble_audit": {"probability_validation": {"native_dtype": "float32",
                "absolute_row_sum_tolerance": 2e-5, "probabilities_renormalized": False,
                "maximum_absolute_row_sum_error": 1e-7}}}


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.campaigns = [(self.root/f"manifest{i}.json", {"output_root": str(self.root/f"variant{i}"),
                           "manifest_id": f"manifest{i}", "protocol": {"variant": name}}, [])
                          for i, name in enumerate(("official16", "budget32x8"))]
        self.plan = {"plan_id": "plan", "run_id": "test", "runtime_root": str(self.root/"variant0/existing_gpu/test")}
        self.owner = {"job": "214135", "step": "1", "rank": 0, "uuid": "abcd", "pci": "pci"}
        self.budget = SimpleNamespace(remaining=lambda: 600)
        entry.full.base.STOP = None

    def queue_patches(self, attempt):
        return (patch.object(entry.full, "attempt", side_effect=attempt),
                patch.object(entry.q, "smoke_tasks", return_value=[task(0), task(1), task(2, "regression"), task(3, "regression")]),
                patch.object(entry.swift, "pending_order", return_value=iter([(0, task(4)), (1, task(4))])),
                patch.object(entry, "checked_fp32_result", return_value={}),
                patch.object(entry.full.base, "cleanup_children", return_value={"all_owned_children_reaped": True}))

    def test_all_eight_current_smokes_precede_any_formal_work(self):
        calls = []
        def attempt(plan, path, man, item, owner, budget, root, smoke=False):
            calls.append((man["protocol"]["variant"], item["dataset_index"], smoke))
            return {"state": "complete", "attempt_output": "verified"}
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self.queue_patches(attempt): stack.enter_context(p)
            summary = entry.run_queues(self.plan, self.campaigns, self.owner, self.budget)
        self.assertEqual([c[2] for c in calls], [True]*8+[False]*2)
        self.assertEqual([c[0] for c in calls[:8]], ["official16"]*4+["budget32x8"]*4)
        gate = entry.q.read(Path(self.plan["runtime_root"])/"smoke-gate.json")
        self.assertTrue(gate["passed"])
        self.assertEqual(len(gate["smokes"]), 8)
        self.assertEqual(len(summary["attempts"]), 2)

    def test_any_smoke_failure_prevents_formal_claim_or_gate(self):
        calls = []
        def attempt(*args, smoke=False):
            calls.append(smoke)
            return {"state": "complete" if len(calls) < 7 else "operationally_deferred", "attempt_output": "verified"}
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self.queue_patches(attempt): stack.enter_context(p)
            with self.assertRaisesRegex(RuntimeError, "smoke failed"):
                entry.run_queues(self.plan, self.campaigns, self.owner, self.budget)
        self.assertEqual(calls, [True]*7)
        self.assertFalse((Path(self.plan["runtime_root"])/"smoke-gate.json").exists())

    def test_formal_deferral_is_retained_not_retried(self):
        calls = []
        def attempt(*args, smoke=False):
            calls.append(smoke)
            return {"state": "complete" if smoke else "retained_resource_deferral", "attempt_output": "verified"}
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self.queue_patches(attempt): stack.enter_context(p)
            summary = entry.run_queues(self.plan, self.campaigns, self.owner, self.budget)
        self.assertEqual(calls, [True]*8+[False])
        self.assertEqual(summary["state"], "operationally_deferred")
        self.assertIn("claim retained", summary["reason"])

    def test_existing_atomic_claim_deduplicates_without_overwrite(self):
        man = self.campaigns[0][1]
        item = task(4)
        first = dict(self.owner, reservation_token="first")
        self.assertTrue(entry.q.claim(man, item, first))
        self.assertFalse(entry.q.claim(man, item, dict(first, reservation_token="second")))
        self.assertEqual(entry.q.read(entry.q.task_path(man, "claims", item))["reservation_token"], "first")

    def test_actual_pending_order_covers_1362_and_alternates_protocols(self):
        campaigns = []
        for path, man, _ in self.campaigns:
            tasks = [task(i) for i in range(457)]+[task(i, "regression") for i in range(224)]
            campaigns.append((path, man, tasks))
        order = list(entry.swift.pending_order(campaigns))
        self.assertEqual(len(order), 1362)
        self.assertEqual([i for i, _ in order], [0, 1]*681)
        self.assertEqual(len({(v, t["task_kind"], t["dataset_index"]) for v, t in order}), 1362)

    def test_result_requires_actual_fp32_not_merely_manifest_claim(self):
        value = result()
        man = {"protocol": value["protocol"]}
        with patch.object(entry.swift, "validated_result", return_value=value):
            self.assertIs(entry.checked_fp32_result("x", man, task(0)), value)
            value["precision"]["parameter_dtypes"] = ["torch.float16"]
            with self.assertRaises(RuntimeError): entry.checked_fp32_result("x", man, task(0))

    def test_relaxed_probability_tolerance_rejected(self):
        value = result(); value["ensemble_audit"]["probability_validation"]["absolute_row_sum_tolerance"] = 0.01
        with patch.object(entry.swift, "validated_result", return_value=value), self.assertRaises(RuntimeError):
            entry.checked_fp32_result("x", {"protocol": value["protocol"]}, task(0))

    def test_memory_guards_are_raw_no_pagecache_credit(self):
        valid = {"own_tree_rss_bytes": 4*entry.GIB, "parent_memory_current": 100*entry.GIB,
                 "parent_memory_max": 2048*entry.GIB, "node_available_bytes": 1000*entry.GIB}
        entry.resource_guard(valid, self.budget)
        for key, value in (("own_tree_rss_bytes", 49*entry.GIB),
                           ("parent_memory_current", 1930*entry.GIB),
                           ("node_available_bytes", 63*entry.GIB)):
            with self.subTest(key=key), self.assertRaises(entry.full.base.OperationalDeferral):
                entry.resource_guard({**valid, key: value}, self.budget)

    def test_flock_kernel_identity_requires_write_owner_device_and_inode(self):
        device = os.makedev(8, 1)
        raw = "12: FLOCK ADVISORY WRITE 4321 08:01:9988 0 EOF\n"
        self.assertTrue(entry.flock_holder_present(raw, pid=4321, device=device, inode=9988))
        self.assertFalse(entry.flock_holder_present(raw, pid=1234, device=device, inode=9988))
        self.assertFalse(entry.flock_holder_present(raw.replace("WRITE", "READ"), pid=4321, device=device, inode=9988))
        self.assertFalse(entry.flock_holder_present(raw, pid=4321, device=device, inode=9999))

    def test_inherited_cooperative_fd_is_passed_to_every_native_worker(self):
        old = entry.ACTIVE_GATE
        entry.ACTIVE_GATE = {"release_method": "cooperative_gpu_lease"}
        calls = []
        fake = SimpleNamespace(Popen=lambda *args, **kwargs: calls.append((args, kwargs)))
        try:
            with patch.object(entry.q, "subprocess", fake), patch.object(entry, "verify_cooperative_lease", return_value={}), \
                    patch.dict(os.environ, TABSWIFT_GPU_LEASE_FD="7"):
                with entry.runtime_guards():
                    entry.q.subprocess.Popen(["worker"], pass_fds=(9,))
                    entry.q.subprocess.Popen(["worker2"])
                self.assertIs(entry.q.subprocess, fake)
            self.assertEqual(calls[0][1]["pass_fds"], (7, 9))
            self.assertEqual(calls[1][1]["pass_fds"], (7,))
        finally:
            entry.ACTIVE_GATE = old

    def test_gpu_lock_is_exclusive_but_inode_not_deleted(self):
        plan = {"parent_job_id": "214135", "gpu": {"uuid": "a"*16}}
        man = self.campaigns[0][1]
        with entry.gpu_lock(plan, man):
            with self.assertRaises(BlockingIOError):
                with entry.gpu_lock(plan, man): pass
        path = Path(man["output_root"])/"existing_gpu_locks"/("214135-"+"a"*16+".lock")
        inode = path.stat().st_ino
        with entry.gpu_lock(plan, man): self.assertEqual(path.stat().st_ino, inode)
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
