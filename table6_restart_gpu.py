"""Immutable Table6 short-worker continuation with a monotonic allocation budget."""
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import time
from table6_restart_deadline import EnvironmentBudget

ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
SOURCE = ROOT / 'stage/table6_gpu_short2h_20260914_v1/short_worker.py'
EXPECTED = '9fa83abb04107c550983cfce5280bf445916022314470c0d8d577b67379c4149'


def load_worker():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError('immutable short-worker identity changed')
    spec = importlib.util.spec_from_file_location('_table6_original_short_worker', SOURCE)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    class MonotonicBudget(worker.Budget):
        def __init__(self):
            self.verified = EnvironmentBudget.from_environment()
            if self.verified.monotonic_end is None:
                raise ValueError('same-node monotonic budget is mandatory')
            super().__init__(self.verified.hard_end_epoch)
            if self.remaining() > 6901:
                raise ValueError('excessive verified remaining allocation time')

        @classmethod
        def from_environment(cls):
            return cls()

        def remaining(self):
            return self.verified.remaining()

    worker.Budget = MonotonicBudget
    return worker


if __name__ == '__main__':
    load_worker().main()
