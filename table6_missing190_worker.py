#!/usr/bin/env python3
"""GPU-reserved, mixed-device missing190 continuation; original outputs untouched.

The immutable short coordinator supplies pair flock, 100-attempt persistent
Optuna, same-RUNNING-trial restart, 15 seeds and owned-process cleanup. Only
this fresh process is rebound to the new plan, fit child and output namespace.
All nine methods must pass real sampled-config validation-only smoke fits
before any formal pair. Each rank reserves a GPU, but the three tree methods
explicitly execute on CPU with GPU visibility removed in the fresh fit child.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from table6_restart_deadline import EnvironmentBudget

ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
R19 = ROOT / 'stage/table6_remaining19_standard_hpo_bg8_20260912_v1'
SHORT = ROOT / 'stage/table6_gpu_short2h_20260914_v1/short_worker.py'
PINNED = {'short_worker.py': '9fa83abb04107c550983cfce5280bf445916022314470c0d8d577b67379c4149',
          'worker.py': '3416759aa2caa3b38a29416dc61c14025711c18e6e0e6b49ee22220530e14263',
          'common.py': '8d03dc74f623ec5e00694da5fc6f9e54897c56036ee473177fdfa3d0fabecad0'}
PROTOCOL = 'standard457_missing190_fresh_hpo100_seed15_v1'
LEGACY_PROTOCOL = 'standard681_remaining19_hpo100_seed15_v1'
CONTRACT = 'missing190_gpu_short2h_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask_cpu_visibility():
    os.environ.update(CPU_ONLY='1', CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='-1',
                      ROCR_VISIBLE_DEVICES='-1', GPU_DEVICE_ORDINAL='-1')


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def contained(path, root):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            'write/read escaped the independent missing190 output: ' + str(path))
    return path


def monotonic_budget(short):
    class MonotonicBudget(short.Budget):
        def __init__(self):
            self.verified = EnvironmentBudget.from_environment()
            require(self.verified.monotonic_end is not None, 'same-node monotonic deadline required')
            super().__init__(self.verified.hard_end_epoch)
            require(self.remaining() <= 6901, 'excessive verified two-hour remaining budget')

        def remaining(self):
            return self.verified.remaining()

    return MonotonicBudget()


def rewrite_child(argv, *, legacy_fit, new_fit, plan_path, smoke):
    argv = list(argv)
    require(len(argv) == 7 and argv[:2] == [sys.executable, '-B'] and
            Path(argv[2]).resolve() == legacy_fit.resolve() and
            argv[3] == '--request' and argv[5] == '--response',
            'unexpected immutable fit child command; refuse launch')
    argv[2] = str(new_fit)
    argv += ['--plan', str(plan_path)]
    if smoke:
        argv.append('--smoke')
    return argv


class Runtime:
    def __init__(self, plan_path, stage=R19, short_source=SHORT):
        require(__debug__, 'python -O would disable inherited scientific validators')
        require('common' not in sys.modules and 'data' not in sys.modules and 'fit' not in sys.modules,
                'worker requires a fresh process, without imported legacy modules')
        self.plan_path = Path(plan_path).resolve(strict=True)
        stage, short_source = Path(stage).resolve(strict=True), Path(short_source).resolve(strict=True)
        for path, name in ((stage/'common.py', 'common.py'), (stage/'worker.py', 'worker.py'),
                           (short_source, 'short_worker.py')):
            require(digest(path) == PINNED[name], 'immutable coordinator source changed: ' + name)
        self.fit_path = Path(__file__).resolve().with_name('table6_missing190_fit.py')
        self.fit = import_file('_missing190_fit_adapter', self.fit_path)
        self.plan = self.fit.validate_plan(json.loads(self.plan_path.read_text()))
        require(self.plan['protocol'] == PROTOCOL, 'wrong scientific result protocol')
        self.out = Path(self.plan['output_root']).resolve()
        require(self.out != stage and not self.out.is_relative_to(stage), 'output overlaps original stage')
        os.environ['T6_BASE_STAGE'] = str(stage)
        os.environ['T6_MISSING190_PLAN'] = str(self.plan_path)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        sys.dont_write_bytecode = True
        # Import first: its immutable startup assertion checks the original19.
        self.short = import_file('_missing190_short_coordinator', short_source)
        self.base = self.short.base
        self.fit.initialize(self.plan_path, stage)
        sys.modules['fit'] = self.fit
        self.gpu_methods = tuple(m for m in self.fit.METHODS if m not in self.fit.CPU_METHODS)
        require(len(self.gpu_methods) == 6, 'wrong GPU method roster')
        self.formal_methods = tuple(self.fit.METHODS)
        self.sources = {'worker': digest(__file__), 'fit': digest(self.fit_path),
                        'deadline': digest(Path(__file__).with_name('table6_restart_deadline.py')),
                        **PINNED}
        self.smoke_mode = False
        self.inside_pair = False
        self._install(stage)

    def load_plan(self):
        current = self.fit.validate_plan(json.loads(self.plan_path.read_text()))
        require(current == self.plan, 'new plan changed during worker lifetime')
        return self.plan

    def _install(self, stage):
        base, short = self.base, self.short
        base.OUT = self.out
        base.METHODS = dict(self.fit.METHODS)
        base.CPU_METHODS = set(self.fit.CPU_METHODS)
        base.load_plan = self.load_plan
        # This immutable variable controls queue admission, not model hardware.
        # The actual CPU registry remains intact in common and in each fit child.
        short.GPU_METHODS = self.formal_methods
        short.CONTRACT = CONTRACT
        original_atomic, original_validate = base.atomic, base.validate_complete
        original_response = base.validate_response
        runtime = self

        def validate_complete(value, pair, plan):
            # The inherited aggregator constructs its historical protocol label
            # only in memory. Publication translates it; cached files MUST be new.
            allowed = {PROTOCOL, LEGACY_PROTOCOL} if runtime.inside_pair else {PROTOCOL}
            require(value.get('protocol') in allowed, 'cached result has wrong protocol')
            if value['protocol'] == PROTOCOL:
                require(value.get('runtime_sources') == runtime.sources, 'cached result runtime identity differs')
            return original_validate({**value, 'protocol': LEGACY_PROTOCOL}, pair, plan)

        def validate_response(value, request, request_path=None):
            require(value.get('missing190_plan_id') == runtime.plan['plan_id'], 'fit child used wrong plan/adapter')
            require(value.get('diagnostic_smoke') is runtime.smoke_mode, 'smoke/formal response crossover')
            device = 'cpu' if request['method'] in runtime.fit.CPU_METHODS else 'cuda'
            require(str(value.get('device', '')).startswith(device), 'fit used the wrong actual execution device')
            return original_response(value, request, request_path)

        def atomic(path, value):
            path = contained(path, runtime.out)
            is_result = path.is_relative_to(runtime.out/'results')
            if is_result:
                require(runtime.inside_pair and value.get('protocol') == LEGACY_PROTOCOL,
                        'unexpected result publication outside native aggregation')
                value = {**value, 'protocol': PROTOCOL, 'runtime_sources': runtime.sources,
                         'coordinator_contract': CONTRACT,
                         'lane': 'gpu_reserved; trees execute CPU-only with masked GPU'}
                pair = next(p for p in runtime.plan['pairs'] if p['key'] == value['key'])
                validate_complete(value, pair, runtime.plan)
            if is_result or path.name in {'response.json', 'selected.json'}:
                # Exclusive publication strengthens the native exists assertion.
                runtime.publish_exclusive(path, value)
            else:
                original_atomic(path, value)

        base.atomic = atomic
        base.validate_complete = validate_complete
        base.validate_response = validate_response
        old_run_pair = short.run_pair

        def run_pair(pair, row, plan, mode, claim):
            require(mode == 'gpu' and pair['method'] in runtime.formal_methods, 'formal work requires a GPU-reserved rank')
            require(pair in runtime.plan['pairs'] and row in runtime.plan['rows'], 'unknown formal pair/row')
            require(not runtime.inside_pair, 'nested pair execution forbidden')
            runtime.inside_pair = True
            try:
                return old_run_pair(pair, row, plan, mode, claim)
            finally:
                runtime.inside_pair = False

        short.run_pair = run_pair
        parent_proxy = base.subprocess

        class ChildProxy:
            def __getattr__(self, name):
                return getattr(parent_proxy, name)

            def Popen(self, argv, **kwargs):
                require(digest(runtime.fit_path) == runtime.sources['fit'] and
                        digest(__file__) == runtime.sources['worker'], 'adapter source changed before child launch')
                runtime.load_plan()
                argv = rewrite_child(argv, legacy_fit=stage/'fit.py', new_fit=runtime.fit_path,
                                     plan_path=runtime.plan_path, smoke=runtime.smoke_mode)
                kwargs['env'] = {**kwargs.get('env', os.environ),
                                 'T6_MISSING190_PLAN': str(runtime.plan_path),
                                 'T6_BASE_STAGE': str(stage), 'PYTHONDONTWRITEBYTECODE': '1'}
                return parent_proxy.Popen(argv, **kwargs)

        base.subprocess = ChildProxy()

    def publish_exclusive(self, path, value):
        import tempfile
        path = contained(path, self.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.missing190-', dir=path.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            os.link(name, path)
        finally:
            Path(name).unlink()

    def activate_budget(self):
        self.short.BUDGET = monotonic_budget(self.short)
        signal.signal(signal.SIGUSR1, self.short.signal_stop)
        signal.signal(signal.SIGTERM, self.short.signal_stop)
        self.short.budget().check()

    def preflight(self, mode, cpus):
        require(mode in {'gpu', 'cpu'} and cpus >= (8 if mode == 'gpu' else 16), 'invalid lane CPU reservation')
        if mode == 'cpu':
            mask_cpu_visibility()
        import torch
        record = {**self.base.identity(), 'mode': mode, 'plan_id': self.plan['plan_id'],
                  'runtime_sources': self.sources, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                  'devices': torch.cuda.device_count(), 'cpus': cpus}
        require(len(record['cpu_affinity']) >= cpus, 'insufficient bound CPU cores')
        if mode == 'gpu':
            require(torch.cuda.is_available() and record['devices'] == 1, 'exactly one bound GPU required')
            values = torch.arange(16, device='cuda')
            require(int((values*values).sum().cpu()) == 1240, 'GPU preflight arithmetic failed')
            hip = ctypes.CDLL(ctypes.util.find_library('amdhip64') or 'libamdhip64.so')
            bus = ctypes.create_string_buffer(64)
            require(hip.hipDeviceGetPCIBusId(bus, 64, 0) == 0, 'physical GPU identity unavailable')
            record['physical_gpu'] = bus.value.decode()
        else:
            require(record['devices'] == 0, 'CPU smoke can see a GPU')
        record.update(passed=True, epoch=time.time())
        self.base.atomic(self.out/'preflight'/record['job_id']/f'{mode}-{record["rank"]}.json', record)
        return record

    def check_smoke(self):
        checks = []
        for method in self.fit.METHODS:
            folder = self.out/'startup_smoke'/method
            receipt = json.loads(contained(folder/'pass.json', self.out).read_text())
            require(receipt.get('passed') is True and receipt.get('method') == method and
                    receipt.get('plan_id') == self.plan['plan_id'] and receipt.get('runtime_sources') == self.sources,
                    'missing or stale real-model smoke: ' + method)
            request_path, response_path = folder/'trial/request.json', folder/'trial/response.json'
            contained(request_path, self.out); contained(response_path, self.out)
            require(digest(request_path) == receipt['request_sha256'] and digest(response_path) == receipt['response_sha256'],
                    'smoke proof changed: ' + method)
            request = json.loads(request_path.read_text())
            self.fit.validate_request(request, self.plan, diagnostic_smoke=True)
            require(request['method'] == method and request['max_epoch'] == 2 and request['batch_size'] == 64,
                    'wrong diagnostic request')
            self.smoke_mode = True
            try:
                result = self.base.validate_response(json.loads(response_path.read_text()), request, request_path)
            finally:
                self.smoke_mode = False
            require(result.get('missing190_adapter', {}).get('method') == method and
                    math.isfinite(float(result['fit_seconds'])) and result['fit_seconds'] > 0,
                    'smoke did not demonstrate an actual model fit')
            require(str(result['device']).startswith('cpu' if method in self.fit.CPU_METHODS else 'cuda'),
                    'smoke ran on wrong lane')
            checks.append({'method': method, 'request_sha256': receipt['request_sha256'],
                           'response_sha256': receipt['response_sha256']})
        require(len(checks) == 9, 'all nine actual model smokes are mandatory')
        return checks

    def smoke(self, method):
        require(method in self.fit.METHODS, 'unknown smoke method')
        lock = contained(self.out/'startup_smoke'/method/'smoke.lock', self.out)
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('a') as stream:
            self.base.fcntl.flock(stream, self.base.fcntl.LOCK_EX | self.base.fcntl.LOCK_NB)
            return self._smoke_locked(method)

    def _smoke_locked(self, method):
        require(method in self.fit.METHODS, 'unknown smoke method')
        mode = 'cpu' if method in self.fit.CPU_METHODS else 'gpu'
        cpus = 16 if mode == 'cpu' else 8
        identity = self.preflight(mode, cpus)
        import optuna
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
        config = self.fit.suggest_config(study.ask(), method, 'classification')
        datasets = {p['dataset'] for p in self.plan['pairs'] if p['method'] == method}
        row = min((r for r in self.plan['rows'] if r['dataset'] in datasets),
                  key=lambda r: (r['work_size'], r['dataset']))
        request = {'mode': 'trial', 'method': method, 'row': row, 'config': config, 'seed': 0,
                   'cpus': cpus, 'batch_size': 64, 'max_epoch': 2}
        folder = self.out/'startup_smoke'/method
        self.smoke_mode = True
        # Its named GPU_METHODS is an admission list, not device configuration.
        # Only this fresh diagnostic process admits one explicitly CPU-masked tree.
        self.short.GPU_METHODS = (method,)
        try:
            result = self.short.execute(request, folder/'trial')
        finally:
            self.short.GPU_METHODS = self.formal_methods
            self.smoke_mode = False
        require(result['fit_seconds'] > 0, 'smoke must run a real fit')
        proof = {'passed': True, 'method': method, 'plan_id': self.plan['plan_id'],
                 'runtime_sources': self.sources, 'preflight': identity,
                 'configuration_origin': 'Optuna TPE seed0 sampled packaged search space; no defaults',
                 'request_sha256': digest(folder/'trial/request.json'),
                 'response_sha256': digest(folder/'trial/response.json')}
        if (folder/'pass.json').exists():
            old = json.loads((folder/'pass.json').read_text())
            require(all(old[k] == proof[k] for k in proof if k != 'preflight'), 'existing smoke proof is incompatible')
        else:
            self.publish_exclusive(folder/'pass.json', proof)
        return proof

    def gate(self, ranks):
        require(type(ranks) is int and 1 <= ranks <= 8, 'GPU rank count must be1..8')
        job = self.base.identity()['job_id']
        records = [json.loads((self.out/'preflight'/job/f'gpu-{rank}.json').read_text()) for rank in range(ranks)]
        for rank, record in enumerate(records):
            require(record['passed'] is True and record['plan_id'] == self.plan['plan_id'] and
                    record['runtime_sources'] == self.sources and record['job_id'] == job and
                    record['rank'] == rank and record['mode'] == 'gpu' and record['devices'] == 1 and
                    record['physical_gpu'] and len(record['cpu_affinity']) >= 8, 'GPU preflight mismatch')
        require(len({r['host'] for r in records}) == 1 and
                len({r['physical_gpu'] for r in records}) == ranks, 'GPU overlap or multiple nodes')
        masks = [set(r['cpu_affinity']) for r in records]
        require(sum(map(len, masks)) == len(set.union(*masks)), 'GPU CPU affinity overlaps')
        checks = self.check_smoke()
        value = {'passed': True, 'contract': CONTRACT, 'plan_id': self.plan['plan_id'], 'job_id': job,
                 'gpu_ranks': ranks, 'cpu_ranks': 0, 'runtime_sources': self.sources,
                 'methods': list(self.formal_methods), 'lane': 'gpu_reserved_mixed_device',
                 'actual_execution_modes': {m: 'cpu' if m in self.fit.CPU_METHODS else 'gpu' for m in self.formal_methods}}
        self.base.atomic(self.out/'preflight'/job/'gate.json', value)
        self.base.atomic(self.out/'preflight'/job/'smoke-gate.json', {**value, 'checks': checks})
        return value

    def worker(self, mode):
        require(mode == 'gpu', 'formal worker requires a GPU-reserved lane, including CPU tree fits')
        checks = self.check_smoke()  # Never trust a bare passed flag from an old gate.
        identity = self.base.identity()
        for name in ('gate.json', 'smoke-gate.json'):
            gate = json.loads((self.out/'preflight'/identity['job_id']/name).read_text())
            require(gate['runtime_sources'] == self.sources and 0 <= identity['rank'] < gate['gpu_ranks'],
                    'unapproved worker runtime/rank')
            if name == 'smoke-gate.json':
                require(gate['checks'] == checks, 'smoke proof differs from launch gate')
        return self.short.worker('gpu')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'smoke', 'check-smoke', 'gate', 'worker'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--base-stage', type=Path, default=R19)
    parser.add_argument('--short-source', type=Path, default=SHORT)
    parser.add_argument('--mode', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--method')
    parser.add_argument('--ranks', type=int, default=8)
    args = parser.parse_args(argv)
    runtime = Runtime(args.plan, args.base_stage, args.short_source)
    try:
        runtime.activate_budget()
        if args.action == 'preflight':
            result = runtime.preflight(args.mode, 8 if args.mode == 'gpu' else 16)
        elif args.action == 'smoke':
            result = runtime.smoke(args.method)
        elif args.action == 'check-smoke':
            result = runtime.check_smoke()
        elif args.action == 'gate':
            result = runtime.gate(args.ranks)
        else:
            result = runtime.worker(args.mode)
        print(json.dumps(result, sort_keys=True), flush=True)
    except runtime.short.BudgetStop as exc:
        runtime.short.lifecycle('paused', phase=args.action, reason=str(exc),
                                actual_execution_mode='cpu' if args.method in runtime.fit.CPU_METHODS else 'gpu')
        print(json.dumps({'state': 'paused', 'reason': str(exc)}), flush=True)


if __name__ == '__main__':
    main()
