"""Bounded diagnostic only: native worker imports and one-GPU compute identity."""
import argparse
import faulthandler
import json
import os
import signal

import tabfm_existing_sidecar_v2 as sidecar


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--allocator', choices=['enabled', 'disabled'], required=True)
    args = p.parse_args()
    faulthandler.enable(all_threads=True)
    signal.alarm(120)
    plan, _, _ = sidecar.load_plan(args.plan)
    sidecar.query_parent(plan)
    env = sidecar.lane_environment(plan, os.environ)
    env['PYTHONFAULTHANDLER'] = '1'
    for name in ('PYTORCH_HIP_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'):
        env.pop(name, None)
        if args.allocator == 'enabled':
            env[name] = 'expandable_segments:True'
    os.environ.clear()
    os.environ.update(env)
    os.nice(19)
    memory = sidecar.rss_snapshot()
    sidecar.guard_snapshot(memory)
    sidecar.require(memory['other_same_uid_rss_bytes'] <= 54 * 1024**3,
                    'This diagnostic needs4GiB plus6GiB parent headroom')
    print('DIAGNOSTIC_MEMORY_4G', json.dumps(memory), flush=True)
    print('CPU_BIND', json.dumps(sidecar.bind_idle_cpu_cores()), flush=True)
    sidecar.check_idle(sidecar.gpu_idle_record())
    print('IMPORT_NUMPY', flush=True)
    import numpy
    print('IMPORT_PANDAS', flush=True)
    import pandas
    print('IMPORT_TORCH', flush=True)
    import torch
    print('IMPORT_SKLEARN', flush=True)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    print('IMPORT_THREADPOOLCTL', flush=True)
    from threadpoolctl import threadpool_limits
    print('TORCH_SET_THREADS', flush=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    print('GPU_IDENTITY', flush=True)
    from pfn_mitra_one import gpu_identity
    print('GPU_IDENTITY_OK', json.dumps(gpu_identity(torch)), flush=True)
    print('THREADPOOL_LIMITS', flush=True)
    with threadpool_limits(limits=4):
        print('PROBE_COMPLETE', json.dumps({'allocator': args.allocator,
              'numpy': numpy.__version__, 'pandas': pandas.__version__, 'torch': torch.__version__}), flush=True)


if __name__ == '__main__':
    main()
