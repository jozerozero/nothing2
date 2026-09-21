"""One guarded, non-published diagnostic of the first frozen native smoke task."""
import faulthandler
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import tabfm_existing_sidecar_v2 as sidecar
import tabfm_default_dispatch as frozen


@contextmanager
def diagnostic_watchdog():
    original = frozen.process_rss
    def guarded(pid):
        snapshot = sidecar.rss_snapshot()
        sidecar.guard_snapshot(snapshot)
        sidecar.require(snapshot['own_tree_rss_bytes'] <= 16 * 1024**3,
                        'Bounded smoke diagnostic exceeds16GiB')
        return original(pid)
    frozen.process_rss = guarded
    try:
        yield
    finally:
        frozen.process_rss = original


def main():
    faulthandler.enable(all_threads=True)
    plan, man, tasks = sidecar.load_plan(sys.argv[1])
    sidecar.query_parent(plan)
    env = sidecar.lane_environment(plan, os.environ)
    local_tmp = tempfile.mkdtemp(prefix='tfm-', dir='/tmp')
    env['TMPDIR'] = local_tmp
    os.environ.clear()
    os.environ.update(env)
    os.nice(19)
    memory = sidecar.rss_snapshot()
    sidecar.guard_snapshot(memory)
    sidecar.require(memory['other_same_uid_rss_bytes'] <= 40 * 1024**3,
                    'Bounded20GiB smoke needs4GiB parent margin')
    print('DIAGNOSTIC_MEMORY_20G', json.dumps(memory), flush=True)
    print('CPU_BIND', json.dumps(sidecar.bind_idle_cpu_cores()), flush=True)
    sidecar.check_idle(sidecar.gpu_idle_record())
    import torch
    from pfn_mitra_one import gpu_identity
    actual = gpu_identity(torch)
    owner = {'rank': 0, 'job': sidecar.PARENT, 'step': os.environ['SLURM_STEP_ID'],
             'node': sidecar.NODE, 'uuid': actual['uuid'], 'pci': actual['pci_bus_id']}
    diagnostic = dict(man, output_root=str(Path(man['output_root']) / 'diagnostics' / 'native-smoke-shorttmp1'))
    task = frozen.smoke_tasks(man, tasks)[0]
    original = frozen.worker_environment
    def traced_env(manifest, owner):
        env = original(manifest, owner)
        env.update(PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1', TMPDIR=local_tmp)
        return env
    frozen.worker_environment = traced_env
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, frozen.stop)
    signal.alarm(600)
    frozen.require(frozen.claim(diagnostic, task, owner, smoke=True), 'Diagnostic already attempted')
    events = []
    print('DIAGNOSTIC_START', json.dumps({'owner': owner, 'TMPDIR': local_tmp}), flush=True)
    with diagnostic_watchdog():
        ok = frozen.launch(diagnostic, Path(plan['campaign_path']), task, owner, smoke=True)
    print('DIAGNOSTIC_DONE', json.dumps({'success': ok, 'events': events}), flush=True)
    signal.alarm(0)


if __name__ == '__main__':
    main()
