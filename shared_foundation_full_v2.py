"""Full frozen memberships on one verified existing GPU, under runtime guards.

This new entry uses the unchanged sidecar's source/ownership/GPU/smoke/budget
checks. Only this process's scheduling policy and unsuccessful-attempt handling
are replaced, for its lifetime. No old file, manifest, scientific argument,
support row, query row, prediction or completed receipt is modified.

Every unclaimed original task is eligible, including unknown/large dimensions.
Eligibility is not a promise it fits: own-tree32GiB, same-UID60GiB, parent64GiB,
child40GiB and inherited two-hour deadline guards remain mandatory. Unsuccessful
attempts retain their canonical claim and isolated evidence, never automatically
retry, and require separately authorized quiescent recovery.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
import uuid

import shared_foundation_sidecar as base

q = base.q
RESOURCE = dict(base.RESOURCE)
ELIGIBILITY = {'policy': 'all_full_original_tasks_with_runtime_guard', 'full_rows_unchanged': True}
ALLOWED_PARENTS = {'196092', '196093', '200798', '200797', '204828',
                   '204827', '204826', '194259', '194180'}
INSTALLED = False


def full_task(task, overlay=None):
    """Describe scheduling only; return no changed task or model inputs."""
    row = task['row']
    shape = (overlay or {}).get(str(task['dataset_index']), row) if task['task_kind'] == 'regression' else row
    dimensions = {key: shape.get(key) for key in ('train_rows', 'test_rows', 'features')}
    return {'eligible': True, 'policy': ELIGIBILITY['policy'], 'dimensions': dimensions,
            'shape_source': 'audit_only_no_admission_threshold', 'unknown_dimensions_skipped': False,
            'memory_fit_not_preclaimed': True, 'full_rows_unchanged': True}


def attempt(plan, campaign_path, man, task, owner, budget, root, smoke=False):
    """Canonical successes only; retain every unsuccessful owned reservation."""
    owner = dict(owner, manifest_id=man['manifest_id'])
    base.guard(base.snapshot(), budget)
    reservation = None
    if not smoke:
        token = uuid.uuid4().hex
        if not q.claim(man, task, dict(owner, reservation_token=token)):
            return {'state': 'already_claimed', 'canonical_touched': False}
        reservation = base.claim_identity(man, task, token)
    attempt_id = ('smoke' if smoke else 'formal')+'-'+task['task_kind']+'-'+str(task['dataset_index'])+'-'+uuid.uuid4().hex
    temporary_root = root/'attempts'/attempt_id
    temporary_man = dict(man, output_root=str(temporary_root))
    events, failure, ok = [], None, False
    try:
        with base.guarded_runtime(budget, events):
            ok = q.launch(temporary_man, campaign_path, task, owner, smoke=smoke)
    except BaseException as exc:
        failure = type(exc).__name__+': '+str(exc)
    # Never publish, release or finish an attempt while an owned child survives.
    cleanup = base.cleanup_children()
    output = q.task_path(temporary_man, 'smoke' if smoke else 'results', task)
    value = {'manifest_id': man['manifest_id'], 'plan_id': plan['plan_id'],
             'task': {key: task[key] for key in ('task_kind', 'dataset_index', 'dataset')},
             'policy': ELIGIBILITY['policy'], 'attempt_id': attempt_id,
             'attempt_output': str(output), 'reservation': reservation, 'cleanup': cleanup,
             'owner': owner, 'guard_events': events, 'exception': failure, 'finished_epoch': time.time()}
    result = None
    if ok and not events and failure is None:
        try:
            result = q.valid_result(output, man, task)
            base.require(result['physical_gpu']['uuid'] == owner['uuid'] and
                         result['physical_gpu']['pci_bus_id'] == owner['pci'], 'Completed result GPU mismatch')
        except Exception as exc:
            failure = type(exc).__name__+': '+str(exc)
            value['exception'] = failure
    if result is None or events or failure is not None:
        if reservation is not None:
            base.require(base.claim_identity(man, task, reservation['token']) == reservation,
                         'Reservation changed during unsuccessful attempt')
        value.update(state='retained_resource_deferral' if reservation is not None else 'operationally_deferred',
                     reason='runtime_guard' if events else 'unsuccessful_or_unverified_native_attempt',
                     reservation_retained=reservation is not None, automatic_retry=False,
                     recovery_requires_quiescent_authorization=reservation is not None,
                     canonical_error_published=False,
                     failure_cause_not_inferred='Native logs and isolated error receipt are retained; '
                                              'not evidence that the scientific model or dataset is invalid.')
        q.atomic(man, root/'deferrals'/(attempt_id+'.json'), value)
        return value
    if not smoke:
        base.require(base.claim_identity(man, task, reservation['token']) == reservation,
                     'Reservation changed before success publication')
        q.atomic(man, q.task_path(man, 'results', task), result)
    value['state'] = 'complete'
    q.atomic(man, root/'finished'/(attempt_id+'.json'), value)
    return value


@contextmanager
def installed_policy():
    """Legacy functions resolve globals in their own module, not this wrapper."""
    global INSTALLED
    base.require(not INSTALLED, 'Nested full-lane policy installation forbidden')
    original = (base.ELIGIBILITY, base.small_task, base.attempt, base.source_checks)
    old_source_checks = base.source_checks

    def source_checks(plan):
        old_source_checks(plan)
        verified = {q.verify_file(record) for record in plan['source_records']}
        base.require(Path(__file__).resolve() in verified and Path(base.__file__).resolve() in verified,
                     'New full-lane entry and inherited sidecar must both be pinned')

    INSTALLED = True
    base.ELIGIBILITY, base.small_task, base.attempt, base.source_checks = ELIGIBILITY, full_task, attempt, source_checks
    try:
        yield
    finally:
        base.ELIGIBILITY, base.small_task, base.attempt, base.source_checks = original
        INSTALLED = False


def load_and_run(path):
    path = Path(path).resolve(strict=True)
    with installed_policy():
        plan, campaigns = base.load_plan(path)
        base.require(str(plan['parent_job_id']) in ALLOWED_PARENTS, 'Parent outside explicit full-lane allowlist')
        base.require(plan['resource'] == RESOURCE and plan['cpus'] == 4 and
                     plan['mem_gib'] == 40 and plan['parent_mem_gib'] == 64 and
                     plan['max_step_seconds'] == 7200, 'Unexpected full-lane resource/deadline contract')
        base.require(Path(plan['entry_script']).resolve() == Path(__file__).resolve(), 'Wrong full-lane entry pin')
        return base.run(plan, campaigns, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args(argv)
    base.require(__debug__, 'Optimized Python unsupported')
    print(json.dumps(load_and_run(args.plan), sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
