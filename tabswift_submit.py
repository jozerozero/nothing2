#!/usr/bin/env python3
"""Submit one held, verified TabSwift two-protocol job, without automatic retry.

The immutable attempt journal is published before sbatch. Any later invocation,
including after an uncertain submission or release, only reports that journal.
This helper does not prepare manifests, cancel jobs, or change existing jobs.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from eval_one import object_digest, publish_new, require, verify_file
from classification32_submit import EXCLUDED_NODES, EXCLUDE


BASE = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
STAGE = BASE / 'stage/tabswift_standard681_20260922_v1'
JOB_NAME = 'swift681x2'
PROTOCOLS = ['official16', 'budget32x8']


def command(argv):
    return subprocess.check_output(argv, text=True).strip()


def read(path):
    return json.loads(Path(path).read_text())


def verify_document(value, id_key):
    require(value.get(id_key) == object_digest({k: v for k, v in value.items() if k != id_key}),
            f'{id_key} content hash mismatch')


def verify_records(records, label):
    require(isinstance(records, list) and records, f'No {label} records')
    verified = {}
    for record in records:
        path = Path(record['path'])
        require(path.is_absolute() and not path.is_symlink(), f'Invalid {label} path: {path}')
        resolved = verify_file(record)
        require(resolved not in verified, f'Duplicate {label} identity: {resolved}')
        verified[resolved] = record
    return verified


def load_plan(plan_path):
    plan_path = Path(plan_path)
    require(plan_path.is_absolute() and not plan_path.is_symlink(), 'Plan must be an absolute regular file')
    require(plan_path.resolve(strict=True) == STAGE / 'plan.json', 'Unexpected campaign plan path')
    plan = read(plan_path)
    verify_document(plan, 'plan_id')
    require(Path(plan['output_root']) == STAGE and not STAGE.is_symlink(), 'Unexpected campaign output root')
    require(plan.get('membership_count') == 1362 and plan.get('protocol_variants') == PROTOCOLS,
            'Exactly official16 and budget32x8, each standard681, are required')
    python = Path(plan['worker_python'])
    require(python.is_absolute() and python.is_file() and os.access(python, os.X_OK),
            'Missing executable absolute worker Python')
    sources = verify_records(plan['source_records'], 'source')
    repo = Path(__file__).resolve().parent
    script = repo / 'tabswift_slurm.sh'
    required = {Path(__file__).resolve(), script, repo / 'tabswift_dispatch.py',
                repo / 'eval_one.py', repo / 'classification32_submit.py',
                repo / 'classification32_campaign.py'}
    require(required <= set(sources), 'Submit/script/dispatcher/helper sources must all be pinned')
    require(script.read_text().startswith('#!/bin/bash'), 'Invalid Slurm shell script')
    manifests = verify_records(plan['campaign_manifests'], 'campaign manifest')
    require(len(manifests) == 2, 'Exactly two distinct campaign manifests required')
    manifest_ids, outputs = [], []
    for path in manifests:
        man = read(path)
        verify_document(man, 'manifest_id')
        require(man.get('membership_count') == 681 and man.get('classification_count') == 457
                and man.get('regression_count') == 224, 'Each protocol must cover classification457 + regression224')
        require(man.get('worker_python') == plan['worker_python'], 'Campaign Python differs from plan')
        output = Path(man['output_root'])
        require(output.is_absolute() and output != STAGE and not output.is_symlink(),
                'Invalid or non-isolated per-protocol output root')
        outputs.append(str(output.resolve()))
        manifest_ids.append(man['manifest_id'])
        for kind, count in (('classification', 457), ('regression', 224)):
            data_path = verify_file(man[kind + '_manifest'])
            data = read(data_path)
            verify_document(data, 'manifest_id')
            require(data.get('membership_count') == len(data['rows']) == count,
                    f'{kind} membership count mismatch')
            indices = [r.get('dataset_index', r.get('position')) for r in data['rows']]
            require(indices == list(range(count)) and len({r['dataset'] for r in data['rows']}) == count,
                    f'{kind} membership identities/order changed')
    require(len(set(manifest_ids)) == len(set(outputs)) == 2, 'Two isolated protocols must not share identity/output')
    return plan, script, manifest_ids


def verify_job(job, script, raw, *, held, run=command):
    fields = dict(re.findall(r'([^\s=]+)=([^\s]+)', raw))
    expected = {'JobId': str(job), 'JobName': JOB_NAME, 'Partition': 'faculty', 'Account': 'faculty-acc',
                'QOS': 'bgqos', 'NumTasks': '4', 'NumCPUs': '16', 'CPUs/Task': '4',
                'MinMemoryNode': '256G', 'Nice': '0', 'Requeue': '0', 'Dependency': '(null)',
                'Command': str(script), 'WorkDir': str(script.parent),
                'StdOut': str(STAGE / 'logs' / f'job-{job}.out'),
                'StdErr': str(STAGE / 'logs' / f'job-{job}.err')}
    for key, value in expected.items():
        require(fields.get(key) == value, f'Submission contract mismatch {key}={fields.get(key)!r}, expected {value!r}')
    require(fields.get('UserId', '').startswith('guangyi.chen('), 'Wrong job owner')
    require(fields.get('NumNodes') in ('1', '1-1') and fields.get('TimeLimit') in ('1-00:00:00', '24:00:00'),
            'Job must reserve one node for 24h')
    require('gres/gpu=4' in fields.get('ReqTRES', '').split(','), 'Job must reserve four GPUs')
    require(set(fields.get('TresPerTask', '').split(',')) == {'cpu=4', 'gres/gpu=1'}
            and fields.get('NtasksPerN:B:S:C', '').split(':')[0] == '4',
            'Job must bind one GPU and four CPUs per rank')
    excluded_text = fields.get('ExcNodeList', '')
    require(excluded_text not in ('', '(null)', 'None'), 'Excluded-node constraint lost')
    # Slurm may omit unavailable nodes from its display; the full submitted list
    # is retained in the immutable command journal and checked at preflight.
    excluded = set(run(['scontrol', 'show', 'hostnames', excluded_text]).splitlines())
    require(excluded and excluded <= EXCLUDED_NODES, 'Unexpected excluded nodes')
    for key in ('NodeList', 'SchedNodeList'):
        if fields.get(key) not in (None, '(null)', 'None'):
            assigned = set(run(['scontrol', 'show', 'hostnames', fields[key]]).splitlines())
            require(not assigned & EXCLUDED_NODES, 'Job assigned an excluded physical node')
    if held:
        require(fields.get('JobState') == 'PENDING' and fields.get('Reason') == 'JobHeldUser',
                'Job is not safely held before release')
    else:
        require(fields.get('Reason') != 'JobHeldUser', 'Release unconfirmed: scheduler still reports user hold')
    return fields


def sbatch_command(script, plan_path):
    return ['sbatch', '--hold', '--parsable', '--job-name=' + JOB_NAME,
            '--partition=faculty', '--account=faculty-acc', '--qos=bgqos', '--nodes=1',
            '--ntasks=4', '--ntasks-per-node=4', '--gpus-per-task=1', '--cpus-per-task=4',
            '--mem=256G', '--time=1-00:00:00', '--no-requeue', '--nice=0', '--exclude=' + EXCLUDE,
            '--chdir=' + str(script.parent), '--output=' + str(STAGE / 'logs/job-%j.out'),
            '--error=' + str(STAGE / 'logs/job-%j.err'), '--export=ALL,PYTHONHASHSEED=0',
            str(script), str(plan_path)]


def status_summary():
    names = ('submission_attempt.json', 'submitted_job.json', 'verified_held_job.json',
             'release_attempt.json', 'released_job.json', 'submission_failure.json')
    receipts = {name: read(STAGE / name) for name in names if (STAGE / name).exists()}
    attempt = receipts.get('submission_attempt.json')
    if attempt is not None:
        require(attempt.get('job_name') == JOB_NAME, 'Existing submission journal belongs to another job')
    submitted = receipts.get('submitted_job.json', {})
    released = receipts.get('released_job.json', {})
    return {'attempt_exists': attempt is not None, 'job_id': submitted.get('job_id'),
            'held_verified': 'verified_held_job.json' in receipts,
            'release_attempted': 'release_attempt.json' in receipts,
            'release_confirmed': bool(released), 'state_at_release': released.get('fields', {}).get('JobState'),
            'submission_uncertain': attempt is not None and not submitted,
            'release_uncertain': 'release_attempt.json' in receipts and not released,
            'failure': receipts.get('submission_failure.json'), 'automatic_retry': False,
            'receipts': receipts, 'note': 'Journal snapshot only, not live completion. Inspect scheduler before manual recovery.'}


def submit(plan_path, run=command):
    require(STAGE.is_dir() and not STAGE.is_symlink(), 'Prepare the isolated campaign first')
    with (STAGE / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (STAGE / 'submission_attempt.json').exists():
            return {'idempotent_noop': True, **status_summary()}
        plan, script, manifest_ids = load_plan(plan_path)
        queue = run(['squeue', '--me', '-h', '-o', '%i|%j|%T'])
        matches = [line for line in queue.splitlines() if len(line.split('|')) >= 2
                   and line.split('|')[1] == JOB_NAME]
        require(not matches, 'Existing swift681x2 job; refusing duplicate submission: ' + '; '.join(matches))
        (STAGE / 'logs').mkdir(exist_ok=True)
        args = sbatch_command(script, plan_path)
        context = {'job_name': JOB_NAME, 'plan_id': plan['plan_id'], 'plan_path': str(plan_path),
                   'campaign_manifests': plan['campaign_manifests'], 'manifest_ids': manifest_ids,
                   'protocol_variants': PROTOCOLS, 'membership_count': 1362,
                   'scope_per_protocol': {'classification': 457, 'regression': 224},
                   'worker_python': plan['worker_python'], 'source_records': plan['source_records'],
                   'output_root': str(STAGE), 'command': args, 'excluded_nodes': sorted(EXCLUDED_NODES),
                   'resources': {'nodes': 1, 'tasks': 4, 'gpus': 4, 'cpus_per_task': 4,
                                 'memory_gib': 256, 'hours': 24, 'nice': 0, 'requeue': False,
                                 'partition': 'faculty', 'account': 'faculty-acc', 'qos': 'bgqos'},
                   'automatic_retry': False, 'epoch': time.time()}
        publish_new(STAGE / 'submission_attempt.json', context)
        phase, job, response = 'submitting_held', None, None
        try:
            response = run(args)
            match = re.fullmatch(r'([0-9]+)(?:;[^\s;]+)?', response)
            require(match is not None, 'Unrecognized sbatch response; do not retry automatically')
            job = match.group(1)
            publish_new(STAGE / 'submitted_job.json', dict(context, job_id=job, sbatch_response=response,
                                                           epoch=time.time()))
            phase = 'verifying_held'
            raw = run(['scontrol', 'show', 'job', job, '-o'])
            fields = verify_job(job, script, raw, held=True, run=run)
            publish_new(STAGE / 'verified_held_job.json', {'job_id': job, 'plan_id': plan['plan_id'],
                        'raw': raw, 'fields': fields, 'epoch': time.time()})
            # A queued job must not be released after any pinned source changed.
            current_plan, current_script, current_ids = load_plan(plan_path)
            require(current_plan == plan and current_script == script and current_ids == manifest_ids,
                    'Plan or pinned source changed before release')
            phase = 'releasing'
            raw = run(['scontrol', 'show', 'job', job, '-o'])
            verify_job(job, script, raw, held=True, run=run)
            publish_new(STAGE / 'release_attempt.json', {'job_id': job, 'plan_id': plan['plan_id'],
                        'final_held_check': raw, 'epoch': time.time(), 'automatic_retry': False})
            release_response = run(['scontrol', 'release', job])
            phase = 'verifying_released'
            raw = run(['scontrol', 'show', 'job', job, '-o'])
            fields = verify_job(job, script, raw, held=False, run=run)
            publish_new(STAGE / 'released_job.json', {'job_id': job, 'plan_id': plan['plan_id'],
                        'raw': raw, 'fields': fields, 'release_response': release_response, 'epoch': time.time()})
        except BaseException as exc:
            publish_new(STAGE / 'submission_failure.json', {'job_id': job, 'plan_id': plan['plan_id'],
                        'phase': phase, 'sbatch_response': response, 'error': f'{type(exc).__name__}: {exc}',
                        'epoch': time.time(), 'automatic_retry': False, 'other_jobs_modified': False})
            raise
    return status_summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('submit', 'status'))
    parser.add_argument('--plan', type=Path, default=STAGE / 'plan.json')
    args = parser.parse_args(argv)
    require(args.plan.is_absolute() and args.plan == STAGE / 'plan.json', 'Only this campaign plan is authorized')
    value = status_summary() if args.mode == 'status' else submit(args.plan)
    print(json.dumps(value, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
