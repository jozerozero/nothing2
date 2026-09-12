"""Release one existing T25 pair after strict Slurm fixed-node-range validation.

This companion does not edit the immutable deployment or submit any new job.
Its only parsing concession is the exact equivalence NumNodes=8-8 == 8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time


def normalize_fixed_nodes(control):
    tokens = re.findall(r'(?<!\S)NumNodes=(\S+)', control)
    if len(tokens) != 1 or tokens[0] not in {'8', '8-8'}:
        raise RuntimeError(f'not an exact eight-node request: {tokens!r}')
    return re.sub(r'(?<!\S)NumNodes=8-8(?=\s|$)', 'NumNodes=8', control)


def exact_job_id(control, expected):
    values = re.findall(r'(?<!\S)JobId=(\S+)', control)
    if values != [expected]:
        raise RuntimeError(f'wrong or ambiguous JobId: {values!r}, expected {expected}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    parser.add_argument('--account', required=True)
    parser.add_argument('--loop3-job', required=True)
    parser.add_argument('--loop4-job', required=True)
    parser.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    sys.path.insert(0, str(stage))
    import submit_pair as frozen
    from validate_launch import load_contract, validate, require
    from prepare_deployment import NAMES, digest, write_new

    require(Path(frozen.__file__).resolve() == stage / 'submit_pair.py',
            'must import original frozen submission validator')
    jobs = {'3': args.loop3_job, '4': args.loop4_job}
    require(all(job.isdigit() for job in jobs.values()) and len(set(jobs.values())) == 2,
            'two distinct explicit job IDs are required')
    state_path = stage / 'submission_state.json'
    state = json.loads(state_path.read_text())
    intent = json.loads((stage / 'submission.intent').read_text())
    manifest = load_contract(stage)
    require(state['stage'] == intent['stage'] == manifest['stage'] == str(stage), 'stage mismatch')
    require(state['account'] == intent['account'] == manifest['account'] == args.account, 'account mismatch')
    require(state['manifest_id'] == intent['manifest_id'] == manifest['manifest_id'], 'manifest mismatch')
    require(set(state['components']) == {'3', '4'}, 'unexpected component set')
    require(state.get('error', {}).get('message') == "held Loop3 NumNodes: observed='8-8', expected='8'",
            'this recovery only handles the recorded fixed-node-range parsing failure')
    require(not state.get('held_release_recovery'), 'recovery already attempted; inspect, never retry blindly')
    for loop, job in jobs.items():
        arm = state['components'][loop]
        require(arm['job_id'] == job and arm['name'] == NAMES[int(loop)], 'stored job identity mismatch')
        require(arm['state'] == 'SUBMITTED_HELD', 'job was already acted on after held submission')
    require(digest(stage / manifest['identity_receipt']) == state['identity_receipt_sha256'],
            'identity receipt changed since original submission')
    frozen.accounting_preflight(args.account)
    queue = frozen.command('squeue', '-h', '-u', frozen.getpass.getuser(), '-o', '%i|%j|%t')
    for loop, job in jobs.items():
        matching = [row.split('|') for row in queue.splitlines()
                    if len(row.split('|')) >= 3 and row.split('|')[1] == NAMES[int(loop)]]
        require(len(matching) == 1 and matching[0][0] == job, 'same-name queue is not unique')
    audit = {}
    for loop, job in jobs.items():
        validate(stage, int(loop), '__CHECKPOINT_DIR__', args.account)
        raw = frozen.command('scontrol', 'show', 'job', '-o', job)
        exact_job_id(raw, job)
        normalized = normalize_fixed_nodes(raw)
        frozen.verify_held(normalized, stage, int(loop), args.account)
        audit[loop] = {'job_id': job, 'raw_control': raw, 'normalized_control': normalized,
                       'validation': 'ALL_ORIGINAL_HELD_CHECKS_PASS_EXACT_8_8_NORMALIZATION_ONLY'}
    require(digest(stage / manifest['identity_receipt']) == state['identity_receipt_sha256'],
            'identity receipt changed during recovery preflight')
    recovery = {'started_epoch': time.time(), 'account': args.account, 'jobs': jobs,
                'original_error': dict(state['error']), 'audit': audit, 'actions': {},
                'method': 'exact NumNodes=8-8 normalization; all other original checks unchanged'}
    write_new(stage / 'held_release_recovery.intent', recovery)
    state['held_release_recovery'] = recovery
    state['phase'] = 'BOTH_HELD_AND_VERIFIED_RECOVERY'
    frozen.atomic_state(state_path, state)
    try:
        for loop, job in jobs.items():
            # Recheck immediately before each action. Preserve raw evidence.
            raw = frozen.command('scontrol', 'show', 'job', '-o', job)
            exact_job_id(raw, job)
            frozen.verify_held(normalize_fixed_nodes(raw), stage, int(loop), args.account)
            recovery['actions'][loop] = {'job_id': job, 'pre_release_control': raw,
                                         'release_intent_epoch': time.time()}
            state['phase'] = f'RECOVERY_RELEASING_LOOP{loop}'
            frozen.atomic_state(state_path, state)
            response = frozen.command('scontrol', 'release', job)
            recovery['actions'][loop]['release_response'] = response
            recovery['actions'][loop]['release_command_succeeded_epoch'] = time.time()
            state['components'][loop]['state'] = 'RELEASE_COMMAND_SUCCEEDED'
            frozen.atomic_state(state_path, state)
            released = frozen.command('scontrol', 'show', 'job', '-o', job)
            exact_job_id(released, job)
            state['components'][loop]['released_control_observed'] = released
            frozen.atomic_state(state_path, state)
        state['phase'] = 'BOTH_RELEASE_COMMANDS_SUCCEEDED'
        state['completed_epoch'] = time.time()
        recovery['status'] = 'RECOVERED_WITHOUT_RESUBMISSION'
        recovery['completed_epoch'] = state['completed_epoch']
        frozen.atomic_state(state_path, state)
    except Exception as exc:
        recovery['status'] = 'RECOVERY_FAILED_OR_UNCERTAIN_INSPECT_BEFORE_FURTHER_ACTION'
        recovery['error'] = {'type': type(exc).__name__, 'message': str(exc), 'epoch': time.time()}
        frozen.atomic_state(state_path, state)
        raise
    print(json.dumps({'phase': state['phase'], 'account': state['account'],
                      'components': state['components'], 'recovery_status': recovery['status']}, sort_keys=True))


if __name__ == '__main__':
    main()
