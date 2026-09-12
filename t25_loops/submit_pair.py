"""Explicit-account, once-only held submission; release only after both verify."""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import time

from prepare_deployment import NAMES, STAGE, EXCLUDE, account_name, arm_root, digest, validated_stage, write_new
from validate_launch import load_contract, validate, require


def command(*args):
    process = subprocess.run(args, text=True, capture_output=True)
    if process.returncode:
        raise RuntimeError(f'{list(args)!r}: exit={process.returncode}\n{process.stdout}\n{process.stderr}')
    return process.stdout.strip()


def atomic_state(path, payload):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def accounting_preflight(account):
    output = command('sacctmgr', '-nP', 'show', 'assoc', f'user={getpass.getuser()}',
                     f'account={account}', 'format=Account,Partition,QOS,MaxSubmitJobs')
    records = [line.split('|') for line in output.splitlines() if line.strip()]
    relevant = [row for row in records if len(row) >= 4 and row[0] == account and row[1] in ('', 'faculty')]
    require(relevant, 'no faculty-compatible account association was returned')
    require(all(row[3] != '0' for row in relevant), 'account MaxSubmitJobs=0 forbids submission')
    return output


def verify_held(control, stage, loop, account):
    fields = dict(re.findall(r'([A-Za-z0-9_/]+)=(\S+)', control))
    expected = {'JobName': NAMES[loop], 'Account': account, 'QOS': 'bgqos', 'Partition': 'faculty',
                'NumNodes': '8', 'NumTasks': '8', 'NumCPUs': '1024', 'CPUs/Task': '128',
                'Requeue': '0', 'Nice': '0', 'TimeLimit': '3-00:00:00', 'JobState': 'PENDING',
                'WorkDir': str(stage), 'Command': str(Path(stage) / f'loop{loop}.slurm'),
                'Dependency': '(null)'}
    for key, value in expected.items():
        require(fields.get(key) == value, f'held Loop{loop} {key}: observed={fields.get(key)!r}, expected={value!r}')
    require(fields.get('Reason') in {'JobHeldUser', 'JobHeldAdmin'}, 'job is not observably held')
    require(fields.get('Priority') == '0', 'held job has nonzero scheduling priority')
    tres = dict(item.split('=', 1) for item in fields.get('ReqTRES', '').split(',') if '=' in item)
    require(tres.get('gres/gpu') == '64', 'requested total GPU count is not 64')
    require(tres.get('mem') in {'16T', '16384G', '16777216M'}, 'requested total memory is not 16 TiB')
    require(fields.get('MinMemoryNode') in {'2T', '2048G', '2097152M'}, 'per-node memory is not 2 TiB')
    require(fields.get('TresPerNode', fields.get('TRESPerNode')) in {'gres/gpu:8', 'gres:gpu:8'}, 'per-node GPU count is not 8')
    require(fields.get('StdOut') == str(arm_root(loop, 'logs', stage) / f'{NAMES[loop]}-{fields.get("JobId")}.out'), 'stdout outside new arm log root')
    require(fields.get('StdErr') == str(arm_root(loop, 'logs', stage) / f'{NAMES[loop]}-{fields.get("JobId")}.err'), 'stderr outside new arm log root')
    wanted = set(command('scontrol', 'show', 'hostnames', EXCLUDE).splitlines())
    actual = set(command('scontrol', 'show', 'hostnames', fields.get('ExcNodeList', '')).splitlines())
    require(actual == wanted, 'held allocation exclusion list differs from frozen historical+unsafe set')
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--account', required=True, help='Explicit user-approved account; no inferred/default account.')
    args = parser.parse_args()
    stage = validated_stage(args.stage)
    account = account_name(args.account)
    manifest = load_contract(stage)
    require(manifest['account'] == account, 'explicit account differs from frozen sbatch files')
    require(not (stage / 'submission.intent').exists() and not (stage / 'submission_state.json').exists(), 'submission already attempted; inspect retained state, never blindly retry')
    queue = command('squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%j|%t')
    require(not any(line.split('|')[1] in NAMES.values() for line in queue.splitlines() if len(line.split('|')) >= 2), 'same-name job already active')
    association = accounting_preflight(account)
    validation = {}
    for loop in NAMES:
        require(not arm_root(loop, stage=stage).exists(), f'checkpoint root is not fresh for Loop{loop}')
        _, receipt, _, config = validate(stage, loop, '__CHECKPOINT_DIR__', account)
        validation[str(loop)] = {'native_identity_status': receipt['status'], 'parsed_config': config,
                                 'sbatch_test_only': command('sbatch', '--test-only', f'--account={account}', str(stage / f'loop{loop}.slurm'))}
    intent = {'stage': str(stage), 'manifest_id': manifest['manifest_id'], 'account': account,
              'intent_epoch': time.time(), 'pid': os.getpid(), 'names': list(NAMES.values())}
    write_new(stage / 'submission.intent', intent)
    state_path = stage / 'submission_state.json'
    state = {**intent, 'phase': 'INTENT_RECORDED', 'account_association_observed': association,
             'queue_before_observed': queue, 'preflight': validation, 'components': {},
             'identity_receipt_sha256': digest(stage / manifest['identity_receipt'])}
    atomic_state(state_path, state)
    try:
        for loop in NAMES:
            # Persist intent before each external action, so ambiguous sbatch
            # outcomes never cause an automatic duplicate submission.
            state['phase'] = f'SUBMITTING_HELD_LOOP{loop}'
            atomic_state(state_path, state)
            raw = command('sbatch', '--hold', '--parsable', f'--account={account}', str(stage / f'loop{loop}.slurm'))
            job = raw.split(';')[0]
            require(job.isdigit(), f'unrecognized sbatch response: {raw!r}')
            state['components'][str(loop)] = {'job_id': job, 'sbatch_response': raw,
                                              'state': 'SUBMITTED_HELD', 'name': NAMES[loop]}
            atomic_state(state_path, state)
        # Validate both only after both job IDs are durable. Any failure leaves
        # both held and records the exact failure; no unrelated job is changed.
        for loop in NAMES:
            arm = state['components'][str(loop)]
            control = command('scontrol', 'show', 'job', '-o', arm['job_id'])
            arm['held_control_observed'] = control
            atomic_state(state_path, state)
            verify_held(control, stage, loop, account)
            arm['state'] = 'HELD_VERIFIED'
            atomic_state(state_path, state)
        # Recheck immutable source and smoke after allocation creation.
        load_contract(stage)
        require(digest(stage / manifest['identity_receipt']) == state['identity_receipt_sha256'], 'identity receipt changed during submit')
        state['phase'] = 'BOTH_HELD_AND_VERIFIED'
        atomic_state(state_path, state)
        for loop in NAMES:
            arm = state['components'][str(loop)]
            state['phase'] = f'RELEASING_LOOP{loop}'
            atomic_state(state_path, state)
            arm['release_response'] = command('scontrol', 'release', arm['job_id'])
            arm['released_control_observed'] = command('scontrol', 'show', 'job', '-o', arm['job_id'])
            arm['state'] = 'RELEASE_COMMAND_SUCCEEDED'
            atomic_state(state_path, state)
        state['phase'] = 'BOTH_RELEASE_COMMANDS_SUCCEEDED'
        state['completed_epoch'] = time.time()
        atomic_state(state_path, state)
    except Exception as exc:
        state['error'] = {'type': type(exc).__name__, 'message': str(exc), 'epoch': time.time()}
        atomic_state(state_path, state)
        raise
    print(json.dumps(state, sort_keys=True))


if __name__ == '__main__':
    main()
