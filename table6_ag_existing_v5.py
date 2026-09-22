"""One bounded CPU-only continuation after completed 206117.207.

This operational adapter reuses the hash-pinned v4 function bodies in a private
globals dictionary. The deployed v3/v4 files, scientific fitter, seed budgets,
claim CAS, memory guards, and model configuration are not modified. Only the
new namespace, exact allowed parent, and previous-completion proof differ.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import types

import table6_ag_existing_v4 as v4

v3, ag, old = v4.v3, v4.ag, v4.old
require, read, publish = v4.require, v4.read, v4.publish
STAGE = ag.ROOT / 'stage/table6_autogluon_existing_20260922_v5'
CAMPAIGN = STAGE.name
PREVIOUS_STAGE = v4.STAGE
ALLOWED = {'206117': 'auh7-1b-gpu-306'}
PREVIOUS_STEPS = {'206117': '207'}
RANKS, RESERVED, EFFECTIVE = 2, 32, 16
PINNED = {
    'table6_ag_existing_v4.py': '48f7cbdd59a3c187af28d29d4a9ca264b161c1b03e25f05e2fa23ff9d8d5d6b9',
    'table6_ag_existing_v3.py': '109055f3213fc121c40a9ccb33da2ab3216f61bbfd9e8e7dbec484aab9f3a106',
    'table6_restart_ag.py': '3ca3bf4d620b026f29409a9361630f0110227c116a68317a975948ccb79ecfaf',
    'table6_restart_ag_sidecar.py': '4b7a609073c3a362795a820ce1f8a336f304e48d5fcb6841387ba06780026a21',
    'table6_restart_deadline.py': '047352b1b52c947ea3b2017692c422d65f451b56dca18283564383b51a8b9950',
}


def sources():
    result = {name: hashlib.sha256((v3.REPO / name).read_bytes()).hexdigest() for name in PINNED}
    require(result == PINNED, 'frozen operational/scientific source changed')
    result[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def target(parent, node, ranks):
    require(type(ranks) is int and ranks == RANKS, 'v5 permits exactly two CPU ranks')
    require(ALLOWED.get(parent) == node, 'v5 permits only parent206117 on node306')


def validate_parent(raw, parent, node, ranks):
    target(parent, node, ranks)
    return v4.validate_parent(raw, parent, node, ranks)


def terminal_identity(parent, step, *, run=subprocess.run, now=None):
    require(re.fullmatch(r'\d+', parent or '') and re.fullmatch(r'\d+', step or ''), 'numeric owner required')
    identity = parent + '.' + step
    env = dict(os.environ, TZ='UTC'); env.pop('SLURM_TIME_FORMAT', None)
    def query(args):
        result = run(args, text=True, capture_output=True, timeout=20, env=env)
        require(result.returncode == 0, 'owner scheduler query failed: ' + result.stderr)
        return result.stdout
    rows = [line.split('|') for line in query(['sacct', '-n', '-P', '-j', identity,
                                             '-o', 'JobIDRaw,State%30,End']).splitlines()]
    exact = [row for row in rows if len(row) >= 3 and row[0] == identity]
    require(len(exact) == 1, 'missing/duplicate exact previous-step accounting')
    _, state, end = exact[0][:3]
    require(state.split() and state.split()[0].rstrip('+') in ag.TERMINAL, 'existing AG step remains active')
    ended = dt.datetime.fromisoformat(end)
    ended = ended.replace(tzinfo=dt.timezone.utc) if ended.tzinfo is None else ended
    now = time.time() if now is None else now
    age = now - ended.timestamp()
    require(age >= 120, 'previous-step termination grace not reached')
    ids = [row.strip() for row in query(['squeue', '--steps', '-h', '-j', parent, '-o', '%i']).splitlines() if row.strip()]
    require(all(re.fullmatch(re.escape(parent) + r'\.(?:\d+|batch|extern)', item) for item in ids),
            'unknown/truncated owner step identity')
    require(identity not in ids, 'previous exact step still queued')
    return {'owner': identity, 'state': state, 'end': end, 'age_seconds': age,
            'checked_epoch': now, 'exact_step_absent_from_queue': True}


def previous_attempt(directory, parent, node, *, run=subprocess.run, now=None):
    """Require the exact completed v4 attempt; never erase or rewrite old claims."""
    target(parent, node, RANKS)
    directory = Path(directory).resolve()
    require(directory.parent == (PREVIOUS_STAGE / 'launches').resolve(), 'not the immutable v4 launch namespace')
    records = [v4.file_record(directory / name) for name in ('plan.json', 'completion.json', 'srun-intent.json')]
    plan, completion, intent = [record['value'] for record in records]
    require(plan['parent'] == parent and plan['node'] == node and plan['plan_id'] == ag.PLAN and
            plan['source_hashes'] == PINNED, 'previous v4 identity/source differs')
    require(plan['ranks'] == RANKS and plan['reserved_cpus_per_rank'] == RESERVED and
            plan['effective_cpus_per_rank'] == EFFECTIVE, 'previous CPU reservation differs')
    require(plan['operational_digest'] == ag.digest({k: v for k, v in plan.items() if k != 'operational_digest'}),
            'previous operational plan digest differs')
    require(intent['command'] == plan['command'] == v4.srun_command(parent, node, RANKS, directory),
            'previous srun identity differs')
    require(type(completion.get('returncode')) is int and completion['returncode'] == 0,
            'only the reviewed successful bounded v4 completion may continue')
    step = PREVIOUS_STEPS[parent]
    proof = terminal_identity(parent, step, run=run, now=now)
    require(proof['state'].split()[0].rstrip('+') == 'COMPLETED', 'reviewed previous v4 step was not COMPLETED')
    for rank in range(RANKS):
        record = v4.file_record(directory / f'rank-{rank}.json')
        value = record['value']
        require(value['job_id'] == parent and value['step_id'] == step and value['host'] == node and
                value['rank'] == rank and value['effective_fit_cpus'] == EFFECTIVE and
                value['slurm_cpus_per_task'] == str(RESERVED), 'previous rank/step identity differs')
        require(v4.select_cpus(value['reserved_cpu_ids'], value['cpu_percent_samples']) == value['selected_cpu_ids'],
                'previous CPU-selection receipt differs')
        records.append(record)
    audit = ag.OUT / 'auxiliary' / PREVIOUS_STAGE.name / f'j{parent}-s{step}'
    require(audit.is_dir() and not audit.is_symlink(), 'previous audit missing')
    require(not list(audit.glob('failed-*.json')) and not list(audit.glob('cleanup-fatal-*.json')),
            'previous worker/cleanup failed; separate review required')
    for rank in range(RANKS):
        record = v4.file_record(audit / f'finished-{rank}.json')
        finished, owner = record['value'], record['value']['owner']
        require(finished['state'] in ('paused', 'queue_exhausted') and owner['job_id'] == parent and
                owner['step_id'] == step and owner['host'] == node and owner['rank'] == rank and
                owner['plan_id'] == ag.PLAN and owner['worker_sha256'] == old.WORKER_SHA,
                'previous rank lacks durable clean finish')
        records.append(record)
    claims_dir = ag.OUT / 'claims'
    require(claims_dir.is_dir() and not claims_dir.is_symlink(), 'canonical claims unavailable')
    matched, terminated_running, checked = [], [], {}
    for path in sorted(claims_dir.glob('*.json')):
        claim = read(path)
        if claim.get('method') != 'AutoGluon':
            continue
        job_id, step_id = str(claim.get('job_id', '')), str(claim.get('step_id', ''))
        if (job_id, step_id) == (parent, step):
            require(claim.get('state') in ('paused', 'complete') and claim.get('descendants_reaped') is True and
                    claim.get('plan_id') == ag.PLAN and claim.get('worker_sha256') == old.WORKER_SHA,
                    'previous step claim lacks clean paused/complete cleanup proof')
            matched.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'state': claim['state']})
        elif claim.get('state') == 'running':
            identity = (job_id, step_id)
            if identity not in checked:
                # Historic legacy claims may have no numeric step. Preserve the
                # frozen, stricter whole-parent terminal+grace policy for these
                # unrelated owners, including its exact "job gone" handling.
                # This is observation only; pair takeover remains under the
                # original frozen worker's flock/CAS policy.
                def supported_query(command, **kwargs):
                    if command[:2] == ['squeue', '--steps']:
                        command = ['%i' if item == '%i|%T' else item for item in command]
                    response = run(command, **kwargs)
                    # This site's Slurm emits this alternate exact spelling
                    # for an expired job. Only normalize the spelling; the
                    # frozen caller still requires prior terminal accounting
                    # and 120-second grace before allowing a missing job.
                    if command[0] == 'squeue' and response.returncode != 0 and not response.stdout.strip() and \
                            response.stderr.strip() == 'slurm_load_jobs error: Invalid job id specified':
                        return types.SimpleNamespace(returncode=response.returncode, stdout=response.stdout,
                                                     stderr='squeue: error: Invalid job id specified')
                    return response
                checked[identity] = v3.ag_original_terminal(claim, run=supported_query, now=now)
            terminated_running.append({'path': str(path), 'terminal': checked[identity]})
    return {'directory': str(directory), 'terminal': proof, 'source_records': records,
            'matching_canonical_claims': matched, 'terminated_historical_running_claims': terminated_running,
            'claims_untouched': True, 'checked_epoch': time.time() if now is None else now}


def no_other_ag_processes(rows, parent, step):
    """Reject other same-user AG wrappers on this node, including between claims."""
    target(parent, socket.gethostname(), RANKS)
    for row in rows:
        text = ' '.join(row['cmdline'])
        if not any(name in text for name in ('table6_restart_ag', 'table6_ag_existing_', 'table6_autogluon')):
            continue
        # Permit only the new adapter's own genuine Slurm step. Other wrappers
        # and their active fit descendants must finish before this launch.
        env = row['environ']
        require(Path(__file__).name in text and env.get('SLURM_JOB_ID') == parent and
                env.get('SLURM_STEP_ID') == step and env.get('SLURM_NTASKS') == str(RANKS),
                'another AG process exists on approved node: ' + str(row['pid']))


def rank_entry(directory):
    import psutil
    rows = []
    for process in psutil.process_iter(['pid', 'uids', 'cmdline']):
        try:
            if process.info['uids'].real != os.getuid():
                continue
            command = process.info['cmdline'] or []
            text = ' '.join(command)
            if any(name in text for name in ('table6_restart_ag', 'table6_ag_existing_', 'table6_autogluon')):
                rows.append({'pid': process.pid, 'cmdline': command, 'environ': process.environ()})
        except psutil.NoSuchProcess:
            continue
        # AccessDenied is intentionally fail-closed: absence cannot be proved.
    no_other_ag_processes(rows, os.environ.get('SLURM_JOB_ID'), os.environ.get('SLURM_STEP_ID'))
    return PRIVATE['rank_entry'](directory)


def configure(directory):
    sources()
    v3.target = target
    PRIVATE['configure'](directory)


# Explicitly reuse immutable v4 bodies with new operational globals, rather
# than replacing __file__ or any globals inside the v4 module itself.
PRIVATE = dict(vars(v4))
PRIVATE.update(__file__=__file__, STAGE=STAGE, CAMPAIGN=CAMPAIGN,
               sources=sources, previous_attempt=previous_attempt,
               validate_parent=validate_parent)
for _name in ('srun_command', 'verify_plan', 'launch', 'rank_entry', 'preflight', 'seed_entry', 'configure'):
    _function = getattr(v4, _name)
    PRIVATE[_name] = types.FunctionType(_function.__code__, PRIVATE, _function.__name__,
                                       _function.__defaults__, _function.__closure__)
launch = PRIVATE['launch']
srun_command = PRIVATE['srun_command']
verify_plan = PRIVATE['verify_plan']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'supervise', 'rank', 'work', 'seed'))
    parser.add_argument('--parent'); parser.add_argument('--node'); parser.add_argument('--proof', type=Path)
    parser.add_argument('--launch-id'); parser.add_argument('--previous-launch-dir', type=Path)
    parser.add_argument('--launch-dir', type=Path); parser.add_argument('--key'); parser.add_argument('--seed', type=int)
    args = parser.parse_args(argv)
    configure(args.launch_dir)
    if args.action == 'launch':
        print(json.dumps(launch(args.parent, args.node, args.proof, args.launch_id, args.previous_launch_dir)), flush=True)
        return 0
    if args.action == 'supervise':
        plan = verify_plan(args.launch_dir)
        previous_attempt(plan['previous_attempt']['directory'], plan['parent'], plan['node'])
        return v3.supervise(args.launch_dir)
    if args.action == 'rank':
        rank_entry(args.launch_dir)
        return 0
    if args.action == 'seed':
        PRIVATE['seed_entry'](args.launch_dir, args.key, args.seed)
        return 0
    return v3.work_entry(args.launch_dir)


if __name__ == '__main__':
    sys.exit(main())
