"""Recoverably archive only claims of this campaign's cancelled GPU smoke.

No result, job, parent, or unrelated claim is changed. Exact step identities and
Slurm terminal accounting are required before moving each claim atomically.
"""
import json
import os
from pathlib import Path
import subprocess

from reg224_mitra8_dispatch import ROOT

STEPS = {'196092.397', '200798.97', '204827.18', '194180.21'}


def main():
    parents = sorted({step.split('.')[0] for step in STEPS})
    active = subprocess.check_output(['squeue', '--steps', '-h', '-j', ','.join(parents),
                                      '-o', '%i|%j'], text=True)
    active_ids = {line.split('|')[0] for line in active.splitlines()}
    assert not STEPS & active_ids, 'Original smoke child still active'
    accounting = subprocess.check_output(['sacct', '-j', ','.join(sorted(STEPS)), '-n', '-P',
                                         '--format=JobIDRaw,State'], text=True)
    states = {line.split('|')[0]: line.split('|')[1].split()[0]
              for line in accounting.splitlines() if '|' in line}
    assert all(states.get(step) == 'CANCELLED' for step in STEPS), states
    archived = []
    for claim in sorted((ROOT / 'claims/mitra').glob('row-*.json')):
        record = json.loads(claim.read_text())
        owner = '.'.join(record['worker'].split('.')[:2])
        assert owner in STEPS and record['model'] == 'mitra', 'Unrelated claim'
        assert record['plan_id'] == 'bb88b773569137b3e7e7733bf7f85d4b56535cadb4c1f6a57e9f1df202006164'
        target = ROOT / 'archived_claims/v1_cancelled_smoke' / claim.name
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists(), 'Archive target already exists'
        os.replace(claim, target)
        archived.append({'owner': owner, 'archive': str(target), 'dataset_index': record['dataset_index']})
    print(json.dumps({'archived': archived, 'terminal_states': states, 'results_unchanged': True}))


if __name__ == '__main__':
    main()
