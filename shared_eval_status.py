"""Read-only status of explicitly named shared evaluation lanes."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

import tabfm_default_dispatch as q
from shared_eval_launch import ROOT, STAGE


def collect(attempt):
    if not attempt.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Unsafe attempt name')
    result = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'attempt': attempt, 'lanes': {}, 'campaigns': {}}
    for suffix in ('gap190', 'swiftdual'):
        directory = STAGE/(attempt+'-'+suffix)
        lane = {'directory': str(directory), 'records': {}}
        for name in ('plan', 'launch-receipt', 'node-resource-gate', 'node-terminal', 'cleanup-fatal'):
            path = directory/(name+'.json')
            if path.is_file():
                value = q.read(path)
                if name == 'plan':
                    value = {k:value.get(k) for k in ('plan_id','sidecar_id','parent_job_id','node','gpu','cpus')}
                lane['records'][name] = value
        path = directory/'launch.log'
        if path.is_file():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size-12000))
                lane['log_tail'] = stream.read().decode(errors='replace')
        for campaign in ('table6_missing190_standard_hpo_20260922_v1',
                         'tabswift_official16_standard681_20260922_v1',
                         'tabswift_budget32x8_standard681_20260922_v1'):
            side = ROOT/'evaluation'/campaign/'sidecars'/(attempt+'-'+suffix)
            if side.exists():
                lane.setdefault('scientific_audits', {})[campaign] = {
                    'preflight': q.read(side/'preflight.json') if (side/'preflight.json').exists() else None,
                    'smoke_gate_exists': (side/'smoke_gate.json').exists(),
                    'finished': q.read(side/'lane_finished.json') if (side/'lane_finished.json').exists() else None,
                    'deferrals': [q.read(p) for p in sorted((side/'deferrals').glob('*.json'))],
                    'finished_attempts': [q.read(p) for p in sorted((side/'finished').glob('*.json'))],
                }
                pf = lane['scientific_audits'][campaign]['preflight']
                if pf is not None:
                    pf['eligibility_counts'] = {
                        kind: sum(x['eligible'] for x in pf['eligibility'] if x['task_kind']==kind)
                        for kind in ('classification','regression')}
                    del pf['eligibility']
        result['lanes'][suffix] = lane
    import tabswift_dispatch  # Original stricter validator for budget32x8 results.
    for name in ('tabfm_defaults_standard681_20260922_v1',
                 'tabswift_official16_standard681_20260922_v1',
                 'tabswift_budget32x8_standard681_20260922_v1'):
        man,tasks = q.load_campaign(ROOT/'evaluation'/name/'manifest.json')
        result['campaigns'][name] = q.status(man,tasks)
    query = subprocess.run(['squeue','-u','guangyi.chen','-h','-o','%i|%j|%T|%q|%b|%R'],
                           capture_output=True,text=True,check=True,timeout=30).stdout
    ids = {'208251','208345','208407', *(str(x) for x in range(208377,208387))}
    result['queued_jobs'] = [x for x in query.splitlines() if x.split('|')[0] in ids]
    result['steps'] = subprocess.run(['squeue','--steps','-j','196092,196093,204826,206116',
                                     '-h','-o','%i|%j|%T|%M|%N'],
                                    capture_output=True,text=True,check=True,timeout=30).stdout
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();value=collect(args.attempt)
    if args.output:
        from table6_restart_ag import publish
        publish(args.output,value)
    print(json.dumps(value,allow_nan=False))
