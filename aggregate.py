"""Strict457 per loop; each loop is separately ranked against fixed baselines."""
import csv
import json
import math
from common import *

def ranks(values):
    return [1 + sum(other > value + 1e-12 for other in values)
            + (sum(abs(other - value) <= 1e-12 for other in values) - 1) / 2 for value in values]

def finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False

def main():
    membership = manifest_rows()
    expected = {r['dataset'] for r in membership}
    suites = {r['dataset']: r['suite'] for r in membership}
    with (STAGE / 'baseline_detail.tsv').open() as f:
        baseline = {r['dataset']: r for r in csv.DictReader(f, delimiter='\t')}
    assert set(baseline) == expected
    report = {'complete': True, 'step': 19750, 'loops': {},
              'rank_rule': 'Each inference-loop variant separately competes with the same eight fixed baselines; tied rank uses accuracy tolerance 1e-12.'}
    all_panels = {}
    for passes in (3, 4):
        results = []
        for rank in range(8):
            payload = json.loads((OUTPUT / f'tasks/rank-{rank:02d}/loop{passes}/results.json').read_text())
            assert payload['complete'] is True and payload['inference_loops'] == passes
            assert payload['step'] == 19750 and payload['training_loops'] == 2
            assert payload['precision'] == 'FP32' and payload['amp'] is False and payload['fa3'] is False
            assert payload['checkpoint_identity'] == checkpoint_identity()
            results.extend(payload['rows'])
        validate_panel(results, expected)
        panel = {r['dataset']: float(r['accuracy']) for r in results}
        all_panels[passes] = panel
        atomic_json(OUTPUT / f'loop{passes}_strict457.json', {'complete': True, 'step': 19750,
                    'inference_loops': passes, 'dataset_count': 457, 'suite_counts': COUNTS,
                    'rows': [dict(r, suite=suites[r['dataset']]) for r in results]})
        scopes = {'All457': expected, 'Displayed428': {d for d in expected if suites[d] != 'PFN'}}
        scopes.update({suite: {d for d in expected if suites[d] == suite} for suite in COUNTS})
        scores = {}
        for scope, names in scopes.items():
            missing = [d for d in sorted(names) if any(not finite(baseline[d].get(f'{m}_accuracy')) for m in METHODS)]
            score = {'memberships': len(names), 'mean_accuracy': sum(panel[d] for d in names) / len(names),
                     'rank_complete': not missing, 'missing_baseline_memberships': missing}
            if not missing:
                accum = {m: [] for m in METHODS + ('g5sc',)}
                for dataset in sorted(names):
                    values = [float(baseline[dataset][f'{m}_accuracy']) for m in METHODS] + [panel[dataset]]
                    for m, rank_value in zip(accum, ranks(values)):
                        accum[m].append(rank_value)
                score['all_method_average_ranks'] = {m: sum(v) / len(v) for m, v in accum.items()}
                score['average_rank'] = score['all_method_average_ranks']['g5sc']
            scores[scope] = score
        report['loops'][str(passes)] = scores
    report['loop4_vs_loop3'] = {'wins': sum(all_panels[4][d] > all_panels[3][d] + 1e-12 for d in expected),
                               'losses': sum(all_panels[4][d] < all_panels[3][d] - 1e-12 for d in expected),
                               'ties': sum(abs(all_panels[4][d] - all_panels[3][d]) <= 1e-12 for d in expected)}
    for scope in ('All457', 'Displayed428'):
        if all(report['loops'][str(p)][scope]['rank_complete'] for p in (3, 4)):
            report[f'rank_best_loop_{scope}'] = min((3, 4), key=lambda p: (
                report['loops'][str(p)][scope]['average_rank'],
                -report['loops'][str(p)][scope]['mean_accuracy'], p))
    atomic_json(OUTPUT / 'comparison.json', report)
    lines = ['# G5SC step19750: inference Loop 3 vs 4', '',
             'Same Loop-2-trained checkpoint. No retraining or parameter changes.', '',
             '| Inference loop | Scope | Memberships | Average rank | Mean accuracy |',
             '|---|---|---:|---:|---:|']
    for passes, scores in report['loops'].items():
        for scope, score in scores.items():
            rank_text = f"{score['average_rank']:.6f}" if score['rank_complete'] else 'Incomplete baseline coverage'
            lines.append(f"| {passes} | {scope} | {score['memberships']} | {rank_text} | {score['mean_accuracy']:.6f} |")
    lines += ['', report['rank_rule'], '', 'All457 includes PFN 29; Displayed428 reproduces the five-suite screenshot coverage.']
    (OUTPUT / 'comparison.md').write_text('\n'.join(lines) + '\n')
    atomic_json(OUTPUT / 'experiment.complete.json', {'complete': True, 'step': 19750,
                'inference_loops': [3, 4], 'datasets_per_loop': 457, 'evaluated_memberships': 914,
                'all457_ranking_complete': all(report['loops'][str(p)]['All457']['rank_complete'] for p in (3, 4))})
    print(json.dumps({'complete': True, 'evaluated_memberships': 914}), flush=True)

if __name__ == '__main__':
    main()
