"""Eight disjoint membership shards; each shard evaluates both authorized loops."""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import torch
from common import *

def main():
    rank = int(os.environ['SLURM_PROCID'])
    assert int(os.environ['SLURM_NTASKS']) == 8 and 0 <= rank < 8
    assert int(os.environ['SLURM_CPUS_PER_TASK']) == 4
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert (OUTPUT / 'preflight/contract_pass.json').is_file()
    rows = manifest_rows()
    assigned = rows[rank::8]
    names = [r['dataset'] for r in assigned]
    task = OUTPUT / 'tasks' / f'rank-{rank:02d}'
    task.mkdir(parents=True, exist_ok=False)
    names_path = task / 'datasets.txt'
    names_path.write_text('\n'.join(names) + '\n')
    atomic_json(task / 'preflight.json', {'complete': True, 'rank': rank, 'job_id': os.environ['SLURM_JOB_ID'],
                'assigned_memberships': names, 'loops': [3, 4], 'single_visible_gpu': True})
    for passes in (3, 4):
        start = time.time()
        tag = f'g5sc-step19750-inference-loop{passes}'
        work = task / f'loop{passes}'
        work.mkdir()
        env = os.environ.copy()
        env.update({'TABICL_SOURCE_ROOT': str(STAGE / 'source_loop'), 'TABICL_EVAL_DISABLE_LOCAL_SRC': '1',
                    'CROSS_TABLE_ARM': 'E4', 'G5SC_INFERENCE_LOOPS': str(passes),
                    'PYTHONHASHSEED': '0', 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4',
                    'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'NUMEXPR_NUM_THREADS': '4',
                    'PYTHONPATH': f'{STAGE}:{STAGE}/source_loop/src:{STAGE}/evaluator'})
        command = [sys.executable, str(STAGE / 'invoke_evaluator.py'),
                   '--model_path', str(CHECKPOINT), '--model_tag', tag,
                   '--data_root', str(BENCH / 'data'), '--cache_root', str(BENCH / 'cache'),
                   '--outdir', str(work), '--dataset_names_file', str(names_path),
                   '--clf_n_estimators', '32', '--clf_norm_methods', 'none,power',
                   '--clf_batch_size', '8', '--clf_n_jobs', '1', '--cpu_threads', '4',
                   '--kv_cache', 'false', '--clf_use_amp', 'false', '--clf_use_fa3', 'false']
        print(json.dumps({'event': 'loop_shard_started', 'rank': rank, 'passes': passes, 'count': len(names)}), flush=True)
        with (work / 'evaluator.log').open('x') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log = (work / 'evaluator.log').read_text()
        assert 'strict_checkpoint_override' in log and 'realized_loop_verified' in log
        with (work / tag / 'talent_detailed.txt').open() as f:
            results = list(csv.DictReader(f, delimiter='\t'))
        validate_panel(results, names)
        identity = checkpoint_identity()
        assert identity == json.loads((OUTPUT / 'input_contract.json').read_text())['checkpoint_identity']
        atomic_json(work / 'results.json', {'complete': True, 'step': 19750, 'training_job': 174381,
                    'training_loops': 2, 'inference_loops': passes, 'rank': rank,
                    'job_id': os.environ['SLURM_JOB_ID'], 'checkpoint_identity': identity,
                    'precision': 'FP32', 'amp': False, 'fa3': False, 'dataset_count': len(names),
                    'elapsed_seconds': time.time() - start, 'rows': results})
        print(json.dumps({'event': 'loop_shard_complete', 'rank': rank, 'passes': passes, 'count': len(names)}), flush=True)
    atomic_json(task / 'complete.json', {'complete': True, 'rank': rank, 'loops': [3, 4]})

if __name__ == '__main__':
    main()
