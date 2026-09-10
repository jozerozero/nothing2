"""Pinned, one-attempt recovery; never mutate source shard results or checkpoints."""
import csv
import json
import math
import os
from pathlib import Path

ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
TAG = 'e4_g5sc_loop3_resume181407_eval_fp32_online_16gpu_gt_step50_20260911_v1'
STAGE = ROOT/'stage'/TAG/'online_resume'
OUT = ROOT/'evaluation/e4_g5sc_loop3_resume181407_fp32_online_24gpu_gt_step50_20260910_v1/E4_G5SC_LOOP3/lineage-178786-181407'
OLD_TAG = 'e4_g5sc_loop3_resume181407_eval_fp32_online_24gpu_gt_step50_20260910_v1'
OLD_WORK = ROOT/'analysis'/OLD_TAG/'E4_G5SC_LOOP3/lineage-178786-181407/job-181580/work'
NAME = 'e4g5sc3r16gt1'
POLICY = ROOT/'stage/rw_sample50_gpu_shard_validation1_v1/gpu_shard_policy_lpt4.json'

def signatures(paths):
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}

def validate_shard(path, checkpoint, step, index, policy):
    record = json.loads((path/'shard_result.json').read_text())
    expected = dict(checkpoint=str(checkpoint), model_tag=f'step-{step}', shard_index=index,
                    dataset_count=len(policy['shards'][index]), explicit_fp32=True,
                    clf_use_amp=False, clf_use_fa3=False, n_estimators=32,
                    outer_batch=8, n_jobs=1, kv_cache=False)
    for key, value in expected.items():
        assert record[key] == value, (path, key, record[key], value)
    panel = path/f'step-{step}'/'talent_detailed.txt'
    with panel.open(newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    names = [r['dataset'] for r in rows]
    assert len(names) == len(set(names)) == len(policy['shards'][index])
    assert set(names) == set(policy['shards'][index])
    assert all(math.isfinite(float(v)) for r in rows for k,v in r.items() if k != 'dataset')
    assert all(0 <= float(r['accuracy']) <= 1 for r in rows)
    return signatures([path/'shard_result.json', panel])

def runtime_contract():
    receipt = json.loads((STAGE/'submission_state.json').read_text())
    assert receipt['registration_complete'] is True
    assert str(receipt['evaluation_job']) == os.environ['SLURM_JOB_ID']
    assert os.environ['SLURM_JOB_NAME'] == NAME
    assert os.environ['SLURM_JOB_QOS'] == 'gtqos'
    assert receipt['retained_steps'] == [8950]
    assert receipt['failed_job'] == 181580
    return receipt

def install_reusable_shard(receipt, step, index, target, checkpoint, policy):
    key = f'{step}/{index}'
    source = receipt['reusable_shards'].get(key)
    if source is None:
        return
    original = Path(source['path'])
    assert original == OLD_WORK/f'step-{step}'/f'shard-{index}'
    assert validate_shard(original, checkpoint, step, index, policy) == source['signatures']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        assert target.resolve() == original.resolve()
    else:
        assert not target.exists(), target
        target.symlink_to(original, target_is_directory=True)
