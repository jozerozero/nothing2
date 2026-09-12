"""Assemble an immutable full-G5SC/T25 deployment; never submit an allocation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
STAGE = ROOT / 'stage/t25_fullg5sc_loop34_bg64_20260912_v2'
BASE = ROOT / 'stage/tabicl_regression_adapted_e4_20260822_v1'
PYTHON = '/vast/users/guangyi.chen/causal_group/zijian.li/tabicl_causal/new_tab/nothing3_clean_sp_lr01_1753412/.conda_env/bin/python'
NAMES = {3: 't25g5sc3v2', 4: 't25g5sc4v2'}
# Historical FULL G5SC exclusions, plus 193, 195, 216, 228, 287 and 296.
EXCLUDE = 'auh7-1b-gpu-[185-193,195,201,203,207,215-216,225,227-228,233,243-244,246,248,253-254,258,261-262,276,279-281,286-287,290-292,296-298,305,311,315]'
RESOURCES = {'partition': 'faculty', 'qos': 'bgqos', 'nodes': 8, 'tasks_per_node': 1,
             'gpus_per_node': 8, 'cpus_per_task': 128, 'memory_per_node': '2T',
             'time_limit': '3-00:00:00', 'nice': 0, 'requeue': False, 'dependency': None}
FROZEN_ENV = {
    'REGRESSION_EXPERIMENT_ARM': 'T25', 'REGRESSION_SAFE_TAIL_ENABLED': 'true',
    'REGRESSION_TL_LENGTH_CURRICULUM_ENABLED': 'false', 'REGRESSION_CROSS_TABLE_E4_ENABLED': 'false',
    'CROSS_TABLE_ENABLED': 'false', 'SYNTHETIC96_RW_DGP_ENABLED': 'false',
    'TRAIN_NP_SEED': '2026080101', 'TRAIN_TORCH_SEED': '2026080101',
    'PRIOR_LOADER_SEED': '2026080101', 'PYTHONHASHSEED': '0',
    'PRIOR_NUM_WORKERS': '4', 'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1', 'MKL_THREADING_LAYER': 'GNU', 'PYTHONDONTWRITEBYTECODE': '1',
    'PYTHONUNBUFFERED': '1', 'WANDB_MODE': 'disabled', 'WANDB_LOG': 'False',
    'BAD_BATCH_LOG_ENABLED': 'False',
    'RESUME_CHECKPOINT_DIR': '', 'RESUME_CHECKPOINT_PATH': '', 'CHECKPOINT_PATH': '',
    'ONLY_LOAD_MODEL': 'False', 'ALLOW_NONEXACT_PRIOR_RESUME': 'False',
    'SWIGLU_ENABLED': 'False', 'CR2_SHARED_REFINEMENT_ENABLED': 'False',
    'QK_PDS_ATTENTION_ENABLED': 'False', 'CLS8_POOLED_ENABLED': 'False', 'CLS8_WIDTH_ENABLED': 'False',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, sort_keys=True, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def account_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('invalid explicit Slurm account name')
    return value


def validated_stage(stage):
    stage = Path(stage).resolve()
    if stage.parent != ROOT / 'stage' or not re.fullmatch(r't25_fullg5sc_loop34_bg64_20260912_v[2-9][0-9]*(?:_[A-Za-z0-9]+)?', stage.name):
        raise RuntimeError('stage must be a fresh versioned full-G5SC stage, never the rejected v1 stage')
    return stage


def arm_root(loop, kind='checkpoints', stage=STAGE):
    return ROOT / kind / validated_stage(stage).name.replace('loop34_', f'loop{loop}_', 1)


def training_arguments(stage, loop, checkpoint_dir):
    """Single canonical argv: native G5 model/optimizer, T25 prior and pinball."""
    if loop not in NAMES:
        raise ValueError('only Loop3 and Loop4 are authorized')
    values = {
        'wandb_log': False, 'wandb_mode': 'disabled', 'wandb_project': 'T25-FullG5SC',
        'wandb_name': NAMES[loop], 'device': 'cuda', 'dtype': 'float32',
        'np_seed': 2026080101, 'torch_seed': 2026080101, 'prior_loader_seed': 2026080101,
        'max_steps': 25000, 'batch_size': 1024, 'micro_batch_size': 2,
        'optimizer': 'muon', 'lr': 6e-4, 'muon_momentum': .95, 'muon_ns_steps': 5,
        'muon_group_by_lr': False, 'cautious_weight_decay': True, 'weight_decay': .01,
        'scheduler': 'cosine_warmup', 'warmup_proportion': .02, 'lr_floor': 0.0,
        'fast_cosine_scheduler': False, 'gradient_clipping': 10.0,
        'gradient_clip_foreach': 'True', 'error_if_nonfinite_grad': False,
        'gradient_clip_error_sync_every': 1, 'gradient_clip_error_sync_until_step': -1,
        'ddp_bucket_cap_mb': 6, 'ddp_gradient_as_bucket_view': True, 'ddp_static_graph': False,
        'amp': True, 'grad_scaler': True, 'model_compile': False,
        'recompute': False, 'recompute_seq_len_threshold': 20000,
        'max_classes': 0, 'regression_method': 'quantile', 'num_quantiles': 999,
        'prior_type': 'graph_scm', 'prior_device': 'cpu', 'prior_n_jobs': 1,
        'prior_num_workers': 4, 'batch_size_per_gp': 4,
        'prior_prefetch_factor': 8, 'prior_persistent_workers': True, 'prior_pin_memory': False,
        'prior_cache_enabled': True, 'prior_cache_max_batches': 64, 'prior_cache_max_gb': 16,
        'prior_cache_prefill_batches': 2, 'prior_cache_prefill_gb': 0,
        'prior_cache_get_timeout_s': 600, 'prior_cache_put_timeout_s': 600, 'prior_cache_stats_every': 1,
        'batch_source_log_enabled': True, 'batch_source_log_every': 1,
        'batch_source_max_records': 25000, 'batch_source_flush_every': 1,
        'batch_source_print_to_stderr': True,
        'bad_batch_log_enabled': False,
        'min_features': 1, 'max_features': 100, 'min_seq_len': 4096, 'max_seq_len': 4096,
        'replay_small': False,
        'log_seq_len': False, 'seq_len_per_gp': True, 'min_train_size': .3, 'max_train_size': .9,
        'graph_noise': False, 'filter_unpredictable_graphs': True, 'filter_unpredictable_datasets': True,
        'allow_act_warping': False, 'min_n_nodes': 2, 'max_n_nodes': 32, 'cauchy_dag_offset': 0.0,
        'regression_target_prior': 'rw_sample50', 'regression_target_mix_probability': .25,
        'regression_target_profile': str(Path(stage) / 'artifacts/profile.json'),
        'embed_dim': 128, 'col_num_blocks': 3, 'col_nhead': 8, 'col_num_inds': 128,
        'col_feature_group': 'same', 'col_feature_group_size': 3, 'col_ssmax': 'qassmax-mlp-elementwise',
        'row_num_blocks': 3, 'row_nhead': 8, 'row_num_cls': 4, 'row_rope_base': 100000,
        'row_use_rope': True, 'icl_num_blocks': 12, 'icl_nhead': 8,
        'icl_ssmax': 'qassmax-mlp-elementwise', 'ff_factor': 2, 'activation': 'gelu',
        'norm_first': True, 'bias_free_ln': False,
        'shared_depth_icl_enabled': True, 'shared_depth_icl_rho': 1.0,
        'shared_depth_icl_dataset_conditioned': True, 'shared_depth_icl_num_passes': loop,
        'checkpoint_dir': str(checkpoint_dir), 'save_temp_every': 25, 'save_perm_every': 25,
        'max_checkpoints': 0, 'strict_training_stage_manifest': True,
        'strict_resume_training_contract': True, 'allow_nonexact_prior_resume': False,
    }
    return [part for key, value in values.items() for part in ('--' + key, str(value))]


def tracked_files(source):
    result = {}
    for path in sorted(Path(source).rglob('*')):
        if any(part in {'.git', '__pycache__', '__MACOSX'} or part.startswith('._') for part in path.parts):
            continue
        if path.suffix in {'.pyc', '.pyo'} or not path.is_file():
            continue
        if path.is_symlink():
            raise RuntimeError(f'source symlinks are not immutable inputs: {path}')
        result[str(path.relative_to(source))] = digest(path)
    if not result:
        raise RuntimeError('empty source tree')
    return result


def source_environment_keys(source):
    keys = set()
    for path in Path(source).rglob('*.py'):
        if '__pycache__' in path.parts or path.name.startswith('._'):
            continue
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get':
                owner = node.func.value
                if isinstance(owner, ast.Attribute) and owner.attr == 'environ' and node.args:
                    if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        keys.add(node.args[0].value)
    retain = {'PATH', 'HOME', 'USER', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'RANK', 'WORLD_SIZE',
              'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
              'ROCR_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'}
    return sorted(key for key in keys if key not in retain and not key.startswith(('SLURM_', 'NCCL_', 'TORCH_'))
                  and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key))


def slurm_text(stage, loop, account, python=PYTHON):
    logs = arm_root(loop, 'logs', stage)
    return f'''#!/usr/bin/env bash
#SBATCH --job-name={NAMES[loop]}
#SBATCH --partition=faculty
#SBATCH --account={account}
#SBATCH --qos=bgqos
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=2T
#SBATCH --time=3-00:00:00
#SBATCH --nice=0
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --exclude={EXCLUDE}
#SBATCH --chdir={stage}
#SBATCH --output={logs}/%x-%j.out
#SBATCH --error={logs}/%x-%j.err
set -euo pipefail
export T25_FULL_STAGE={shlex.quote(str(stage))}
export T25_LOOP_PASSES={loop}
export T25_PYTHON_BIN={shlex.quote(python)}
exec bash {shlex.quote(str(Path(stage) / 'run_full_g5sc_training.sh'))}
'''


def prepare_deployment(args):
    from prepare_source import prepare

    stage = validated_stage(args.stage)
    account = account_name(args.account)
    if stage.exists():
        raise FileExistsError(f'refusing to replace existing stage {stage}')
    for path in (args.profile, args.profile_audit):
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f'missing frozen profile input {path}')
        json.loads(path.read_text())
    stage.mkdir(parents=True, exist_ok=False)
    source_receipt = prepare(args.g5_source, stage / 'source', t25_source=args.t25_source,
                             t25_prior_overlay=args.t25_prior_overlay or args.t25_source)
    if source_receipt.get('g5sc_model_bytes_identical') is not True:
        raise RuntimeError('source assembly did not prove complete native model identity')
    (stage / 'artifacts').mkdir()
    shutil.copyfile(args.profile, stage / 'artifacts/profile.json')
    shutil.copyfile(args.profile_audit, stage / 'artifacts/profile_audit.json')
    files = ['prepare_deployment.py', 'validate_launch.py', 'submit_pair.py', 'run_full_g5sc_training.sh',
             'test_identity.py', 'runtime_probe.py']
    for name in files:
        shutil.copyfile(HERE / name, stage / name)
    # The isolated prior package owns Safe-Tail installation, including spawned
    # DataLoader workers. Never load the rejected external patch a second time.
    write_new(stage / 'sitecustomize.py',
              '"""Load only the frozen prior-owned Safe-Tail installation."""\n'
              'import os\n'
              'if os.environ.get("REGRESSION_SAFE_TAIL_ENABLED", "false").lower() == "true":\n'
              '    import tabicl.prior\n')
    files.append('sitecustomize.py')
    env_keys = source_environment_keys(stage / 'source')
    env_text = '# Generated from the frozen complete-source environment references.\n'
    for key in env_keys:
        env_text += f'unset {key}\n'
    for key, value in FROZEN_ENV.items():
        env_text += f'export {key}={shlex.quote(value)}\n'
    write_new(stage / 'frozen_environment.sh', env_text)
    for loop in NAMES:
        if arm_root(loop, stage=stage).exists():
            raise RuntimeError(f'fresh checkpoint root already exists: {arm_root(loop, stage=stage)}')
        arm_root(loop, 'logs', stage).mkdir(parents=True, exist_ok=True)
        write_new(stage / f'loop{loop}.slurm', slurm_text(stage, loop, account, args.python))
        write_new(stage / f'loop{loop}.argv.json', training_arguments(stage, loop, '__CHECKPOINT_DIR__'))
    hashes = tracked_files(stage / 'source')
    write_new(stage / 'source.sha256', ''.join(f'{sha}  source/{path}\n' for path, sha in hashes.items()))
    scripts = [*files, 'frozen_environment.sh', 'loop3.slurm', 'loop4.slurm', 'loop3.argv.json',
               'loop4.argv.json', 'source.sha256', 'source_contract.json', 'artifacts/profile.json', 'artifacts/profile_audit.json']
    manifest = {
        'schema_version': 2, 'status': 'PREPARED_REQUIRES_NATIVE_IDENTITY_SMOKE', 'stage': str(stage),
        'account': account, 'python': args.python, 'resources': RESOURCES, 'exclude': EXCLUDE,
        'names': {str(k): v for k, v in NAMES.items()}, 'frozen_environment': FROZEN_ENV,
        'sanitized_environment_keys': env_keys, 'source_hashes': hashes,
        'source_contract_sha256': digest(stage / 'source_contract.json'),
        'g5sc_model_hashes': source_receipt['g5sc_model_hashes'],
        'source_identity_sha256': object_digest(source_receipt['g5sc_model_hashes']),
        'files': {name: digest(stage / name) for name in scripts},
        'reference_inputs': {str(path): digest(path) for path in (args.profile, args.profile_audit)},
        'checkpoint_count_per_arm': 1000, 'checkpoint_steps': '25:25:25000',
        'identity_receipt': 'native_identity_smoke.json', 'submission_authorized_by_this_script': False,
    }
    manifest['manifest_id'] = object_digest(manifest)
    write_new(stage / 'deployment_manifest.json', manifest)
    for path in [stage / 'run_full_g5sc_training.sh', stage / 'loop3.slurm', stage / 'loop4.slurm']:
        subprocess.run(['bash', '-n', str(path)], check=True)
    print(json.dumps({'prepared': str(stage), 'submitted': False, 'manifest_id': manifest['manifest_id']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g5-source', required=True, type=Path)
    parser.add_argument('--t25-source', required=True, type=Path)
    parser.add_argument('--t25-prior-overlay', type=Path)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--profile-audit', required=True, type=Path)
    parser.add_argument('--account', required=True)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--python', default=PYTHON)
    prepare_deployment(parser.parse_args())


if __name__ == '__main__':
    main()
