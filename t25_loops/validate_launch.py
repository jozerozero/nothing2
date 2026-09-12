"""Fail-closed full-source, native-model, parser, and smoke launch gate."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

from prepare_deployment import (FROZEN_ENV, NAMES, RESOURCES, STAGE, EXCLUDE, account_name,
                                arm_root, digest, object_digest, tracked_files, training_arguments, validated_stage, write_new)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def normalize_model_hashes(values):
    require(isinstance(values, dict) and values, 'missing model hash mapping')
    result = {key.split('_model/', 1)[-1]: value for key, value in values.items()}
    require(len(result) == len(values), 'duplicate normalized model hash keys')
    return result


def load_contract(stage):
    stage = validated_stage(stage)
    manifest = json.loads((stage / 'deployment_manifest.json').read_text())
    expected_id = manifest.pop('manifest_id')
    require(object_digest(manifest) == expected_id, 'deployment manifest content changed')
    manifest['manifest_id'] = expected_id
    require(manifest['stage'] == str(stage), 'stage identity mismatch')
    require(manifest['resources'] == RESOURCES and manifest['exclude'] == EXCLUDE, 'resource/exclusion contract drift')
    require(manifest['names'] == {str(k): v for k, v in NAMES.items()}, 'job name contract drift')
    require(manifest['checkpoint_count_per_arm'] == 1000, 'checkpoint cadence drift')
    for name, sha in manifest['files'].items():
        require(digest(stage / name) == sha, f'frozen deployment input changed: {name}')
    require(tracked_files(stage / 'source') == manifest['source_hashes'], 'complete source tree changed')
    source = json.loads((stage / 'source_contract.json').read_text())
    require(source.get('g5sc_model_bytes_identical') is True, 'native complete G5 model identity not proved')
    require(source['g5sc_model_hashes'] == manifest['g5sc_model_hashes'], 'source/model identity mismatch')
    for relative, sha in manifest['g5sc_model_hashes'].items():
        path = stage / 'source/src/tabicl' / relative
        require(digest(path) == sha, f'original G5 model module changed: {relative}')
    return manifest


def check_identity_receipt(stage, manifest):
    receipt = json.loads((stage / manifest['identity_receipt']).read_text())
    require(receipt.get('status') == 'PASS_T25_G5SC_NATIVE_IDENTITY', 'native identity/data/pinball smoke not PASS')
    checks = receipt.get('checks')
    require(isinstance(checks, dict) and checks and all(value == 'PASS' for value in checks.values()), 'identity smoke has incomplete checks')
    mixes = receipt.get('generator_mixes', {})
    require(set(mixes) >= {'0.0', '1.0'}, 'real T25 generator branches were not both tested')
    hashes = receipt.get('g5sc_model_hashes', receipt.get('model_hashes'))
    expected_hashes = normalize_model_hashes(manifest['g5sc_model_hashes'])
    require(normalize_model_hashes(hashes) == expected_hashes, 'smoke receipt belongs to another model source')
    require(receipt.get('source_manifest_sha256') == digest(Path(stage) / 'source.sha256'), 'smoke must bind the complete source manifest')
    require(receipt.get('source_contract_sha256') == digest(Path(stage) / 'source_contract.json'), 'smoke must bind this exact source assembly')
    require(receipt.get('assembled_python_identity_verified') is True, 'smoke did not verify complete trainer/prior source identity')
    require(receipt.get('regression_target_profile_sha256') == digest(Path(stage) / 'artifacts/profile.json'), 'smoke used another or unfrozen target profile')
    require(receipt.get('tested_passes') == [1, 3, 4], 'native pass-count equivalence was not fully tested')
    require(checks.get('actual_trainer_dataloader_muon_step_checkpoint_loop34_no_cross_entropy') == 'PASS', 'real trainer/optimizer/checkpoint smoke is required')
    for loop in ('3', '4'):
        step = receipt.get('trainer_steps', {}).get(loop, {})
        require(step.get('status') == 'PASS' and step.get('optimizer_step_calls') == 2, f'Loop{loop} real Muon steps were not proved')
    for branch in ('0.0', '1.0'):
        require(mixes[branch].get('status') == 'PASS', f'T25 generator mix={branch} did not pass')
    return receipt


def parser_arguments(stage, loop, checkpoint_dir, manifest):
    for key in manifest['sanitized_environment_keys']:
        os.environ.pop(key, None)
    os.environ.update(FROZEN_ENV)
    os.environ['SHARED_DEPTH_ICL_ENABLED'] = 'True'
    os.environ['SHARED_DEPTH_ICL_DATASET_CONDITIONED'] = 'True'
    os.environ['SHARED_DEPTH_ICL_NUM_PASSES'] = str(loop)
    os.environ['TRAINING_STAGE_MANIFEST_PATH'] = str(Path(stage) / 'source.sha256')
    os.environ['TRAINING_STAGE_MANIFEST_SHA256'] = digest(Path(stage) / 'source.sha256')
    sys.path[:0] = [str(stage), str(Path(stage) / 'source/src')]
    config_module = importlib.import_module('tabicl.train._train_config')
    expected_module = Path(stage) / 'source/src/tabicl/train/_train_config.py'
    require(Path(config_module.__file__).resolve() == expected_module.resolve(), 'parser imported outside isolated complete source')
    argv = training_arguments(stage, loop, checkpoint_dir)
    saved = json.loads((Path(stage) / f'loop{loop}.argv.json').read_text())
    require(saved == training_arguments(stage, loop, '__CHECKPOINT_DIR__'), 'frozen argv is not canonical')
    config = config_module.build_parser().parse_args(argv)
    adapter = importlib.import_module('tabicl.train._t25_regression_adapter')
    adapter.validate_regression_task(config)
    require(config.max_classes == 0 and config.num_quantiles == 999, 'not continuous 999-quantile regression')
    require(config.shared_depth_icl_enabled and config.shared_depth_icl_dataset_conditioned, 'native G5 gate is inactive')
    require(config.shared_depth_icl_num_passes == loop and config.icl_num_blocks == 12, 'shared-depth contract mismatch')
    require(config.col_num_blocks == 3 and config.row_num_blocks == 3 and config.row_num_cls == 4, 'G5 backbone shape changed')
    require(config.embed_dim * config.row_num_cls == 512 and config.ff_factor == 2, 'G5 ICL width/FFN changed')
    require(config.optimizer == 'muon' and config.muon_momentum == .95 and config.cautious_weight_decay, 'native Muon semantics changed')
    require(config.scheduler == 'cosine_warmup' and config.warmup_proportion == .02 and config.lr_floor == 0, 'schedule drift')
    require(config.amp and config.grad_scaler and not config.recompute and config.recompute_seq_len_threshold == 20000, 'precision/recompute drift')
    return argv, vars(config)


def validate(stage, loop, checkpoint_dir, account=None):
    manifest = load_contract(stage)
    if account is not None:
        require(account_name(account) == manifest['account'], 'explicit account differs from frozen deployment')
    receipt = check_identity_receipt(stage, manifest)
    argv, config = parser_arguments(stage, loop, checkpoint_dir, manifest)
    return manifest, receipt, argv, config


def check_complete(stage, loop, directory):
    manifest = load_contract(stage)
    directory = Path(directory).resolve()
    require(directory.parent == arm_root(loop, stage=stage), 'checkpoint completion path is outside this arm')
    steps = sorted(int(path.stem.split('-')[1]) for path in directory.glob('step-*.ckpt'))
    require(steps == list(range(25, 25001, 25)), 'expected all 1000 checkpoints at steps 25..25000')
    for rank in range(64):
        receipt = json.loads((directory / f'preflight-rank-{rank}.json').read_text())
        require(receipt.get('status') == 'PASS', f'rank {rank} full-model probe did not pass')
        require(receipt.get('passes') == loop and receipt.get('head_quantiles') == 999, f'rank {rank} probe architecture mismatch')
        expected_hashes = normalize_model_hashes(manifest['g5sc_model_hashes'])
        require(normalize_model_hashes(receipt.get('g5sc_model_hashes', receipt.get('model_hashes'))) == expected_hashes, f'rank {rank} probe model identity mismatch')
        require(receipt.get('source_manifest_sha256') == digest(Path(stage) / 'source.sha256')
                and receipt.get('source_contract_sha256') == digest(Path(stage) / 'source_contract.json'), f'rank {rank} probe full-source binding mismatch')
    write_new(directory / 'training.complete', {'status': 'COMPLETE', 'steps': 25000, 'checkpoints': 1000, 'passes': loop})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--passes', type=int, choices=(3, 4), required=True)
    parser.add_argument('--account')
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--exec-training', action='store_true')
    parser.add_argument('--check-complete', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.check_complete:
        require(args.checkpoint_dir is not None, '--check-complete requires --checkpoint-dir')
        check_complete(args.stage, args.passes, args.checkpoint_dir)
        return
    directory = args.checkpoint_dir or Path('__CHECKPOINT_DIR__')
    manifest, smoke, argv, config = validate(args.stage, args.passes, directory, args.account)
    payload = {'status': 'PASS_FULL_G5SC_T25_LAUNCH', 'passes': args.passes,
               'manifest_id': manifest['manifest_id'], 'source_identity_sha256': manifest['source_identity_sha256'],
               'native_identity_receipt_sha256': digest(args.stage / manifest['identity_receipt']), 'argv': argv,
               'parsed_config': config}
    if args.receipt:
        write_new(args.receipt, payload)
    if args.exec_training:
        require(args.checkpoint_dir is not None and 'RANK' in os.environ, 'training must run inside torchrun')
        require(int(os.environ['WORLD_SIZE']) == 64, 'formal world size must be 64')
        require(args.checkpoint_dir.resolve().parent == arm_root(args.passes, stage=args.stage), 'training checkpoint path is outside fresh arm root')
        write_new(args.checkpoint_dir / f'launch-rank-{os.environ["RANK"]}.json', payload)
        sys.argv = ['tabicl.train', *argv]
        runpy.run_module('tabicl.train', run_name='__main__')
    else:
        print(json.dumps(payload, default=str))


if __name__ == '__main__':
    main()
