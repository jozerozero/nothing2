"""T25 task wiring; the complete G5SC model, Muon and scheduler stay unchanged."""
import os

import torch


def pinball_loss(pred, query_targets):
    """Exact T25 mean pinball loss, without sorting or auxiliary objectives."""
    if pred.ndim != 3 or query_targets.shape != pred.shape[:-1]:
        raise ValueError(f'Expected [B,Q,K] and [B,Q], got {pred.shape}, {query_targets.shape}')
    alphas = torch.linspace(0.0, 1.0, pred.shape[-1] + 2,
                            device=pred.device, dtype=pred.dtype)[1:-1].view(1, 1, -1)
    errors = query_targets.unsqueeze(-1) - pred
    return torch.maximum(alphas * errors, (alphas - 1) * errors).mean()


def validate_regression_task(config):
    """Fail closed on classification-only additions or changed backbone flags."""
    if config.regression_method != 'quantile' or config.num_quantiles != 999:
        raise ValueError('This experiment requires T25 quantile regression with 999 outputs')
    if config.max_classes != 0:
        raise ValueError('Pass --max_classes 0 to select both native continuous-label encoders')
    if config.bias_free_ln:
        raise ValueError('G5SC LayerNorm includes bias; --bias_free_ln must be False')
    for flag in ('shared_depth_icl_enabled', 'shared_depth_icl_dataset_conditioned'):
        if not bool(getattr(config, flag, False)):
            raise ValueError(f'{flag} must be enabled')
    if config.shared_depth_icl_num_passes not in (3, 4) or config.shared_depth_icl_rho != 1.0:
        raise ValueError('Native G5SC requires 3 or 4 passes and rho=1')
    for flag in ('swiglu_enabled', 'swiglu_conditioned', 'attention_gate_enabled',
                 'layer_gate_enabled', 'schema_expert_enabled', 'schema_film_enabled',
                 'function_tokens_enabled', 'cr2_shared_refinement_enabled', 'qk_pds_attention_enabled',
                 'cls8_pooled_enabled', 'cls8_width_enabled', 'plasticity_reg_enabled',
                 'plasticity_proto_enabled', 'plasticity_attn_enabled', 'continual_bp_enabled',
                 'use_pseudo_ssmax_thinking', 'freeze_col', 'freeze_row', 'freeze_icl'):
        if bool(getattr(config, flag, False)):
            raise ValueError(f'{flag} is outside the native G5SC/T25 contract')
    for flag in ('stage3_kd_weight', 'stage3_l2sp_weight', 'stage3_anchor_pullback',
                 'stage3_ema_decay', 'firewall_kd_weight', 'train_only_affine_symmetry_weight',
                 'train_only_knn_mixup_weight', 'stage3_minority_kd_weight',
                 'balanced_ce_weight', 'support_balanced_ce_weight', 'support_prior_kl_weight',
                 'support_prior_floor_weight', 'support_margin_weight'):
        if float(getattr(config, flag, 0.0) or 0.0) != 0.0:
            raise ValueError(f'{flag} must be zero for the sole T25 pinball objective')
    if config.prior_type != 'graph_scm' or config.prior_dir is not None:
        raise ValueError('T25 requires online graph_scm data generation')
    if config.regression_target_prior != 'rw_sample50' or config.regression_target_mix_probability != 0.25:
        raise ValueError('T25 RW25 requires rw_sample50 and mix_probability=0.25')
    if not config.regression_target_profile:
        raise ValueError('The frozen official-train-only anonymous target profile is required')
    for flag in ('REGRESSION_TL_LENGTH_CURRICULUM_ENABLED', 'REGRESSION_CROSS_TABLE_E4_ENABLED',
                 'CROSS_TABLE_ENABLED', 'SYNTHETIC96_RW_DGP_ENABLED'):
        if os.environ.get(flag, 'false').lower() == 'true':
            raise ValueError(f'{flag} must be false for T25 data')
    if os.environ.get('CROSS_TABLE_ARM', '').upper() in ('E5', 'E6'):
        raise ValueError('E5/E6 model-side cross-table features are forbidden')


def build_t25_prior(config):
    """Return the real T25 PriorDataset from the assembled parser config.

    Tests may override mix_probability=0/1 and reduce shapes. Formal training
    validates RW25 separately. Inner n_jobs MUST stay 1: DataLoader workers
    already parallelize generation and cannot create another process pool.
    """
    from tabicl.prior import PriorDataset
    from tabicl.prior.graph_lib._config import PriorConfig
    return PriorDataset(
        regression=True, batch_size=config.batch_size,
        batch_size_per_gp=config.batch_size_per_gp,
        min_features=config.min_features, max_features=config.max_features,
        max_classes=0, min_seq_len=config.min_seq_len,
        max_seq_len=config.max_seq_len, log_seq_len=config.log_seq_len,
        log_n_features=config.log_n_features, seq_len_per_gp=config.seq_len_per_gp,
        min_train_size=config.min_train_size, max_train_size=config.max_train_size,
        replay_small=config.replay_small, prior_type='graph_scm',
        config=PriorConfig.from_args(config), device=config.prior_device,
        n_jobs=1, num_threads_per_generate=config.prior_num_threads_per_generate,
    )


def run_regression_micro_batch(trainer, micro_batch, micro_batch_idx, num_micro_batches, timings=None):
    """T25 query supervision inside G5SC's existing accumulated update loop."""
    from tabicl.train._run import NonFiniteMicroBatchError
    micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch[:5]
    with trainer._timed_phase(timings, 'micro_cpu_prepare'):
        seq_len, train_size = trainer.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = trainer.align_micro_batch(micro_X, micro_y, micro_d, seq_len)
    with trainer._timed_phase(timings, 'micro_to_device'):
        non_blocking = bool(getattr(trainer.config, 'data_transfer_non_blocking', False))
        micro_X = micro_X.to(trainer.config.device, non_blocking=non_blocking)
        micro_y = micro_y.to(trainer.config.device, non_blocking=non_blocking)
        micro_d = micro_d.to(trainer.config.device, non_blocking=non_blocking)
    input_issue = trainer._local_micro_batch_issue(micro_X, micro_y)
    if trainer._sync_nonfinite_issue(input_issue):
        raise NonFiniteMicroBatchError(input_issue or 'another rank reported non-finite regression data')
    y_train, y_test = micro_y[:, :train_size], micro_y[:, train_size:]
    if trainer.ddp:
        trainer.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1
    model_d = None if getattr(trainer.raw_model, 'col_feature_group', False) else micro_d
    with trainer._timed_phase(timings, 'micro_forward'):
        with trainer.amp_ctx:
            pred = trainer.model(micro_X, y_train, model_d)
            if pred.shape != (*y_test.shape, trainer.config.num_quantiles):
                raise ValueError(f'Invalid regression output shape {pred.shape}')
            loss = pinball_loss(pred, y_test)
    issue = '; '.join(reason for reason in (
        trainer._nonfinite_tensor_reason('predicted_quantiles', pred),
        trainer._nonfinite_tensor_reason('query_targets', y_test),
        trainer._nonfinite_tensor_reason('pinball_loss', loss),
    ) if reason)
    if trainer._sync_nonfinite_issue(issue or None):
        raise NonFiniteMicroBatchError(issue or 'another rank reported non-finite regression output')
    with trainer._timed_phase(timings, 'micro_backward'):
        trainer.scaler.scale(loss / num_micro_batches).backward()
    if not trainer._train_metrics_enabled():
        return {}
    with torch.no_grad(), trainer._timed_phase(timings, 'micro_metrics'):
        results = {'pinball': float(loss.item()) / num_micro_batches,
                   'mse': float((pred.mean(dim=-1) - y_test).square().mean().item()) / num_micro_batches}
        results.update({name: value / num_micro_batches
                        for name, value in trainer._model_structure_metrics().items()})
    return results
