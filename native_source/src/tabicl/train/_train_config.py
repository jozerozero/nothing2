"""Define argument parser for TabICL training."""

import argparse
import os

from tabicl.prior.graph_lib._config import PriorConfig


def str2bool(value):
    return value.lower() == "true"


def feature_group_type(value):
    value_lower = value.lower()
    if value_lower == "false":
        return False
    if value_lower == "true":
        return True
    if value_lower in {"same", "valid"}:
        return value_lower
    raise argparse.ArgumentTypeError("Feature grouping must be one of: False, True, same, valid")


def train_size_type(value):
    """Custom type function to handle both int and float train sizes."""
    value = float(value)
    if 0 < value < 1:
        return value
    elif value.is_integer():
        return int(value)
    else:
        raise argparse.ArgumentTypeError(
            "Train size must be either an integer (absolute position) "
            "or a float between 0 and 1 (ratio of sequence length)."
        )


def optional_int_type(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return int(value)


def build_parser():
    """Build an argument parser with all TabICL training arguments.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all training, model, and
        checkpoint arguments.
    """
    parser = argparse.ArgumentParser()

    ###########################################################################
    ###### Wandb Config #######################################################
    ###########################################################################
    parser.add_argument("--wandb_log", default=False, type=str2bool, help="Log results using wandb")
    parser.add_argument("--wandb_project", type=str, default="TabICL", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb run ID")
    parser.add_argument("--wandb_dir", type=str, default=None, help="Wandb logging directory")
    parser.add_argument(
        "--wandb_mode", default="offline", type=str, help="Wandb logging mode: online, offline, or disabled"
    )

    ###########################################################################
    ###### Training Config ####################################################
    ###########################################################################
    parser.add_argument("--device", default="cuda", type=str, help="Device for training: cpu, cuda, cuda:0")
    parser.add_argument(
        "--dtype", default="float32", type=str, help="Data type (supported for float16, float32) used for training"
    )
    parser.add_argument("--np_seed", type=int, default=42, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch")
    parser.add_argument(
        "--prior_loader_seed",
        type=int,
        default=int(os.environ.get("PRIOR_LOADER_SEED", "42")),
        help="Independent DataLoader worker base seed; paired arms must use the same value.",
    )
    parser.add_argument(
        "--paired_batch_audit_steps",
        type=int,
        default=0,
        help=(
            "For a fresh paired run, write an architecture-independent data fingerprint "
            "from every DDP rank for the first N batches. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--paired_batch_audit_dir",
        type=str,
        default=None,
        help=(
            "Directory for per-rank paired-batch audit JSONL files. Defaults to "
            "checkpoint_dir/paired_batch_audit."
        ),
    )
    parser.add_argument(
        "--paired_batch_audit_sample_values",
        type=int,
        default=4096,
        help="Maximum deterministic tensor values included per tensor in each paired-batch fingerprint.",
    )
    parser.add_argument(
        "--strict_resume_training_contract",
        default=True,
        type=str2bool,
        help="Reject optimizer resumes whose saved architecture/optimizer/data-seed contract differs.",
    )
    parser.add_argument(
        "--strict_training_stage_manifest",
        default=False,
        type=str2bool,
        help="Require TRAINING_STAGE_MANIFEST_SHA256 to be exported by an immutable launcher.",
    )
    parser.add_argument(
        "--allow_nonexact_prior_resume",
        default=False,
        type=str2bool,
        help=(
            "Explicitly allow optimizer resume with an online stochastic prior. Such a resume restarts "
            "worker RNG/counters and is not an exact data continuation."
        ),
    )
    parser.add_argument("--max_steps", type=int, default=60000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--micro_batch_size", type=int, default=8, help="Size of micro-batches for gradient accumulation"
    )
    parser.add_argument(
        "--skip_nonfinite_batches",
        default=True,
        type=str2bool,
        help="If True, skip micro-batches whose inputs, logits, loss, or labels are non-finite/invalid.",
    )
    parser.add_argument(
        "--abort_on_nonfinite_batch",
        default=False,
        type=str2bool,
        help="If True, raise immediately instead of skipping when a non-finite micro-batch is detected.",
    )
    parser.add_argument(
        "--nonfinite_max_bad_micro_batch_fraction",
        type=float,
        default=0.1,
        help="If the fraction of non-finite micro-batches in a batch exceeds this value, skip the optimizer update.",
    )
    parser.add_argument(
        "--nonfinite_check_every",
        type=int,
        default=1,
        help="Run non-finite/invalid tensor diagnostics every N steps. Set 0 to disable these diagnostics.",
    )
    parser.add_argument(
        "--nonfinite_check_until_step",
        type=int,
        default=-1,
        help="Last training step to run non-finite diagnostics. Set -1 to keep checking for the full run.",
    )
    parser.add_argument(
        "--error_if_nonfinite_grad",
        default=True,
        type=str2bool,
        help="If True, make gradient norm clipping fail on non-finite total grad norm and skip the optimizer update.",
    )
    parser.add_argument(
        "--log_grad_norm_every",
        type=int,
        default=0,
        help="Print per-top-level-module gradient norms every N steps on rank 0. Set 0 to disable.",
    )
    parser.add_argument(
        "--train_metrics_every",
        type=int,
        default=1,
        help="Collect synchronized training metrics such as CE, accuracy, and total grad norm every N steps.",
    )
    parser.add_argument(
        "--train_metrics_until_step",
        type=int,
        default=-1,
        help="Last training step to collect per-step training metrics. Set -1 to collect for the full run.",
    )
    parser.add_argument(
        "--speed_trace_every",
        type=int,
        default=0,
        help="Write rank-0 per-step speed diagnostics to JSONL every N steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--speed_trace_until_step",
        type=int,
        default=-1,
        help="Last training step to write speed diagnostics. Set -1 to keep tracing for the full run.",
    )
    parser.add_argument(
        "--speed_trace_path",
        type=str,
        default=None,
        help="Optional path for rank-0 JSONL speed diagnostics.",
    )
    parser.add_argument(
        "--float32_matmul_precision",
        default="",
        type=str,
        choices=["", "highest", "high", "medium"],
        help=(
            "Optional torch.set_float32_matmul_precision value. Empty preserves the PyTorch process default. "
            "This keeps float32 tensors but may allow faster internal matmul kernels depending on the value."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        default=True,
        type=str2bool,
        help=(
            "If True, allow CUDA matmul/cuDNN TF32 kernels for float32 tensors. "
            "Set False for stricter FP32 arithmetic at the cost of speed."
        ),
    )
    parser.add_argument(
        "--sdpa_backends",
        default="",
        type=str,
        help=(
            "Optional comma-separated SDPA backend allowlist: flash, mem_efficient, math, cudnn, or all. "
            "Empty preserves PyTorch's default backend selection."
        ),
    )
    parser.add_argument(
        "--ddp_init_sync",
        default=True,
        type=str2bool,
        help=(
            "If True, DistributedDataParallel verifies and broadcasts module states during construction. "
            "Set False only when every rank initializes the same model weights."
        ),
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        default=False,
        type=str2bool,
        help="Pass find_unused_parameters to DistributedDataParallel.",
    )
    parser.add_argument(
        "--ddp_bucket_cap_mb",
        type=float,
        default=0.0,
        help="If > 0, pass bucket_cap_mb to DistributedDataParallel.",
    )
    parser.add_argument(
        "--ddp_gradient_as_bucket_view",
        default=False,
        type=str2bool,
        help="If True, pass gradient_as_bucket_view=True to DistributedDataParallel when supported.",
    )
    parser.add_argument(
        "--ddp_static_graph",
        default=False,
        type=str2bool,
        help="If True, pass static_graph=True to DistributedDataParallel when supported.",
    )
    parser.add_argument(
        "--debug_timing",
        default=False,
        type=str2bool,
        help="If True, print per-rank timing checkpoints for early training steps.",
    )
    parser.add_argument(
        "--debug_timing_steps",
        type=int,
        default=2,
        help="Number of initial steps covered by debug_timing logs.",
    )
    parser.add_argument(
        "--profile_timing",
        default=False,
        type=str2bool,
        help="If True, add rank-0 fine-grained phase timing fields to training metrics.",
    )
    parser.add_argument(
        "--profile_timing_every",
        type=int,
        default=1,
        help="Collect profile timing every N steps when --profile_timing is enabled.",
    )
    parser.add_argument(
        "--profile_timing_until_step",
        type=int,
        default=200,
        help="Last step to collect profile timing. Set -1 to keep profiling for the full run.",
    )
    parser.add_argument(
        "--profile_timing_sync_cuda",
        default=True,
        type=str2bool,
        help="If True, synchronize CUDA around profiled phases for more accurate timings.",
    )

    # Optimization Config
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--lr_floor",
        type=float,
        default=0.0,
        help="Absolute learning-rate floor applied after warmup and scheduler scaling. Set 0 to disable.",
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine_warmup", help="Learning rate scheduler: see optim.py for options."
    )
    parser.add_argument(
        "--warmup_proportion",
        type=float,
        default=0.2,
        help="The proportion of total steps over which we warmup."
        "If this value is set to -1, we warmup for a fixed number of steps.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="The number of steps over which we warm up. Only used when warmup_proportion is set to -1",
    )
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="If > 0, clip gradients.")
    parser.add_argument(
        "--gradient_clip_foreach",
        default="auto",
        choices=["auto", "true", "false", "True", "False"],
        help="Foreach mode for torch.nn.utils.clip_grad_norm_. Use auto to keep PyTorch's default.",
    )
    parser.add_argument(
        "--gradient_clip_error_sync_every",
        type=int,
        default=1,
        help="Synchronize gradient clipping failures across DDP ranks every N steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--gradient_clip_error_sync_until_step",
        type=int,
        default=-1,
        help="Last step to synchronize gradient clipping failures. Set -1 to synchronize for the full run.",
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay / L2 regularization penalty")
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing epsilon for supervised cross-entropy losses. Set 0 to disable.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "muon"],
        help="Optimizer to use for pretraining.",
    )
    parser.add_argument(
        "--cautious_weight_decay",
        default=False,
        type=str2bool,
        help="For Muon, apply decoupled weight decay only where update and parameter have the same sign.",
    )
    parser.add_argument("--muon_momentum", type=float, default=0.95, help="Momentum coefficient for Muon.")
    parser.add_argument(
        "--muon_ns_steps",
        type=int,
        default=5,
        help="Newton-Schulz orthogonalization iterations for Muon matrix updates.",
    )
    parser.add_argument(
        "--muon_group_by_lr",
        default=False,
        type=str2bool,
        help="For Muon, place parameters with identical computed learning rates in the same optimizer group.",
    )
    parser.add_argument(
        "--fast_cosine_scheduler",
        default=False,
        type=str2bool,
        help=(
            "For scheduler=cosine_warmup, use a scheduler that computes the cosine factor once per step "
            "instead of once per optimizer group."
        ),
    )
    parser.add_argument(
        "--plasticity_reg_enabled",
        default=False,
        type=str2bool,
        help=(
            "Enable lightweight table-level representation health regularization. "
            "Cosine and variance terms pool row representations per table first, then act across tables."
        ),
    )
    parser.add_argument(
        "--plasticity_reg_start_step",
        type=int,
        default=10000,
        help="Training step at which plasticity regularization starts. Set 0 to enable from the beginning.",
    )
    parser.add_argument(
        "--plasticity_reg_ramp_steps",
        type=int,
        default=2000,
        help="Number of steps used to linearly ramp plasticity regularization to full weight.",
    )
    parser.add_argument(
        "--plasticity_reg_end_step",
        type=int,
        default=-1,
        help="If >=0, disable table-level plasticity regularization at and after this training step.",
    )
    parser.add_argument(
        "--plasticity_reg_span",
        type=str,
        default="query",
        choices=["query", "support", "full"],
        help="Rows used to build each table representation before table-level plasticity losses.",
    )
    parser.add_argument(
        "--plasticity_reg_source",
        type=str,
        default="row",
        choices=["row", "icl"],
        help="Representation source for table-level plasticity losses: row-interactor output or selected ICL layers.",
    )
    parser.add_argument(
        "--plasticity_reg_layers",
        type=str,
        default="8,9,10",
        help="Comma-separated zero-based ICL block indices used when plasticity_reg_source=icl.",
    )
    parser.add_argument(
        "--plasticity_reg_table_sample_rows",
        type=int,
        default=0,
        help="If >0, deterministically subsample this many selected rows per table before pooling.",
    )
    parser.add_argument(
        "--plasticity_cos_weight",
        type=float,
        default=0.0,
        help="Weight for table-level off-diagonal cosine-squared anisotropy loss.",
    )
    parser.add_argument(
        "--plasticity_var_weight",
        type=float,
        default=0.0,
        help="Weight for table-level dimension-wise variance floor loss.",
    )
    parser.add_argument(
        "--plasticity_var_floor",
        type=float,
        default=0.005,
        help="Minimum desired variance per hidden dimension for the table-level variance floor.",
    )
    parser.add_argument(
        "--plasticity_conf_weight",
        type=float,
        default=0.0,
        help="Weight for query-logit confidence penalty.",
    )
    parser.add_argument(
        "--plasticity_conf_threshold",
        type=float,
        default=0.80,
        help="Confidence threshold above which max softmax probability is penalized.",
    )
    parser.add_argument(
        "--plasticity_entropy_floor_weight",
        type=float,
        default=0.0,
        help="Weight for query-logit normalized entropy floor penalty.",
    )
    parser.add_argument(
        "--plasticity_entropy_floor",
        type=float,
        default=0.40,
        help="Minimum normalized entropy over support-present classes before applying the entropy floor penalty.",
    )
    parser.add_argument(
        "--plasticity_cka_weight",
        type=float,
        default=0.0,
        help="Weight for table-level linear CKA anchor loss against a frozen reference checkpoint.",
    )
    parser.add_argument(
        "--plasticity_cka_reference_checkpoint",
        type=str,
        default="",
        help="Frozen checkpoint used as the reference for plasticity CKA anchor loss.",
    )
    parser.add_argument(
        "--plasticity_cka_eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon for linear CKA anchor loss.",
    )
    parser.add_argument(
        "--plasticity_low_high_cka_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for same-model low-to-high CKA regularization. "
            "Low ICL layers are pooled per table and used as stop-gradient teachers for high ICL layers."
        ),
    )
    parser.add_argument(
        "--plasticity_low_high_teacher_layers",
        type=str,
        default="5,6",
        help="Comma- or semicolon-separated ICL layers used as low-level CKA teachers.",
    )
    parser.add_argument(
        "--plasticity_low_high_student_layers",
        type=str,
        default="8,9,10",
        help="Comma- or semicolon-separated ICL layers regularized to match low-level CKA teachers.",
    )
    parser.add_argument(
        "--plasticity_low_high_detach_teacher",
        default=True,
        type=str2bool,
        help="Stop gradients through low-level teacher representations in low-to-high CKA.",
    )
    parser.add_argument(
        "--plasticity_proto_enabled",
        default=False,
        type=str2bool,
        help=(
            "Enable support-prototype anti-collapse regularization on selected ICL layers. "
            "This keeps query rows from concentrating on too few support-row prototypes."
        ),
    )
    parser.add_argument(
        "--plasticity_proto_layers",
        type=str,
        default="8,9,10",
        help="Comma- or semicolon-separated ICL layers used for support-prototype anti-collapse loss.",
    )
    parser.add_argument(
        "--plasticity_proto_start_step",
        type=int,
        default=12000,
        help="Training step at which support-prototype anti-collapse regularization starts.",
    )
    parser.add_argument(
        "--plasticity_proto_ramp_steps",
        type=int,
        default=2000,
        help="Number of steps used to linearly ramp support-prototype regularization to full weight.",
    )
    parser.add_argument(
        "--plasticity_proto_end_step",
        type=int,
        default=-1,
        help="If >=0, disable support-prototype regularization at and after this training step.",
    )
    parser.add_argument(
        "--plasticity_proto_tau",
        type=float,
        default=0.07,
        help="Temperature for cosine query-to-support prototype assignment.",
    )
    parser.add_argument(
        "--plasticity_proto_entropy_min",
        type=float,
        default=0.88,
        help="Minimum normalized query-to-support assignment entropy before applying entropy penalty.",
    )
    parser.add_argument(
        "--plasticity_proto_top1_limit",
        type=float,
        default=0.08,
        help="Maximum allowed single-support assignment mass before applying top-1 penalty.",
    )
    parser.add_argument(
        "--plasticity_proto_entropy_weight",
        type=float,
        default=0.0,
        help="Weight for support-prototype assignment entropy floor loss.",
    )
    parser.add_argument(
        "--plasticity_proto_top1_weight",
        type=float,
        default=0.0,
        help="Weight for support-prototype top-1 concentration loss.",
    )
    parser.add_argument(
        "--plasticity_proto_usage_weight",
        type=float,
        default=0.0,
        help="Weight for support-prototype usage concentration loss.",
    )
    parser.add_argument(
        "--plasticity_proto_query_sample_rows",
        type=int,
        default=128,
        help="If >0, deterministically subsample this many query rows per table for prototype loss.",
    )
    parser.add_argument(
        "--plasticity_proto_support_sample_rows",
        type=int,
        default=512,
        help="If >0, deterministically subsample this many support rows per table for prototype loss.",
    )
    parser.add_argument(
        "--plasticity_attn_enabled",
        default=False,
        type=str2bool,
        help=(
            "Enable direct query-to-support attention anti-collapse regularization on selected ICL layers. "
            "Unlike prototype loss, this recomputes each selected block's actual Q/K attention map."
        ),
    )
    parser.add_argument(
        "--plasticity_attn_layers",
        type=str,
        default="8,9,10",
        help="Comma- or semicolon-separated ICL layers used for direct attention anti-collapse loss.",
    )
    parser.add_argument(
        "--plasticity_attn_start_step",
        type=int,
        default=12000,
        help="Training step at which direct attention anti-collapse regularization starts.",
    )
    parser.add_argument(
        "--plasticity_attn_ramp_steps",
        type=int,
        default=2000,
        help="Number of steps used to linearly ramp direct attention regularization to full weight.",
    )
    parser.add_argument(
        "--plasticity_attn_end_step",
        type=int,
        default=18000,
        help="If >=0, disable direct attention regularization at and after this training step.",
    )
    parser.add_argument(
        "--plasticity_attn_entropy_min",
        type=float,
        default=0.60,
        help="Minimum normalized query-to-support attention entropy before applying entropy penalty.",
    )
    parser.add_argument(
        "--plasticity_attn_top1_limit",
        type=float,
        default=0.22,
        help="Maximum allowed single-support attention mass before applying top-1 penalty.",
    )
    parser.add_argument(
        "--plasticity_attn_entropy_weight",
        type=float,
        default=0.0,
        help="Weight for direct attention entropy floor loss.",
    )
    parser.add_argument(
        "--plasticity_attn_top1_weight",
        type=float,
        default=0.0,
        help="Weight for direct attention top-1 concentration loss.",
    )
    parser.add_argument(
        "--plasticity_attn_usage_weight",
        type=float,
        default=0.0,
        help="Weight for direct attention support-usage concentration loss.",
    )
    parser.add_argument(
        "--plasticity_attn_query_sample_rows",
        type=int,
        default=128,
        help="If >0, deterministically subsample this many query rows per table for attention loss.",
    )
    parser.add_argument(
        "--plasticity_attn_support_sample_rows",
        type=int,
        default=512,
        help="If >0, deterministically subsample this many support rows per table for attention loss.",
    )
    parser.add_argument(
        "--qassmax_cap_enabled",
        default=False,
        type=str2bool,
        help="After model construction, lower QASSMax clamp values on selected ICL layers.",
    )
    parser.add_argument(
        "--qassmax_cap_layers",
        type=str,
        default="7,8,9,10",
        help="Comma- or semicolon-separated ICL layers whose QASSMax clamp values are overridden.",
    )
    parser.add_argument(
        "--qassmax_cap_base_scale",
        type=float,
        default=16.0,
        help="max_abs_base_scale value used when qassmax_cap_enabled=True.",
    )
    parser.add_argument(
        "--qassmax_cap_scale",
        type=float,
        default=16.0,
        help="max_abs_scale value used when qassmax_cap_enabled=True.",
    )
    parser.add_argument(
        "--qassmax_cap_query_logit",
        type=float,
        default=8.0,
        help="max_abs_query_logit value used when qassmax_cap_enabled=True.",
    )
    parser.add_argument(
        "--late_icl_freeze_enabled",
        default=False,
        type=str2bool,
        help=(
            "Freeze updates to selected late ICL blocks after late_icl_freeze_start_step. "
            "Gradients are cleared after backward for DDP-safe update freezing."
        ),
    )
    parser.add_argument(
        "--late_icl_freeze_start_step",
        type=int,
        default=18000,
        help="Optimizer step from which late ICL block updates are frozen.",
    )
    parser.add_argument(
        "--late_icl_freeze_layers",
        type=str,
        default="7,8,9,10",
        help="Comma- or semicolon-separated ICL block indices whose updates are frozen late in training.",
    )
    parser.add_argument(
        "--continual_bp_enabled",
        default=False,
        type=str2bool,
        help="Enable conservative continual backpropagation on selected ICL FFN hidden channels.",
    )
    parser.add_argument(
        "--continual_bp_layers",
        type=str,
        default="8,9,10",
        help="Comma- or semicolon-separated zero-based ICL block indices for continual BP.",
    )
    parser.add_argument(
        "--continual_bp_target",
        type=str,
        default="ffn",
        choices=["ffn"],
        help="Module family targeted by continual BP. Currently only ICL FFN hidden channels are supported.",
    )
    parser.add_argument(
        "--continual_bp_start_step",
        type=int,
        default=12000,
        help="First optimizer step at which continual BP replacement events may run.",
    )
    parser.add_argument(
        "--continual_bp_maturity_steps",
        type=int,
        default=5000,
        help="Minimum channel age before it is eligible for continual BP replacement.",
    )
    parser.add_argument(
        "--continual_bp_replace_every",
        type=int,
        default=200,
        help="Run continual BP replacement selection every N optimizer steps.",
    )
    parser.add_argument(
        "--continual_bp_replacement_rate",
        type=float,
        default=1e-6,
        help="Per-step FFN-channel replacement rate accumulated between replacement events.",
    )
    parser.add_argument(
        "--continual_bp_max_replace_per_event",
        type=int,
        default=1,
        help="Maximum number of FFN channels to replace per layer at one replacement event.",
    )
    parser.add_argument(
        "--continual_bp_utility_decay",
        type=float,
        default=0.99,
        help="EMA decay for continual BP utility estimates.",
    )
    parser.add_argument(
        "--continual_bp_reset_outgoing_zero",
        default=True,
        type=str2bool,
        help="If True, zero the outgoing linear2 column for each reset FFN channel.",
    )
    parser.add_argument(
        "--continual_bp_reset_incoming",
        type=str,
        default="kaiming_uniform",
        choices=["kaiming_uniform"],
        help="Initialization used for reset linear1 rows.",
    )
    parser.add_argument(
        "--cosine_num_cycles",
        type=int,
        default=1,
        help="Number of hard restarts for cosine schedule. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument(
        "--cosine_amplitude_decay",
        type=float,
        default=1.0,
        help="Amplitude scaling factor per cycle. Only used when scheduler is cosine_with_restarts",
    )
    parser.add_argument("--cosine_lr_end", type=float, default=0, help="Final learning rate for cosine_with_restarts")
    parser.add_argument(
        "--poly_decay_lr_end", type=float, default=1e-7, help="Final learning rate for polynomial decay scheduler"
    )
    parser.add_argument(
        "--poly_decay_power", type=float, default=1.0, help="Power factor for polynomial decay scheduler"
    )

    # Prior Dataset Config
    parser.add_argument(
        "--prior_dir",
        type=str,
        default=None,
        help="If set, load pre-generated prior datasets directly from this directory on disk instead of generating them on the fly.",
    )
    parser.add_argument(
        "--load_prior_start",
        type=int,
        default=0,
        help="Batch index to start loading from pre-generated prior data. Only used when prior_dir is set.",
    )
    parser.add_argument(
        "--delete_after_load",
        default=False,
        type=str2bool,
        help="Delete prior data after loading. Only used when prior_dir is set.",
    )
    parser.add_argument("--batch_size_per_gp", type=int, default=4, help="Batch size per group")
    parser.add_argument("--min_features", type=int, default=5, help="The minimum number of features")
    parser.add_argument("--max_features", type=int, default=100, help="The maximum number of features")
    parser.add_argument("--max_classes", type=int, default=10, help="The maximum number of classes")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum samples per dataset")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum samples per dataset")
    parser.add_argument(
        "--log_seq_len",
        default=False,
        type=str2bool,
        help="If True, sample sequence length from log-uniform distribution between min_seq_len and max_seq_len",
    )
    parser.add_argument(
        "--log_n_features",
        default=False,
        type=str2bool,
        help="If True, sample graph_scm feature counts from a log-uniform distribution.",
    )
    parser.add_argument(
        "--graph_meta_shape_match_ratio",
        type=float,
        default=0.0,
        help=(
            "For graph_scm, fraction of subgroups whose class-count/feature-count "
            "shape is sampled from --graph_meta_shape_joint_bin_weights. The "
            "remaining fraction keeps the exact official uniform shape sampler."
        ),
    )
    parser.add_argument(
        "--graph_meta_shape_joint_bin_weights",
        type=str,
        default=None,
        help=(
            "For graph_scm matched subgroups, 25 class-major weights for the joint "
            "bins class=[2,3,4-5,6-10,>10] x "
            "features=[2-5,6-10,11-20,21-50,51-100]."
        ),
    )
    parser.add_argument(
        "--seq_len_per_gp",
        default=False,
        type=str2bool,
        help="If True, sample sequence length independently for each group",
    )
    parser.add_argument(
        "--min_train_size",
        type=train_size_type,
        default=0.1,
        help="Starting position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--max_train_size",
        type=train_size_type,
        default=0.9,
        help="Ending position/ratio for train/test split. If int, absolute position. If float (0-1), ratio of seq_len",
    )
    parser.add_argument(
        "--replay_small",
        default=False,
        type=str2bool,
        help="If True, occasionally sample smaller sequence lengths to ensure model robustness on smaller datasets",
    )
    parser.add_argument(
        "--prior_type",
        default="mix_scm",
        type=str,
        help=(
            "Prior type: dummy, mlp_scm, tree_scm, mix_scm, tabiclv2_cls, "
            "talent_single_mix, tabiclv2_cls_talent_mix, "
            "tabiclv2_cls_protected_batch_mix, graph_scm, hybrid178"
        ),
    )
    parser.add_argument(
        "--talent_mix_ratio",
        type=float,
        default=0.0,
        help="Batch-level probability of drawing from TALENT-sim when prior_type=tabiclv2_cls_talent_mix.",
    )
    parser.add_argument(
        "--talent_mix_source",
        type=str,
        default="default51",
        help=(
            "TALENT-sim source: default51, protected_generated, protected_restore, "
            "protected_weighted, or shape_weighted."
        ),
    )
    parser.add_argument(
        "--talent_mix_dataset_path",
        type=str,
        default=None,
        help="Optional path to the packaged TALENT single-mix dataset.py.",
    )
    parser.add_argument(
        "--protected_batch_mix_ratio",
        type=float,
        default=0.0,
        help="Within-batch fraction replaced by protected auxiliary tasks when prior_type=tabiclv2_cls_protected_batch_mix.",
    )
    parser.add_argument(
        "--protected_batch_mix_source",
        type=str,
        default="gt",
        help="Protected batch-internal source: gt, aug/protected_like, or p05_hard.",
    )
    parser.add_argument(
        "--protected_batch_mix_cache_dirs",
        type=str,
        default=None,
        help="Optional comma/semicolon/colon-separated protected npz cache directories.",
    )
    parser.add_argument(
        "--protected_batch_mix_dataset_names",
        type=str,
        default=None,
        help="Optional comma/semicolon-separated protected dataset names for gt/profile augmentation.",
    )
    parser.add_argument(
        "--hybrid178_data_root",
        type=str,
        default=None,
        help=(
            "Root containing the 178 dataset directories. Required for prior_type=hybrid178. "
            "Only train arrays are read by default."
        ),
    )
    parser.add_argument(
        "--hybrid178_include_val",
        default=False,
        type=str2bool,
        help="Also allow validation rows in hybrid178 profiles. Test rows are never allowed.",
    )
    parser.add_argument(
        "--hybrid178_profile_ratio",
        type=float,
        default=0.20,
        help="Within-batch ratio for official GraphSCM plus empirical marginal transport.",
    )
    parser.add_argument(
        "--hybrid178_copula_ratio",
        type=float,
        default=0.05,
        help="Within-batch ratio for class-conditional shrinkage rank-Gaussian copula tasks.",
    )
    parser.add_argument(
        "--hybrid178_profile_transport_ratio",
        type=float,
        default=1.0,
        help=(
            "Fraction of official-profile slots that apply full frozen "
            "class-conditional marginal transport. Remaining slots retain "
            "profile dimensions, class count and class balance but keep raw "
            "official GraphSCM feature distributions."
        ),
    )
    parser.add_argument(
        "--hybrid178_protected_profile_transport_ratio",
        type=float,
        default=None,
        help=(
            "Optional full-marginal transport ratio used only for protected "
            "profiles; unset inherits the global profile transport ratio."
        ),
    )
    parser.add_argument(
        "--hybrid178_target_profile_transport_ratio",
        type=float,
        default=None,
        help=(
            "Optional full-marginal transport ratio used only for target "
            "profiles; unset inherits the global profile transport ratio."
        ),
    )
    parser.add_argument(
        "--hybrid178_profile_shape_jitter_strength",
        type=float,
        default=0.0,
        help=(
            "Neighborhood width in [0,1] for feature count, class count and "
            "class-balance jitter on shape-only official-profile tasks."
        ),
    )
    parser.add_argument(
        "--hybrid178_protected_profile_shape_jitter_strength",
        type=float,
        default=None,
        help="Optional protected-profile override for shape-only jitter.",
    )
    parser.add_argument(
        "--hybrid178_target_profile_shape_jitter_strength",
        type=float,
        default=None,
        help="Optional target-profile override for shape-only jitter.",
    )
    parser.add_argument(
        "--hybrid178_profile_supervised_candidate_count",
        type=int,
        default=1,
        help=(
            "For shape-only profile slots, generate this many official "
            "GraphSCM candidates and select the closest frozen-profile match "
            "by label NMI, higher-order dependence and task difficulty."
        ),
    )
    parser.add_argument(
        "--hybrid178_protected_profile_supervised_candidate_count",
        type=int,
        default=None,
        help=(
            "Optional protected-profile override for supervised GraphSCM "
            "candidate selection; unset inherits the global count."
        ),
    )
    parser.add_argument(
        "--hybrid178_target_profile_supervised_candidate_count",
        type=int,
        default=None,
        help=(
            "Optional target-profile override for supervised GraphSCM "
            "candidate selection; unset inherits the global count."
        ),
    )
    parser.add_argument(
        "--hybrid178_profile_copula_blend_scale",
        type=float,
        default=1.0,
        help=(
            "Scale in [0,1] applied to the frozen target-copula weight in the "
            "official-profile branch; lower values retain more official GraphSCM "
            "dependence while preserving profile dimensions, labels and marginals."
        ),
    )
    parser.add_argument(
        "--hybrid178_protected_profile_copula_blend_scale",
        type=float,
        default=None,
        help=(
            "Optional [0,1] target-copula scale used only for protected "
            "official-profile tasks; unset preserves the global scale exactly."
        ),
    )
    parser.add_argument(
        "--hybrid178_target_profile_copula_blend_scale",
        type=float,
        default=None,
        help=(
            "Optional [0,1] target-copula scale used only for target-group "
            "official-profile tasks; unset preserves the global scale exactly."
        ),
    )
    parser.add_argument(
        "--hybrid178_smooth_ratio",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility flag; must remain zero. Use the explicitly "
            "bounded GT-atom mixture rather than a smoothed row-bootstrap branch."
        ),
    )
    parser.add_argument(
        "--hybrid178_train_fraction",
        type=float,
        default=0.80,
        help="Support fraction used by hybrid178; 0.80 matches the data178 evaluation split.",
    )
    parser.add_argument(
        "--hybrid178_seed",
        type=int,
        default=178,
        help="Base seed; rank and DataLoader worker IDs are deterministically folded into it.",
    )
    parser.add_argument(
        "--hybrid178_schedule_start_batch",
        type=int,
        default=0,
        help=(
            "Absolute DDP-global synthetic-batch index for the first batch. "
            "Set this to the checkpoint step when resuming so prior batches "
            "are not replayed from index zero."
        ),
    )
    parser.add_argument(
        "--hybrid178_profile_cache_size",
        type=int,
        default=2,
        help="Maximum number of raw train-side datasets cached by each prior worker.",
    )
    parser.add_argument(
        "--hybrid178_profile_max_rows",
        type=int,
        default=200000,
        help="Maximum stratified train-side rows cached for one hybrid178 template.",
    )
    parser.add_argument(
        "--hybrid178_exact_replay",
        default=False,
        type=str2bool,
        help=(
            "Deprecated whole-task replay flag; must remain false. A small disclosed "
            "empirical component is controlled separately by hybrid178_gt_atom_ratio."
        ),
    )
    parser.add_argument(
        "--hybrid178_gt_atom_ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of each Hybrid-178 task replaced by exact preprocessed GT-train "
            "rows. This explicitly creates empirical point mass and is not synthetic-only."
        ),
    )
    parser.add_argument(
        "--hybrid178_gt_atom_min_rows",
        type=int,
        default=1,
        help=(
            "Minimum exact GT rows per task when hybrid178_gt_atom_ratio is positive; "
            "one gives a deterministic finite-sample hit guarantee."
        ),
    )
    parser.add_argument(
        "--hybrid178_gt_atom_require_full_schema",
        default=True,
        type=str2bool,
        help="Reject GT atom injection if the task does not preserve every GT feature column.",
    )
    parser.add_argument(
        "--hybrid178_runtime_isolated",
        default=False,
        type=str2bool,
        help=(
            "Require a compiled synthetic surrogate manifest and hard-block every "
            "runtime path under hybrid178_forbidden_data_root."
        ),
    )
    parser.add_argument(
        "--hybrid178_forbidden_data_root",
        type=str,
        default=None,
        help="GT root that runtime-isolated Hybrid-178 must never open.",
    )
    parser.add_argument(
        "--hybrid178_quality_gate",
        default=True,
        type=str2bool,
        help="Reject isolated tasks that drift too far from their frozen surrogate profile.",
    )
    parser.add_argument(
        "--hybrid178_protected_priority_extras",
        default=False,
        type=str2bool,
        help=(
            "After assigning one conditioned task to every Hybrid-178 profile, "
            "allocate surplus conditioned slots to the 20 protected profiles "
            "with deterministic round-robin balancing."
        ),
    )
    parser.add_argument(
        "--hybrid178_protected_priority_hardness_weighted",
        default=False,
        type=str2bool,
        help=(
            "After the protected one-extra minimum, sample only repeated "
            "protected surplus slots using frozen row-free blueprint hardness "
            "scores. Requires hybrid178_protected_priority_extras."
        ),
    )
    parser.add_argument(
        "--hybrid178_protected_priority_collapse_risk_weighted",
        default=False,
        type=str2bool,
        help=(
            "After the protected one-extra minimum, weight repeated protected "
            "surplus slots using frozen row-free late-collapse risk statistics. "
            "Requires protected-priority extras and is mutually exclusive with "
            "hardness weighting."
        ),
    )
    parser.add_argument("--prior_device", default="cpu", type=str, help="Device for prior data generation")
    parser.add_argument(
        "--prior_num_workers",
        type=int,
        default=-1,
        help="DataLoader worker count for prior generation. Use -1 to auto-size from CPU budget per rank.",
    )
    parser.add_argument(
        "--prior_prefetch_factor",
        type=int,
        default=8,
        help="DataLoader prefetch factor for prior generation when num_workers > 0.",
    )
    parser.add_argument(
        "--prior_persistent_workers",
        default=True,
        type=str2bool,
        help="Keep prior DataLoader workers alive across iterator reuse when num_workers > 0.",
    )
    parser.add_argument(
        "--prior_n_jobs",
        type=int,
        default=1,
        help="Internal joblib parallelism used by PriorDataset generation.",
    )
    parser.add_argument(
        "--prior_num_threads_per_generate",
        type=int,
        default=1,
        help="Maximum inner threads per generated dataset when prior_n_jobs > 1.",
    )
    parser.add_argument(
        "--prior_pin_memory",
        default=True,
        type=str2bool,
        help="If True, pin CPU prior batches before accelerator transfer.",
    )
    parser.add_argument(
        "--data_transfer_non_blocking",
        default=False,
        type=str2bool,
        help="If True, pass non_blocking=True when moving prior tensors to the training device.",
    )
    parser.add_argument(
        "--prior_cache_enabled",
        default=False,
        type=str2bool,
        help="If True, use a rank-local in-memory async cache between DataLoader and training.",
    )
    parser.add_argument(
        "--prior_cache_max_batches",
        type=int,
        default=32,
        help="Maximum number of batches kept in the rank-local prior cache.",
    )
    parser.add_argument(
        "--prior_cache_max_gb",
        type=float,
        default=64.0,
        help="Maximum CPU memory budget in GB for the rank-local prior cache.",
    )
    parser.add_argument(
        "--prior_cache_prefill_batches",
        type=int,
        default=8,
        help="Number of batches to prefill before training starts when prior cache is enabled.",
    )
    parser.add_argument(
        "--prior_cache_prefill_gb",
        type=float,
        default=0.0,
        help="Additional CPU memory target in GB to prefill before training starts when prior cache is enabled.",
    )
    parser.add_argument(
        "--prior_cache_get_timeout_s",
        type=float,
        default=300.0,
        help="Seconds to wait for a batch when consuming from the prior cache.",
    )
    parser.add_argument(
        "--prior_cache_put_timeout_s",
        type=float,
        default=300.0,
        help="Seconds to wait for free space when producing into the prior cache.",
    )
    parser.add_argument(
        "--prior_cache_stats_every",
        type=int,
        default=1,
        help="Collect rank-0 prior cache stats every N completed steps; set 0 to disable stable-step stats.",
    )
    parser.add_argument(
        "--prior_extra_trees_filter",
        default=True,
        type=str2bool,
        help="For tabiclv2_cls, enable the ExtraTrees bootstrap usefulness filter.",
    )
    parser.add_argument(
        "--prior_filter_bootstrap_samples",
        type=int,
        default=200,
        help="For tabiclv2_cls, number of bootstrap subsamples used by the ExtraTrees filter.",
    )
    parser.add_argument(
        "--tabiclv2_class_conditional_cauchy_multiclass",
        default=False,
        type=str2bool,
        help=(
            "For tabiclv2_cls, replace multiclass converter-label generation with a "
            "class-conditional Cauchy DAG where y is sampled first as a semantic root cause. "
            "When hard-diversity replay is active, multiclass hard-diversity batches use this "
            "generator while binary stress batches are kept."
        ),
    )
    parser.add_argument(
        "--tabiclv2_base_class_bin_weights",
        type=str,
        default=None,
        help=(
            "For tabiclv2_cls base prior, optional comma- or semicolon-separated weights for class-count bins: "
            "binary, 3, 4-5, 6-10, and >10."
        ),
    )
    parser.add_argument(
        "--tabiclv2_base_feature_bin_weights",
        type=str,
        default=None,
        help=(
            "For tabiclv2_cls base prior, optional comma- or semicolon-separated weights for feature-count bins: "
            "2-5, 6-10, 11-20, 21-50, and 51-100."
        ),
    )
    parser.add_argument(
        "--tabiclv2_base_cat_ratio_bin_weights",
        type=str,
        default=None,
        help=(
            "For tabiclv2_cls base prior, optional comma- or semicolon-separated weights for categorical-ratio bins: "
            "all_num, low_cat, mixed, high_cat, and all_cat."
        ),
    )
    parser.add_argument(
        "--tabiclv2_categorical_max_cardinality",
        type=int,
        default=9,
        help=(
            "For tabiclv2_cls base synthetic prior, maximum input categorical cardinality sampled by the "
            "categorical converter. Default 9 preserves the original prior."
        ),
    )
    parser.add_argument(
        "--tabiclv2_graph_nodes_min",
        type=int,
        default=2,
        help="For tabiclv2_cls, minimum sampled Cauchy-DAG graph nodes. Default 2 preserves the original prior.",
    )
    parser.add_argument(
        "--tabiclv2_graph_nodes_max",
        type=int,
        default=32,
        help="For tabiclv2_cls, maximum sampled Cauchy-DAG graph nodes. Default 32 preserves the original prior.",
    )
    parser.add_argument(
        "--tabiclv2_node_extra_dim_min",
        type=int,
        default=1,
        help="For tabiclv2_cls, minimum hidden extra dimensions per graph node. Default 1 preserves the original prior.",
    )
    parser.add_argument(
        "--tabiclv2_node_extra_dim_max",
        type=int,
        default=32,
        help="For tabiclv2_cls, maximum hidden extra dimensions per graph node. Default 32 preserves the original prior.",
    )
    parser.add_argument(
        "--tabiclv2_latent_needed_nodes_min",
        type=optional_int_type,
        default=None,
        help="For tabiclv2_cls, optional lower guard on hidden ancestor nodes used to generate assigned X/y nodes.",
    )
    parser.add_argument(
        "--tabiclv2_latent_needed_nodes_max",
        type=optional_int_type,
        default=None,
        help="For tabiclv2_cls, optional upper guard on hidden ancestor nodes used to generate assigned X/y nodes.",
    )
    parser.add_argument(
        "--tabiclv2_dynamic_reweight_path",
        type=str,
        default=None,
        help=(
            "For tabiclv2_cls base prior, optional JSON file with live class/feature/categorical "
            "bin weights. When the file changes, workers reload it and blend it with the base weights."
        ),
    )
    parser.add_argument(
        "--tabiclv2_dynamic_reweight_reload_sec",
        type=float,
        default=30.0,
        help="Minimum seconds between checking the tabiclv2 dynamic reweight JSON for updates.",
    )
    parser.add_argument(
        "--tabiclv2_dynamic_reweight_blend",
        type=float,
        default=1.0,
        help="Blend factor for dynamic bin weights: 0 keeps base weights, 1 uses the JSON weights.",
    )
    parser.add_argument(
        "--bad_batch_log_enabled",
        default=False,
        type=str2bool,
        help="If True, rank 0 writes JSONL diagnostics for unusually hard synthetic training batches.",
    )
    parser.add_argument(
        "--bad_batch_log_path",
        default=None,
        type=str,
        help="Path for bad-batch JSONL diagnostics. Defaults to checkpoint_dir/bad_batches_rank0.jsonl.",
    )
    parser.add_argument(
        "--bad_batch_start_step",
        type=int,
        default=10000,
        help="Do not record bad-batch diagnostics before this training step.",
    )
    parser.add_argument(
        "--bad_batch_accuracy_threshold",
        type=float,
        default=0.55,
        help="Record a batch when its rank-0 micro-batch accuracy is at or below this value.",
    )
    parser.add_argument(
        "--bad_batch_ce_threshold",
        type=float,
        default=1.25,
        help="Record a batch when its rank-0 micro-batch CE is at or above this value.",
    )
    parser.add_argument(
        "--bad_batch_max_records",
        type=int,
        default=50000,
        help="Maximum bad-batch diagnostics to write on rank 0. Set 0 for no limit.",
    )
    parser.add_argument(
        "--bad_batch_log_every",
        type=int,
        default=1,
        help="Record only every Nth bad-batch match on rank 0.",
    )
    parser.add_argument(
        "--bad_batch_print_to_stderr",
        default=True,
        type=str2bool,
        help="If True, print one concise bad-batch provenance line to stderr for each JSONL record.",
    )
    parser.add_argument(
        "--batch_source_log_enabled",
        default=False,
        type=str2bool,
        help="If True, rank 0 writes JSONL provenance for sampled synthetic batches.",
    )
    parser.add_argument(
        "--batch_source_log_path",
        default=None,
        type=str,
        help="Path for batch-source JSONL diagnostics. Defaults to checkpoint_dir/batch_sources_rank0.jsonl.",
    )
    parser.add_argument(
        "--batch_source_log_every",
        type=int,
        default=100,
        help="Record one rank-0 batch-source entry every N training steps.",
    )
    parser.add_argument(
        "--batch_source_max_records",
        type=int,
        default=10000,
        help="Maximum batch-source diagnostics to write on rank 0. Set 0 for no limit.",
    )
    parser.add_argument(
        "--batch_source_flush_every",
        type=int,
        default=100,
        help="Flush buffered batch-source JSONL records to disk every N records.",
    )
    parser.add_argument(
        "--batch_source_print_to_stderr",
        default=True,
        type=str2bool,
        help="If True, print one concise batch-source provenance line to stderr for each JSONL record.",
    )
    parser.add_argument(
        "--progress_to_stdout",
        default=False,
        type=str2bool,
        help="If True, send tqdm progress to stdout instead of stderr.",
    )
    parser.add_argument(
        "--progress_min_interval",
        type=float,
        default=0.1,
        help="Minimum seconds between tqdm refreshes.",
    )
    parser.add_argument(
        "--progress_min_iters",
        type=int,
        default=1,
        help="Minimum training iterations between tqdm refreshes.",
    )
    parser.add_argument(
        "--progress_max_interval",
        type=float,
        default=10.0,
        help="Maximum seconds between tqdm refreshes.",
    )
    parser.add_argument(
        "--progress_dynamic_ncols",
        default=True,
        type=str2bool,
        help="If True, let tqdm adapt progress bar width to terminal size.",
    )
    parser.add_argument(
        "--progress_postfix_refresh",
        default=True,
        type=str2bool,
        help="If True, refresh tqdm immediately after updating per-step metrics.",
    )
    parser.add_argument(
        "--progress_postfix_every",
        type=int,
        default=1,
        help="Update tqdm postfix every N steps on rank 0. Set 1 to update every step.",
    )
    parser.add_argument(
        "--empty_cache_every",
        type=int,
        default=1,
        help="Call torch.cuda.empty_cache() every N training steps; set to 0 to disable.",
    )
    parser.add_argument(
        "--zero_grad_begin",
        default=True,
        type=str2bool,
        help="If True, clear optimizer gradients at the beginning of each batch in addition to the end.",
    )
    parser.add_argument(
        "--set_train_every_step",
        default=True,
        type=str2bool,
        help="If True, call model.train() at the beginning of every optimizer step.",
    )
    parser.add_argument(
        "--validate_micro_batch_shapes",
        default=True,
        type=str2bool,
        help="If True, validate that seq_len and train_size are uniform within every micro-batch.",
    )

    ###########################################################################
    ##### Model Architecture Config ###########################################
    ###########################################################################
    parser.add_argument(
        "--amp",
        default=True,
        type=str2bool,
        help="If True, use automatic mixed precision (AMP) which can provide significant speedups on compatible GPU",
    )
    parser.add_argument(
        "--grad_scaler",
        default=True,
        type=str2bool,
        help="If True, enable GradScaler when AMP is enabled.",
    )
    parser.add_argument(
        "--model_compile",
        default=False,
        type=str2bool,
        help="If True, compile the model using torch.compile for speedup",
    )
    parser.add_argument(
        "--model_compile_parts",
        default="",
        type=str,
        help=(
            "Comma-separated TabICL submodules to compile with torch.compile. "
            "Supported values: col, row, icl. Empty means no submodule-only compile."
        ),
    )
    parser.add_argument(
        "--model_compile_mode",
        default=None,
        type=str,
        help="Optional torch.compile mode, for example reduce-overhead or max-autotune.",
    )

    # Column Embedding Config
    parser.add_argument("--embed_dim", type=int, default=128, help="Base embedding dimension")
    parser.add_argument("--col_num_blocks", type=int, default=3, help="Number of blocks in column embedder")
    parser.add_argument("--col_nhead", type=int, default=4, help="Number of attention heads in column embedder")
    parser.add_argument("--col_num_inds", type=int, default=128, help="Number of inducing points in column embedder")
    parser.add_argument(
        "--col_feature_group",
        type=feature_group_type,
        default=False,
        help="Column feature grouping mode: False, True/same, or valid. Training priors with variable d require False.",
    )
    parser.add_argument("--col_feature_group_size", type=int, default=3, help="Number of features per group")
    parser.add_argument(
        "--col_ssmax",
        type=str,
        default="qassmax-mlp-elementwise",
        choices=[
            "none",
            "ssmax",
            "ssmax-mlp",
            "ssmax-mlp-elementwise",
            "qassmax-mlp",
            "qassmax-mlp-elementwise",
        ],
        help="Scalable-softmax mode for the column embedder. Use none for ablation.",
    )
    parser.add_argument(
        "--col_output_layer_norm",
        default=False,
        type=str2bool,
        help="Apply LayerNorm to the column embedder output before row interaction.",
    )
    parser.add_argument("--freeze_col", default=False, type=str2bool, help="Whether to freeze the column embedder")

    # Row Interaction Config
    parser.add_argument("--row_num_blocks", type=int, default=3, help="Number of blocks in row interactor")
    parser.add_argument("--row_nhead", type=int, default=8, help="Number of attention heads in row interactor")
    parser.add_argument("--row_num_cls", type=int, default=4, help="Number of CLS tokens in row interactor")
    parser.add_argument("--row_rope_base", type=float, default=100000, help="RoPE base value for row interactor")
    parser.add_argument(
        "--row_use_rope",
        default=True,
        type=str2bool,
        help="Whether to use RoPE in the row interaction transformer.",
    )
    parser.add_argument("--freeze_row", default=False, type=str2bool, help="Whether to freeze the row interactor")

    # ICL Config
    parser.add_argument("--icl_num_blocks", type=int, default=12, help="Number of transformer blocks in ICL predictor")
    parser.add_argument("--icl_nhead", type=int, default=4, help="Number of attention heads in ICL predictor")
    parser.add_argument(
        "--icl_ssmax",
        type=str,
        default="qassmax-mlp-elementwise",
        choices=[
            "none",
            "ssmax",
            "ssmax-mlp",
            "ssmax-mlp-elementwise",
            "qassmax-mlp",
            "qassmax-mlp-elementwise",
        ],
        help="Scalable-softmax mode for the ICL transformer.",
    )
    parser.add_argument("--freeze_icl", default=False, type=str2bool, help="Whether to freeze the ICL predictor")
    parser.add_argument(
        "--layer_gate_enabled",
        default=False,
        type=str2bool,
        help="Enable query-wise logit mixing over selected ICL layer outputs.",
    )
    parser.add_argument(
        "--layer_gate_layers",
        type=str,
        default="4;6;8;10;11",
        help="Semicolon/comma-separated ICL layer indices decoded and mixed by the layer gate.",
    )
    parser.add_argument("--layer_gate_hidden_dim", type=int, default=128, help="Hidden size of the layer gate MLP.")
    parser.add_argument(
        "--layer_gate_temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for layer gate weights.",
    )
    parser.add_argument(
        "--layer_gate_low_layer_floor",
        type=float,
        default=0.0,
        help="Minimum total gate mass reserved for layers <= --layer_gate_low_layer_max.",
    )
    parser.add_argument(
        "--layer_gate_low_layer_max",
        type=int,
        default=6,
        help="Largest ICL layer index treated as low/mid layer for the low-layer floor.",
    )
    parser.add_argument(
        "--layer_gate_use_confidence_features",
        default=False,
        type=str2bool,
        help="Include deepest-layer confidence/entropy features in the query-wise layer gate.",
    )
    parser.add_argument(
        "--layer_gate_repr_source",
        type=str,
        default="deepest",
        choices=["deepest", "multi_mean", "multi_delta"],
        help="Representation used by the layer gate: deepest selected layer, mean of selected layers, or low/deep/delta concat.",
    )
    parser.add_argument(
        "--layer_gate_confidence_source",
        type=str,
        default="deepest",
        choices=["none", "deepest", "all_layers"],
        help="Confidence features supplied to the layer gate.",
    )
    parser.add_argument(
        "--layer_gate_max_weight_target",
        type=float,
        default=0.85,
        help="Target maximum single-layer gate weight for optional max-weight regularization.",
    )
    parser.add_argument(
        "--layer_gate_entropy_reg_weight",
        type=float,
        default=0.0,
        help="Weight for layer-gate entropy regularization; encourages non-collapsed layer mixing.",
    )
    parser.add_argument(
        "--layer_gate_max_weight_reg_weight",
        type=float,
        default=0.0,
        help="Weight for penalizing layer-gate max weight above --layer_gate_max_weight_target.",
    )
    parser.add_argument(
        "--attention_gate_enabled",
        default=str2bool(os.environ.get("ATTENTION_GATE_ENABLED", "False")),
        type=str2bool,
        help="Enable support-dataset-conditioned multiplicative gates on ICL SDPA head outputs.",
    )
    parser.add_argument(
        "--attention_gate_shape",
        type=str,
        default=os.environ.get("ATTENTION_GATE_SHAPE", "scalar"),
        choices=["scalar", "channel"],
        help="Per-head scalar or per-head-channel gate shape.",
    )
    parser.add_argument(
        "--attention_gate_layers",
        type=str,
        default=os.environ.get("ATTENTION_GATE_LAYERS", "8;9;10;11"),
        help="Semicolon/comma-separated zero-based ICL blocks carrying the attention gate.",
    )
    parser.add_argument(
        "--attention_gate_hidden_dim",
        type=int,
        default=int(os.environ.get("ATTENTION_GATE_HIDDEN_DIM", "128")),
        help="Support-schema embedding dimension for the attention-gate conditioner.",
    )
    parser.add_argument(
        "--attention_gate_rho",
        type=float,
        default=float(os.environ.get("ATTENTION_GATE_RHO", "0.25")),
        help="Bound in gate=1+rho*tanh(logit).",
    )
    parser.add_argument(
        "--swiglu_enabled",
        default=str2bool(os.environ.get("SWIGLU_ENABLED", "False")),
        type=str2bool,
        help="Replace only the ICL GELU FFNs with SwiGLU FFNs.",
    )
    parser.add_argument(
        "--swiglu_conditioned",
        default=str2bool(os.environ.get("SWIGLU_CONDITIONED", "False")),
        type=str2bool,
        help="Condition ICL SwiGLU gates on support-only schema statistics and group RFF.",
    )
    parser.add_argument(
        "--swiglu_hidden_dim",
        type=int,
        default=int(os.environ.get("SWIGLU_HIDDEN_DIM", "1024")),
        help="Inner dimension of each ICL SwiGLU block.",
    )
    parser.add_argument(
        "--swiglu_context_hidden_dim",
        type=int,
        default=int(os.environ.get("SWIGLU_CONTEXT_HIDDEN_DIM", "128")),
        help="Support-schema embedding dimension used by conditioned SwiGLU.",
    )
    parser.add_argument(
        "--swiglu_rho",
        type=float,
        default=float(os.environ.get("SWIGLU_RHO", "0.25")),
        help="Bound for conditioned SwiGLU gamma/beta modulation.",
    )
    parser.add_argument(
        "--swiglu_output_scale",
        type=float,
        default=float(os.environ.get("SWIGLU_OUTPUT_SCALE", "1.0")),
        help="Fixed multiplier on the SwiGLU product before the down projection.",
    )
    parser.add_argument(
        "--swiglu_product_tanh_rms_multiple",
        type=float,
        default=float(os.environ.get("SWIGLU_PRODUCT_TANH_RMS_MULTIPLE", "0")),
        help=(
            "Smoothly cap the SwiGLU gate-value product at this multiple of each token's "
            "detached product RMS; zero disables the intervention."
        ),
    )
    parser.add_argument(
        "--swiglu_product_tanh_last_n_layers",
        type=int,
        default=int(os.environ.get("SWIGLU_PRODUCT_TANH_LAST_N_LAYERS", "0")),
        help=(
            "Apply product-RMS smoothing only to the final N ICL blocks; zero disables it. "
            "The multiple and depth must be enabled together."
        ),
    )
    parser.add_argument(
        "--swiglu_init_seed_base",
        type=int,
        default=int(os.environ.get("SWIGLU_INIT_SEED_BASE", "2026082601")),
        help="Base of the layer-specific isolated RNG namespace for SwiGLU-only parameters.",
    )
    parser.add_argument(
        "--cr2_shared_refinement_enabled",
        default=str2bool(os.environ.get("CR2_SHARED_REFINEMENT_ENABLED", "False")),
        type=str2bool,
        help=(
            "Enable identity-gated C1-R1-C1-R1 refinement using the existing shared "
            "column and row weights; only two scalar gates are added."
        ),
    )
    parser.add_argument(
        "--qk_pds_attention_enabled",
        default=str2bool(os.environ.get("QK_PDS_ATTENTION_ENABLED", "False")),
        type=str2bool,
        help="Replace ICL QASSMax with per-head Q/K RMSNorm and learned per-dimension query scale.",
    )
    parser.add_argument(
        "--cls8_pooled_enabled",
        default=str2bool(os.environ.get("CLS8_POOLED_ENABLED", "False")),
        type=str2bool,
        help="Use eight row CLS tokens and fixed pairwise pooling to four ICL slots.",
    )
    parser.add_argument(
        "--shared_depth_icl_enabled",
        default=str2bool(os.environ.get("SHARED_DEPTH_ICL_ENABLED", "False")),
        type=str2bool,
        help="Apply the same ICL block stack recurrently with identity-gated extra passes.",
    )
    parser.add_argument(
        "--shared_depth_icl_rho",
        default=float(os.environ.get("SHARED_DEPTH_ICL_RHO", "1.0")),
        type=float,
        help="Maximum tanh gate magnitude for every shared-depth ICL extra pass.",
    )
    parser.add_argument(
        "--shared_depth_icl_dataset_conditioned",
        default=str2bool(os.environ.get("SHARED_DEPTH_ICL_DATASET_CONDITIONED", "False")),
        type=str2bool,
        help=(
            "Use support-only c_D with alpha_D=tanh(a+0.1*(2*sigmoid(w^T c_D)-1)); "
            "w is initialized to zero."
        ),
    )
    parser.add_argument(
        "--shared_depth_icl_num_passes",
        default=int(os.environ.get("SHARED_DEPTH_ICL_NUM_PASSES", "2")),
        type=int,
        choices=range(2, 17),
        help="Number of recurrent passes through the shared ICL block stack.",
    )
    parser.add_argument(
        "--cls8_width_enabled",
        default=str2bool(os.environ.get("CLS8_WIDTH_ENABLED", "False")),
        type=str2bool,
        help="Retain all eight 128-d row CLS outputs as a 1024-d ICL representation.",
    )
    parser.add_argument(
        "--swiglu_muon_lr_multiplier",
        type=float,
        default=float(os.environ.get("SWIGLU_MUON_LR_MULTIPLIER", "1.0")),
        help=(
            "Explicit multiplier on native shape-scaled Muon LR for ICL SwiGLU matrices. "
            "Keep 1.0 for parameter-matched width; values such as sqrt(2/3) are ablations."
        ),
    )
    parser.add_argument(
        "--swiglu_value_proj_weight_decay",
        type=float,
        default=float(os.environ.get("SWIGLU_VALUE_PROJ_WEIGHT_DECAY", "-1")),
        help="Weight decay for SwiGLU value projections; negative means inherit global weight decay.",
    )
    parser.add_argument(
        "--swiglu_down_lr_bootstrap_multiplier",
        type=float,
        default=float(os.environ.get("SWIGLU_DOWN_LR_BOOTSTRAP_MULTIPLIER", "1.0")),
        help=(
            "Temporary multiplier for ICL SwiGLU down-projection matrix LR. "
            "The safe paired ablation range is [1, 1.5]; 1 disables it."
        ),
    )
    parser.add_argument(
        "--swiglu_down_lr_bootstrap_steps",
        type=int,
        default=int(os.environ.get("SWIGLU_DOWN_LR_BOOTSTRAP_STEPS", "0")),
        help=(
            "Number of initial optimizer steps using the temporary down-projection LR boost. "
            "During this window down-projection WD is divided by the same multiplier."
        ),
    )
    parser.add_argument(
        "--layer_gate_aux_ce_weight",
        type=float,
        default=0.0,
        help="Weight for auxiliary CE on selected layer-exit logits.",
    )
    parser.add_argument(
        "--layer_gate_aux_ce_layers",
        type=str,
        default="4;6;8",
        help="Semicolon/comma-separated layer indices for auxiliary layer-exit CE.",
    )

    # Schema-conditioned expert / FiLM config
    parser.add_argument(
        "--schema_expert_enabled",
        default=False,
        type=str2bool,
        help="Enable support-schema-conditioned routed adapter experts before the ICL predictor.",
    )
    parser.add_argument("--schema_expert_num_experts", type=int, default=32, help="Number of schema adapter experts")
    parser.add_argument("--schema_expert_top_k", type=int, default=2, help="Top-k experts selected per table")
    parser.add_argument("--schema_expert_bottleneck", type=int, default=32, help="Adapter bottleneck dimension")
    parser.add_argument("--schema_expert_hidden_dim", type=int, default=128, help="Schema encoder hidden dimension")
    parser.add_argument(
        "--schema_expert_router_temperature",
        type=float,
        default=1.0,
        help="Router softmax temperature for schema experts.",
    )
    parser.add_argument(
        "--schema_expert_adapter_scale",
        type=float,
        default=0.1,
        help="Residual scale for schema routed adapter updates.",
    )
    parser.add_argument(
        "--schema_film_enabled",
        default=False,
        type=str2bool,
        help="Enable support-schema-conditioned FiLM on ICL input representations.",
    )
    parser.add_argument("--schema_film_scale", type=float, default=0.1, help="Residual scale for schema FiLM")
    parser.add_argument("--schema_expert_dropout", type=float, default=0.0, help="Dropout inside schema adapters")
    parser.add_argument(
        "--schema_context_source",
        type=str,
        default="support",
        choices=["support"],
        help="Which table portion is allowed to build schema context.",
    )

    # Support-conditioned function-token config
    parser.add_argument(
        "--function_tokens_enabled",
        default=False,
        type=str2bool,
        help="Enable support-conditioned dataset/query/latent function-token updates before ICL.",
    )
    parser.add_argument(
        "--function_token_use_dataset_token",
        default=True,
        type=str2bool,
        help="Use dataset-level function tokens.",
    )
    parser.add_argument(
        "--function_token_use_query_token",
        default=True,
        type=str2bool,
        help="Use learnable query function-token offsets.",
    )
    parser.add_argument(
        "--function_token_use_latent_tokens",
        default=False,
        type=str2bool,
        help="Use latent function-router tokens.",
    )
    parser.add_argument("--function_token_dataset_count", type=int, default=1, help="Number of dataset tokens")
    parser.add_argument("--function_token_query_count", type=int, default=1, help="Number of query tokens per row")
    parser.add_argument("--function_token_latent_count", type=int, default=8, help="Number of latent function tokens")
    parser.add_argument("--function_token_num_heads", type=int, default=8, help="Function-token attention heads")
    parser.add_argument("--function_token_num_layers", type=int, default=1, help="Function-token cross-attention layers")
    parser.add_argument("--function_token_hidden_dim", type=int, default=128, help="Function-token MLP hidden dim")
    parser.add_argument("--function_token_scale", type=float, default=0.1, help="Residual scale for function-token updates")
    parser.add_argument("--function_token_dropout", type=float, default=0.0, help="Dropout inside function-token blocks")
    parser.add_argument(
        "--function_token_context_source",
        type=str,
        default="support",
        choices=["support"],
        help="Which table portion is allowed to build function-token context.",
    )

    # Pseudo-label thinking config (reserved off-by-default path)
    parser.add_argument(
        "--use_pseudo_ssmax_thinking",
        default=False,
        type=str2bool,
        help="If True, enable teacher-forced pseudo-label second-pass training.",
    )
    parser.add_argument(
        "--pseudo_thinking_prob",
        type=float,
        default=0.25,
        help="Probability of running the pseudo-label second pass for a training micro-batch.",
    )
    parser.add_argument(
        "--pseudo_thinking_fraction",
        type=float,
        default=0.25,
        help="Fraction of test rows used as teacher-forced pseudo context in the second pass.",
    )
    parser.add_argument(
        "--pseudo_thinking_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the second-pass CE loss when pseudo-label thinking is active.",
    )
    parser.add_argument(
        "--stage3_teacher_checkpoint_path",
        default=None,
        type=str,
        help="Optional frozen Stage-2 teacher checkpoint for ordinary Stage-3 KD.",
    )
    parser.add_argument(
        "--stage3_kd_weight",
        type=float,
        default=0.0,
        help="Weight for ordinary Stage-3 teacher KL distillation. 0 disables the teacher path.",
    )
    parser.add_argument(
        "--stage3_kd_temperature",
        type=float,
        default=2.0,
        help="Temperature for ordinary Stage-3 teacher KL distillation.",
    )
    parser.add_argument(
        "--stage3_l2sp_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for Stage-2 parameter anchoring loss. 0 disables L2-SP. "
            "The anchor is captured after loading --checkpoint_path."
        ),
    )
    parser.add_argument(
        "--stage3_anchor_pullback",
        type=float,
        default=0.0,
        help=(
            "Post-optimizer interpolation strength toward the loaded Stage-2 anchor. "
            "0 disables pullback; values like 0.01-0.05 prevent late Stage-3 drift."
        ),
    )
    parser.add_argument(
        "--stage3_ema_decay",
        type=float,
        default=0.0,
        help="EMA decay for Stage-3 evaluation/checkpoint weights. 0 disables EMA.",
    )
    parser.add_argument(
        "--stage3_ema_start_step",
        type=int,
        default=0,
        help="First training step index at which Stage-3 EMA updates are applied.",
    )
    parser.add_argument(
        "--stage3_save_ema_checkpoints",
        default=False,
        type=str2bool,
        help="If True, saved step checkpoints contain EMA weights while training continues with raw weights.",
    )
    parser.add_argument(
        "--balanced_ce_weight",
        type=float,
        default=0.0,
        help="Weight for an optional inverse-frequency CE term on the synthetic query rows. 0 disables it.",
    )
    parser.add_argument(
        "--balanced_ce_max_weight",
        type=float,
        default=3.0,
        help="Maximum per-row class weight used by --balanced_ce_weight.",
    )
    parser.add_argument(
        "--support_balanced_ce_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for an optional support-set-aware minority CE term. "
            "0 disables it; unlike --balanced_ce_weight, weights come from y_train class frequencies."
        ),
    )
    parser.add_argument(
        "--support_balanced_ce_gamma",
        type=float,
        default=0.5,
        help="Exponent for support-set inverse-frequency CE weighting.",
    )
    parser.add_argument(
        "--support_balanced_ce_max_weight",
        type=float,
        default=4.0,
        help="Maximum per-row weight for support-set-aware minority CE.",
    )
    parser.add_argument(
        "--support_balanced_ce_min_count",
        type=int,
        default=2,
        help="Minimum support rows required before boosting a class in support-set-aware minority CE.",
    )
    parser.add_argument(
        "--support_balanced_ce_minority_threshold",
        type=float,
        default=0.35,
        help="Only classes with support fraction at or below this threshold are boosted.",
    )
    parser.add_argument(
        "--support_prior_kl_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for matching the student's average query prediction distribution "
            "to the support-set class prior. 0 disables it."
        ),
    )
    parser.add_argument(
        "--support_prior_kl_min_count",
        type=int,
        default=2,
        help="Minimum support rows for a class to participate in support-prior KL.",
    )
    parser.add_argument(
        "--support_prior_kl_eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon used by support-prior KL.",
    )
    parser.add_argument(
        "--support_prior_floor_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for an asymmetric support-prior floor loss. "
            "It penalizes query prediction mass that falls too far below support-present class priors."
        ),
    )
    parser.add_argument(
        "--support_prior_floor_ratio",
        type=float,
        default=0.6,
        help="Required fraction of each eligible support prior that the mean query prediction should retain.",
    )
    parser.add_argument(
        "--support_prior_floor_min_count",
        type=int,
        default=2,
        help="Minimum support rows for a class to participate in support-prior floor loss.",
    )
    parser.add_argument(
        "--support_prior_floor_min_prior",
        type=float,
        default=0.005,
        help="Minimum support prior for a class to participate in support-prior floor loss.",
    )
    parser.add_argument(
        "--support_prior_floor_max_prior",
        type=float,
        default=0.5,
        help="Maximum support prior for a class to participate in support-prior floor loss.",
    )
    parser.add_argument(
        "--support_margin_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a support-set-aware logit margin loss on synthetic query rows. "
            "This discourages decision-boundary collapse for support-present non-majority classes."
        ),
    )
    parser.add_argument(
        "--support_margin_value",
        type=float,
        default=0.5,
        help="Required true-class logit margin over the strongest competing class.",
    )
    parser.add_argument(
        "--support_margin_gamma",
        type=float,
        default=0.5,
        help="Exponent for support-set inverse-frequency weighting in support-margin loss.",
    )
    parser.add_argument(
        "--support_margin_max_weight",
        type=float,
        default=6.0,
        help="Maximum per-row weight for support-margin loss.",
    )
    parser.add_argument(
        "--support_margin_min_count",
        type=int,
        default=2,
        help="Minimum support rows before a class participates in support-margin loss.",
    )
    parser.add_argument(
        "--support_margin_threshold",
        type=float,
        default=0.5,
        help="Only classes with support fraction at or below this threshold receive support-margin loss.",
    )
    parser.add_argument(
        "--stage3_minority_kd_weight",
        type=float,
        default=0.0,
        help=(
            "Extra teacher KL weight on query rows where the teacher predicts a support-set minority class. "
            "Requires --stage3_teacher_checkpoint_path."
        ),
    )
    parser.add_argument(
        "--stage3_minority_kd_gamma",
        type=float,
        default=0.5,
        help="Exponent for support-set inverse-frequency teacher-minority KD weighting.",
    )
    parser.add_argument(
        "--stage3_minority_kd_max_weight",
        type=float,
        default=4.0,
        help="Maximum per-row weight for teacher-minority KD.",
    )
    parser.add_argument(
        "--stage3_minority_kd_min_confidence",
        type=float,
        default=0.35,
        help="Minimum teacher probability required before applying teacher-minority KD.",
    )
    parser.add_argument(
        "--stage3_minority_kd_min_count",
        type=int,
        default=2,
        help="Minimum support rows for the teacher-predicted class before applying teacher-minority KD.",
    )
    parser.add_argument(
        "--stage3_minority_kd_threshold",
        type=float,
        default=0.35,
        help="Only teacher-predicted classes with support fraction at or below this threshold are boosted.",
    )
    parser.add_argument(
        "--stage3_support_classwise_kd_weight",
        type=float,
        default=0.0,
        help=(
            "Extra binary classwise teacher KL on support-present non-majority classes. "
            "This preserves Stage-2 probability mass for mid/rare classes without using real heldout datasets."
        ),
    )
    parser.add_argument(
        "--stage3_support_classwise_kd_gamma",
        type=float,
        default=0.5,
        help="Exponent for support-prior inverse-frequency class weights in classwise KD.",
    )
    parser.add_argument(
        "--stage3_support_classwise_kd_max_weight",
        type=float,
        default=6.0,
        help="Maximum class weight for support classwise KD.",
    )
    parser.add_argument(
        "--stage3_support_classwise_kd_min_count",
        type=int,
        default=2,
        help="Minimum support rows for a class to participate in support classwise KD.",
    )
    parser.add_argument(
        "--stage3_support_classwise_kd_min_prior",
        type=float,
        default=0.005,
        help="Minimum support prior for a class to participate in support classwise KD.",
    )
    parser.add_argument(
        "--stage3_support_classwise_kd_max_prior",
        type=float,
        default=0.5,
        help="Maximum support prior for a class to participate in support classwise KD.",
    )
    parser.add_argument(
        "--firewall_kd_dataset_dir",
        default=None,
        type=str,
        help="Optional data178/internet_firewall directory for conservative Stage-3 heldout KD.",
    )
    parser.add_argument(
        "--firewall_kd_weight",
        type=float,
        default=0.0,
        help="Weight for heldout internet_firewall teacher KL. 0 disables this auxiliary path.",
    )
    parser.add_argument(
        "--firewall_kd_ce_weight",
        type=float,
        default=0.0,
        help="Optional true-label CE weight on heldout internet_firewall query rows.",
    )
    parser.add_argument(
        "--firewall_kd_temperature",
        type=float,
        default=2.0,
        help="Temperature for heldout internet_firewall teacher KL.",
    )
    parser.add_argument(
        "--firewall_kd_interval",
        type=int,
        default=1,
        help="Run heldout internet_firewall KD every N training steps.",
    )
    parser.add_argument(
        "--firewall_kd_prob",
        type=float,
        default=1.0,
        help="Probability of running heldout internet_firewall KD on an eligible step.",
    )
    parser.add_argument(
        "--firewall_kd_support_rows",
        type=int,
        default=8192,
        help="Number of internet_firewall train+val rows sampled as in-context support.",
    )
    parser.add_argument(
        "--firewall_kd_query_rows",
        type=int,
        default=512,
        help="Number of internet_firewall train+val heldout rows sampled as auxiliary query.",
    )
    parser.add_argument(
        "--firewall_kd_query_class2_fraction",
        type=float,
        default=0.5,
        help="Fraction of auxiliary query rows sampled from class 2 to protect its recall.",
    )
    parser.add_argument(
        "--firewall_kd_class2_weight",
        type=float,
        default=3.0,
        help="Per-row loss multiplier for internet_firewall query rows with true label 2.",
    )
    parser.add_argument(
        "--firewall_kd_min_support_per_class",
        type=int,
        default=4,
        help="Minimum support rows per class when sampling the internet_firewall auxiliary task.",
    )
    parser.add_argument(
        "--firewall_kd_norm_method",
        default="none",
        type=str,
        choices=["none", "power", "quantile", "quantile_rtdl", "robust"],
        help="Single preprocessing normalization method used for heldout internet_firewall KD.",
    )

    # Shared Architecture Config
    parser.add_argument("--ff_factor", type=int, default=2, help="Expansion factor for feedforward dimensions")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function type")
    parser.add_argument(
        "--norm_first", default=True, type=str2bool, help="If True, use pre-norm transformer architecture"
    )
    parser.add_argument(
        "--bias_free_ln",
        default=False,
        type=str2bool,
        help="If True, remove biases from LayerNorm layers.",
    )
    parser.add_argument(
        "--recompute",
        default=False,
        type=str2bool,
        help="If True, enable gradient checkpointing in model blocks.",
    )
    parser.add_argument(
        "--recompute_seq_len_threshold",
        type=int,
        default=0,
        help="If >0, enable recompute when max_seq_len exceeds this threshold.",
    )

    ###########################################################################
    ###### Checkpointing ######################################################
    ###########################################################################
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Directory for checkpoint saving and loading")
    parser.add_argument("--save_temp_every", default=50, type=int, help="Steps between temporary checkpoints")
    parser.add_argument("--save_perm_every", default=5000, type=int, help="Steps between permanent checkpoints")
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=0,
        help="Maximum number of temporary checkpoints to keep. Set 0 to keep every checkpoint.",
    )
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to specific checkpoint file to load")
    parser.add_argument("--only_load_model", default=False, type=str2bool, help="Whether to only load model weights")

    # Exact configuration surface used by the official TabICLv2 GraphSCM.
    PriorConfig.add_args_to_parser(parser)

    parser.add_argument("--regression_method", choices=("quantile",), default="quantile")
    parser.add_argument("--num_quantiles", type=int, default=999)
    return parser
