#!/usr/bin/env bash
set -euo pipefail

ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1
PROJECT=/vast/users/guangyi.chen/causal_group/zijian.li/tabicl_causal/new_tab/nothing3_clean_sp_lr01_1753412
STAGE=${ROOT}/stage/tabiclv2_reg_fixed4096_matrix_20260824_v1
BASE_STAGE=${ROOT}/stage/tabicl_regression_adapted_e4_20260822_v1
SOURCE_ROOT=${BASE_STAGE}/source
PYTHON_BIN=${PROJECT}/.conda_env/bin/python
PROFILE=${BASE_STAGE}/gt_regression_official_train_only_quantile_profile_v2.json
PROFILE_AUDIT=${BASE_STAGE}/gt_regression_official_train_only_audit_v2.json
ARM=${REGRESSION_EXPERIMENT_ARM:?REGRESSION_EXPERIMENT_ARM is required}

case "${ARM}" in
  T25)
    EXPECTED_JOB=rgt025k
    CHECKPOINT_BASE=${ROOT}/checkpoints/tabiclv2_reg_t25_4096_8node_20260824_v1
    LOG_BASE=${ROOT}/logs/tabiclv2_reg_t25_4096_8node_20260824_v1
    export REGRESSION_TL_LENGTH_CURRICULUM_ENABLED=false
    ;;
  TL)
    EXPECTED_JOB=rgtl25k
    CHECKPOINT_BASE=${ROOT}/checkpoints/tabiclv2_reg_tl_4096_8node_20260824_v1
    LOG_BASE=${ROOT}/logs/tabiclv2_reg_tl_4096_8node_20260824_v1
    export REGRESSION_TL_LENGTH_CURRICULUM_ENABLED=true
    ;;
  *)
    echo "unknown REGRESSION_EXPERIMENT_ARM=${ARM}" >&2
    exit 2
    ;;
esac

export REGRESSION_SAFE_TAIL_ENABLED=true
export REGRESSION_CROSS_TABLE_E4_ENABLED=false

test "${SLURM_JOB_NAME}" = "${EXPECTED_JOB}"
test "${SLURM_NNODES}" -eq 8
test "${SLURM_NTASKS}" -eq 8
test -x "${PYTHON_BIN}"
test -f "${SOURCE_ROOT}/src/tabicl/train/_run.py"
test -f "${SOURCE_ROOT}/src/tabicl/prior/_dataset.py"
test -f "${SOURCE_ROOT}/src/tabicl/prior/_regression_target_prior.py"
test -s "${PROFILE}"
test -s "${PROFILE_AUDIT}"
test -s "${STAGE}/regression_safe_tail_length_runtime_patch.py"

active_same_name=$(squeue -u "${USER}" -h -n "${EXPECTED_JOB}" -o '%i' | wc -l | tr -d ' ')
test "${active_same_name}" -eq 1
"${PYTHON_BIN}" "${STAGE}/validate_fixed4096_contract.py" --stage "${STAGE}" --base-stage "${BASE_STAGE}"

mkdir -p "${CHECKPOINT_BASE}" "${LOG_BASE}"
run_dir=${CHECKPOINT_BASE}/${EXPECTED_JOB}-${SLURM_JOB_ID}
test ! -e "${run_dir}"
mkdir -p "${run_dir}"

master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
master_port=$((41000 + SLURM_JOB_ID % 1000))
export MASTER_ADDR=${master_addr}
export MASTER_PORT=${master_port}
export PYTHONPATH=${STAGE}:${SOURCE_ROOT}/src:${PYTHONPATH:-}
export PYTHONHASHSEED=0
export PRIOR_NUM_WORKERS=4
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,BOOTSTRAP
export NCCL_DEBUG_FILE=${LOG_BASE}/nccl-${SLURM_JOB_ID}-%h-%p.log
export NCCL_SOCKET_IFNAME=enp69s0f0np0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" - "${run_dir}" "${PROFILE}" "${SOURCE_ROOT}" "${STAGE}" "${ARM}" <<'PY'
import hashlib, json, pathlib, sys
run_dir = pathlib.Path(sys.argv[1])
profile = pathlib.Path(sys.argv[2])
source = pathlib.Path(sys.argv[3])
stage = pathlib.Path(sys.argv[4])
arm = sys.argv[5]
tracked = [
    source / "src/tabicl/prior/_dataset.py",
    source / "src/tabicl/prior/_graph_scm.py",
    source / "src/tabicl/prior/_regression_target_prior.py",
    source / "src/tabicl/prior/_support_only_preprocessing.py",
    source / "src/tabicl/train/_run.py",
    stage / "regression_safe_tail_length_runtime_patch.py",
    stage / "sitecustomize.py",
]
payload = {
    "schema_version": 1,
    "arm": arm,
    "causal_anchor": "148137 regad4k8",
    "only_pair_difference": "TL length/support curriculum" if arm == "TL" else "none",
    "regression_target_prior": "rw_sample50",
    "regression_target_mix_probability": 0.25,
    "target_template_mapping": {
        "interior": "stored nonuniform levels with support empirical CDF",
        "tail": "bounded monotone tanh in template target space",
        "gamma": "0.20*min(1,sqrt(n_support/512))",
        "loss_clip": [-8.0, 8.0],
    },
    "length_curriculum": {
        "enabled": arm == "TL",
        "sequence_buckets": [[256, 1024, 0.15], [1025, 2048, 0.25], [2049, 3072, 0.20], [3073, 4096, 0.40]],
        "support_fraction_buckets": [[0.30, 0.50, 0.15], [0.50, 0.70, 0.35], [0.70, 0.90, 0.50]],
    },
    "training": {
        "batch_size_global": 1024,
        "batch_size_per_rank": 16,
        "micro_batch_size": 2,
        "max_seq_len": 4096,
        "max_steps": 25000,
        "checkpoint_interval_steps": 50,
        "expected_checkpoint_count": 500,
        "retain_all_checkpoints": True,
        "world_size": 64,
        "nodes": 8,
        "gpus_per_node": 8,
        "dtype": "float32",
        "optimizer": "Muon",
    },
    "preprocessing": {
        "X_fit_split": "support_only",
        "y_fit_split": "support_only",
        "query_targets_fit_statistics": False,
        "cross_table_e4_enabled": False,
    },
    "profile": {"path": str(profile), "size": profile.stat().st_size, "mtime_ns": profile.stat().st_mtime_ns},
    "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked},
}
(run_dir / "training_contract.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "[$(date -Is)] start job=${SLURM_JOB_ID} arm=${ARM} run_dir=${run_dir} master=${MASTER_ADDR}:${MASTER_PORT}" | tee "${run_dir}/launcher.log"

srun --nodes=8 --ntasks=8 --ntasks-per-node=1 --kill-on-bad-exit=1 --label \
  bash -lc '
    set -euo pipefail
    exec "'"${PYTHON_BIN}"'" -m torch.distributed.run \
      --nnodes="${SLURM_NNODES}" \
      --nproc_per_node=8 \
      --node_rank="${SLURM_PROCID}" \
      --master_addr="'"${MASTER_ADDR}"'" \
      --master_port="'"${MASTER_PORT}"'" \
      -m tabicl.train \
      --wandb_log False \
      --wandb_mode disabled \
      --wandb_project TabICLv2-Regression \
      --wandb_name "'"${EXPECTED_JOB}"'" \
      --device cuda \
      --dtype float32 \
      --np_seed 43 \
      --torch_seed 43 \
      --max_steps 25000 \
      --batch_size 1024 \
      --micro_batch_size 2 \
      --lr 8e-4 \
      --muon True \
      --beta1 0.9 \
      --weight_decay 0.01 \
      --use_cautious_wd False \
      --scheduler cosine_with_restarts \
      --warmup_proportion 0.01 \
      --cosine_num_cycles 1 \
      --cosine_amplitude_decay 1 \
      --cosine_lr_end 1e-7 \
      --gradient_clipping 10.0 \
      --regression_method quantile \
      --num_quantiles 999 \
      --prior_type graph_scm \
      --prior_device cpu \
      --n_jobs 4 \
      --batch_size_per_gp 4 \
      --min_features 1 \
      --max_features 100 \
      --max_seq_len 4096 \
      --min_train_size 0.3 \
      --max_train_size 0.9 \
      --seq_len_per_gp True \
      --graph_noise False \
      --filter_unpredictable_graphs True \
      --filter_unpredictable_datasets True \
      --allow_act_warping False \
      --min_n_nodes 2 \
      --max_n_nodes 32 \
      --cauchy_dag_offset 0.0 \
      --regression_target_prior rw_sample50 \
      --regression_target_profile "'"${PROFILE}"'" \
      --regression_target_mix_probability 0.25 \
      --embed_dim 128 \
      --col_num_blocks 3 \
      --col_nhead 8 \
      --col_num_inds 128 \
      --col_affine False \
      --col_feature_group same \
      --col_feature_group_size 3 \
      --col_target_aware True \
      --col_ssmax True \
      --row_num_blocks 3 \
      --row_nhead 8 \
      --row_num_cls 4 \
      --row_rope_base 100000 \
      --row_rope_interleaved False \
      --icl_num_blocks 12 \
      --icl_nhead 8 \
      --icl_ssmax True \
      --ssmax_type qassmax-mlp-elementwise \
      --ff_factor 2 \
      --norm_first True \
      --zero_init False \
      --use_flash_attn3 False \
      --norm_type layernorm_nobias \
      --checkpoint_dir "'"${run_dir}"'" \
      --save_temp_every 50 \
      --save_perm_every 500 \
      --max_checkpoints 0
  '

test -s "${run_dir}/step-25000.ckpt"
checkpoint_count=$(find "${run_dir}" -maxdepth 1 -type f -name 'step-*.ckpt' | wc -l | tr -d ' ')
test "${checkpoint_count}" -eq 500
"${PYTHON_BIN}" - "${run_dir}" <<'PY'
import pathlib, sys
run_dir = pathlib.Path(sys.argv[1])
actual = sorted(int(path.stem.split("-")[1]) for path in run_dir.glob("step-*.ckpt"))
assert actual == list(range(50, 25001, 50))
PY
printf 'job_id=%s arm=%s step=25000 checkpoints=500 status=complete\n' "${SLURM_JOB_ID}" "${ARM}" > "${run_dir}/training.complete"
echo "[$(date -Is)] complete run_dir=${run_dir}" | tee -a "${run_dir}/launcher.log"
