#!/usr/bin/env bash
set -euo pipefail

STAGE=${T25_FULL_STAGE:?new immutable full-G5SC stage is required}
PASSES=${T25_LOOP_PASSES:?Loop3 or Loop4 is required}
ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1
[[ "${STAGE}" == "${ROOT}/stage/t25_fullg5sc_loop34_bg64_20260912_v"* ]]
[[ "${PASSES}" == 3 || "${PASSES}" == 4 ]]
EXPECTED_JOB=t25g5sc${PASSES}v2
test "${SLURM_JOB_NAME:?}" = "${EXPECTED_JOB}"
test "${SLURM_NNODES:?}" -eq 8
test "${SLURM_NTASKS:?}" -eq 8
test "${SLURM_RESTART_COUNT:-0}" -eq 0
source "${STAGE}/frozen_environment.sh"
PYTHON_BIN=${T25_PYTHON_BIN:?frozen Python executable is required}
test -x "${PYTHON_BIN}"
export PYTHONPATH="${STAGE}:${STAGE}/source/src"
export T25_FULL_STAGE="${STAGE}" T25_LOOP_PASSES="${PASSES}"
export SHARED_DEPTH_ICL_ENABLED=True SHARED_DEPTH_ICL_DATASET_CONDITIONED=True
export SHARED_DEPTH_ICL_NUM_PASSES="${PASSES}"
export TRAINING_STAGE_MANIFEST_PATH="${STAGE}/source.sha256"
export TRAINING_STAGE_MANIFEST_SHA256="$(sha256sum "${STAGE}/source.sha256" | cut -d' ' -f1)"
export SLURM_EXPORT_ENV=ALL

STAGE_NAME=${STAGE##*/}
ARM_NAME=${STAGE_NAME/loop34_/loop${PASSES}_}
CHECKPOINT_BASE=${ROOT}/checkpoints/${ARM_NAME}
LOG_BASE=${ROOT}/logs/${ARM_NAME}
RUN_DIR=${CHECKPOINT_BASE}/${EXPECTED_JOB}-${SLURM_JOB_ID:?}
test ! -e "${RUN_DIR}"
"${PYTHON_BIN}" "${STAGE}/validate_launch.py" --stage "${STAGE}" --passes "${PASSES}" --account "${SLURM_JOB_ACCOUNT:?}"
mkdir -p "${CHECKPOINT_BASE}" "${LOG_BASE}"
mkdir "${RUN_DIR}"
export T25_RUN_DIR="${RUN_DIR}" PYTHON_BIN
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT=$((41000 + SLURM_JOB_ID % 1000))
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,BOOTSTRAP
export NCCL_DEBUG_FILE="${LOG_BASE}/nccl-${SLURM_JOB_ID}-%h-%p.log"
export NCCL_SOCKET_IFNAME=enp69s0f0np0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

srun --nodes=8 --ntasks=8 --ntasks-per-node=1 --cpus-per-task=128 --kill-on-bad-exit=1 --label \
  bash -c '
    set -euo pipefail
    exec "${PYTHON_BIN}" -m torch.distributed.run \
      --nnodes=8 --nproc_per_node=8 --node_rank="${SLURM_PROCID}" \
      --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
      "${T25_FULL_STAGE}/validate_launch.py" --stage "${T25_FULL_STAGE}" \
      --passes "${T25_LOOP_PASSES}" --checkpoint-dir "${T25_RUN_DIR}" --exec-training
  '

"${PYTHON_BIN}" "${STAGE}/validate_launch.py" --stage "${STAGE}" --passes "${PASSES}" \
  --checkpoint-dir "${RUN_DIR}" --check-complete
