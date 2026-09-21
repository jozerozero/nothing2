#!/usr/bin/env bash
set -euo pipefail
FT_ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/reg_loop3_step22175_finetune50_20260921_v1
FT_PY=/vast/users/guangyi.chen/anaconda3/envs/tabicl/bin/python
test "${SLURM_JOB_ID}" = 204827
test "${SLURM_NTASKS}" = 1
test "${SLURM_CPUS_PER_TASK}" = 4
test "$(hostname -s)" = auh7-1b-gpu-257
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH="${FT_ROOT}/repo/native_source/src"
printf 'FT50_START job=%s step=%s host=%s time=%s\n' "${SLURM_JOB_ID}" "${SLURM_STEP_ID}" "$(hostname -s)" "$(date -u +%FT%TZ)"
"${FT_PY}" "${FT_ROOT}/repo/finetune50.py" \
  --source "${FT_ROOT}/repo/native_source/src" \
  --checkpoint "${FT_ROOT}/input/step-22175.ckpt" \
  --manifest "${FT_ROOT}/train_data_v2/manifest.json" \
  --output "${FT_ROOT}/run50" \
  --learning-rate 0.000001 --support-max 128 --query-max 32
printf 'FT50_SUCCESS time=%s\n' "$(date -u +%FT%TZ)"
