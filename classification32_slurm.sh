#!/bin/bash
set -euo pipefail
test "${CLASS32_SHARD:?}" -ge 0
test "$CLASS32_SHARD" -le 3
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
CLASS32_PY=/vast/users/guangyi.chen/causal_group/zijian.li/tabicl_causal/tabicl-main-paper2602-dataset/.conda_env/bin/python3
CLASS32_REPO=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/classification_actual32_20260921_v1/repo
cd "$CLASS32_REPO"
srun --exact --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=1 "$CLASS32_PY" classification32_dispatch.py preflight
"$CLASS32_PY" classification32_dispatch.py check
srun --exact --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=1 "$CLASS32_PY" classification32_dispatch.py smoke --shard "$CLASS32_SHARD"
srun --exact --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=1 "$CLASS32_PY" classification32_dispatch.py run
