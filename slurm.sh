#!/usr/bin/env bash
#SBATCH --job-name=g5sc19750l34
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --time=3-00:00:00
#SBATCH --nice=0
#SBATCH --no-requeue
#SBATCH --exclude=auh7-1b-gpu-[193,195,216,228,287,296]
#SBATCH --output=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/logs/g5sc19750_inference_loop34_gt8_20260908_v1/slurm-%j.out
#SBATCH --error=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/logs/g5sc19750_inference_loop34_gt8_20260908_v1/slurm-%j.err
set -euo pipefail
STAGE=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/g5sc19750_inference_loop34_gt8_20260908_v1
LOG=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/logs/g5sc19750_inference_loop34_gt8_20260908_v1
test "${SLURM_NNODES}" = 1
test "${SLURM_NTASKS}" = 8
test "${SLURM_CPUS_PER_TASK}" = 4
test "${SLURM_JOB_QOS}" = gtqos
test "$(squeue --me -h -n g5sc19750l34 -o '%i' | wc -l | tr -d ' ')" = 1
source /vast/users/guangyi.chen/causal_group/zijian.li/tabicl_causal/tabicl-main-paper2602-dataset/scripts/activate_local_conda.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CROSS_TABLE_ARM=E4 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
cd "${STAGE}"
srun --nodes=1 --ntasks=8 --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-task=4 \
  --cpu-bind=cores --gpu-bind=single:1 --kill-on-bad-exit=1 \
  --output="${LOG}/gpu-${SLURM_JOB_ID}-%t.out" --error="${LOG}/gpu-${SLURM_JOB_ID}-%t.err" python gpu_probe.py
for spec in old:2 loop:2 loop:3 loop:4; do
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-task=4 \
    --cpu-bind=cores --gpu-bind=single:1 --kill-on-bad-exit=1 \
    python model_probe.py --source "${spec%:*}" --passes "${spec#*:}"
done
python verify_preflight.py
srun --nodes=1 --ntasks=8 --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-task=4 \
  --cpu-bind=cores --gpu-bind=single:1 --kill-on-bad-exit=1 \
  --output="${LOG}/worker-${SLURM_JOB_ID}-%t.out" --error="${LOG}/worker-${SLURM_JOB_ID}-%t.err" python run_worker.py
python aggregate.py
echo '__G5SC19750_LOOP34_COMPLETE__'
