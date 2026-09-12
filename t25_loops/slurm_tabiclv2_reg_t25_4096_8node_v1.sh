#!/usr/bin/env bash
#SBATCH --job-name=rgt025k
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=2T
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --exclude=auh7-1b-gpu-[185-192,203,207,215,225,227,233,243-244,246,248,261-262,276,279-281,286,290-292,297-298,311,315]
#SBATCH --output=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/logs/tabiclv2_reg_t25_4096_8node_20260824_v1/slurm-%j.out
#SBATCH --error=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/logs/tabiclv2_reg_t25_4096_8node_20260824_v1/slurm-%j.err

set -euo pipefail
export REGRESSION_EXPERIMENT_ARM=T25
exec /vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/tabiclv2_reg_fixed4096_matrix_20260824_v1/run_tabiclv2_reg_fixed4096_8node_v1.sh
