#!/bin/bash
#SBATCH --job-name=tabfm681
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --nice=0
#SBATCH --exclude=auh7-1b-gpu-[193,195,207,216,228,239,274,287,292,296]
set -euo pipefail

# Submission owns --output/--error under this campaign's logs and --chdir.
TABFM_CAMPAIGN_PATH="${1:-${TABFM_CAMPAIGN:-}}"
if [[ -z "$TABFM_CAMPAIGN_PATH" || ! -f "$TABFM_CAMPAIGN_PATH" ]]; then
  echo 'Usage: tabfm_default_slurm.sh /absolute/path/to/campaign/manifest.json' >&2
  exit 2
fi
TABFM_PYTHON="$("${TABFM_BOOTSTRAP_PYTHON:-python3}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["worker_python"])' "$TABFM_CAMPAIGN_PATH")"
TABFM_SCRIPT_DIR="$("$TABFM_PYTHON" -c 'import json,pathlib,sys; print(pathlib.Path(json.load(open(sys.argv[1]))["worker_script"]["path"]).resolve().parent)' "$TABFM_CAMPAIGN_PATH")"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
cd "$TABFM_SCRIPT_DIR"
TABFM_SRUN=(srun --exact --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=1)
"${TABFM_SRUN[@]}" "$TABFM_PYTHON" tabfm_default_dispatch.py preflight --campaign "$TABFM_CAMPAIGN_PATH"
"$TABFM_PYTHON" tabfm_default_dispatch.py check --campaign "$TABFM_CAMPAIGN_PATH"
"${TABFM_SRUN[@]}" "$TABFM_PYTHON" tabfm_default_dispatch.py smoke --campaign "$TABFM_CAMPAIGN_PATH"
"$TABFM_PYTHON" tabfm_default_dispatch.py check-smoke --campaign "$TABFM_CAMPAIGN_PATH"
"${TABFM_SRUN[@]}" "$TABFM_PYTHON" tabfm_default_dispatch.py run --campaign "$TABFM_CAMPAIGN_PATH"
"$TABFM_PYTHON" tabfm_default_dispatch.py status --campaign "$TABFM_CAMPAIGN_PATH"
