#!/bin/bash
#SBATCH --job-name=swift681x2
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

# The held submission helper supplies campaign-specific --chdir/--output/--error.
if [[ $# -ne 1 || "$1" != /* || ! -f "$1" ]]; then
  echo 'Usage: tabswift_slurm.sh /absolute/path/to/tabswift_standard681_20260922_v1/plan.json' >&2
  exit 2
fi
TABSWIFT_PLAN_PATH="$1"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

TABSWIFT_PYTHON="$("${TABSWIFT_BOOTSTRAP_PYTHON:-python3}" -c 'import json,os,pathlib,sys; p=pathlib.Path(json.load(open(sys.argv[1]))["worker_python"]); assert p.is_absolute() and p.is_file() and os.access(p,os.X_OK); print(p)' "$TABSWIFT_PLAN_PATH")"
# BASH_SOURCE points into Slurm's spool, not the original frozen repository.
TABSWIFT_DISPATCH_PATH="$("$TABSWIFT_PYTHON" - "$TABSWIFT_PLAN_PATH" <<'PY'
import hashlib
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
digest = hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != 'plan_id'},
                                 sort_keys=True, allow_nan=False).encode()).hexdigest()
if plan.get('plan_id') != digest:
    raise RuntimeError('Plan content hash mismatch')
dispatch = []
for record in plan['source_records']:
    path = pathlib.Path(record['path'])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f'Invalid pinned source: {path}')
    before = path.stat()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    if ((before.st_ino, before.st_size, before.st_mtime_ns) !=
        (after.st_ino, after.st_size, after.st_mtime_ns) or
        after.st_size != record['size_bytes'] or after.st_mtime_ns != record['mtime_ns'] or
        actual != record['sha256']):
        raise RuntimeError(f'Frozen source changed: {path}')
    if path.name == 'tabswift_dispatch.py':
        dispatch.append(path)
if len(dispatch) != 1:
    raise RuntimeError('Exactly one pinned TabSwift dispatcher is required')
print(dispatch[0])
PY
)"
cd "$(dirname "$TABSWIFT_DISPATCH_PATH")"

TABSWIFT_SRUN=(srun --exact --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=1)
# Each of four ranks checks both manifests and runs both protocol smoke sets.
# The dispatcher then alternates the two protocols on its own pinned GPU.
"${TABSWIFT_SRUN[@]}" "$TABSWIFT_PYTHON" "$TABSWIFT_DISPATCH_PATH" preflight --plan "$TABSWIFT_PLAN_PATH"
"$TABSWIFT_PYTHON" "$TABSWIFT_DISPATCH_PATH" check --plan "$TABSWIFT_PLAN_PATH"
"${TABSWIFT_SRUN[@]}" "$TABSWIFT_PYTHON" "$TABSWIFT_DISPATCH_PATH" smoke --plan "$TABSWIFT_PLAN_PATH"
"$TABSWIFT_PYTHON" "$TABSWIFT_DISPATCH_PATH" check-smoke --plan "$TABSWIFT_PLAN_PATH"
"${TABSWIFT_SRUN[@]}" "$TABSWIFT_PYTHON" "$TABSWIFT_DISPATCH_PATH" run --plan "$TABSWIFT_PLAN_PATH"
"$TABSWIFT_PYTHON" "$TABSWIFT_DISPATCH_PATH" status --plan "$TABSWIFT_PLAN_PATH"
