#!/bin/bash
#SBATCH --job-name=swift2g8
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus=8
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
#SBATCH --nice=0
set -euo pipefail
[[ $# == 1 && "$1" = /* && -f "$1" ]] || { echo 'Expected absolute immutable runtime plan' >&2; exit 2; }
ALLOC8_PLAN="$1"
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
ALLOC8_INFO="$(python3 - "$ALLOC8_PLAN" <<'PY'
import hashlib,json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p['plan_id']==hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='plan_id'},sort_keys=True,allow_nan=False).encode()).hexdigest()
assert p['family']=='tabswift'
r=p['runtime_script']; f=pathlib.Path(r['path'])
assert f.is_absolute() and f.is_file() and not f.is_symlink()
assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
assert f.name=='foundation_alloc8.py' and pathlib.Path(p['worker_python']).is_absolute()
print(p['worker_python']);print(f)
PY
)"
mapfile -t ALLOC8_FIELDS <<< "$ALLOC8_INFO"
[[ ${#ALLOC8_FIELDS[@]} == 2 ]] || exit 2
exec "${ALLOC8_FIELDS[0]}" "${ALLOC8_FIELDS[1]}" launch-job --family tabswift --plan "$ALLOC8_PLAN"
