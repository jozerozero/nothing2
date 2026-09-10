#!/usr/bin/env bash
#SBATCH --job-name=e4g5sc4r8gt1
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=gtqos
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --exclude=auh7-1b-gpu-[185-193,195,203,205,207,215-216,225,227-229,233,243-244,246,248,260-262,266,276,279-281,286-287,290-292,296-298,307,310-311,315,318]
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --nice=0
#SBATCH --chdir=/vast/users/guangyi.chen
#SBATCH --output=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/logs/e4_g5sc_loop4_resume180825_eval_fp32_online_gt8_st4_step50_20260910_v1/slurm-%j.out
#SBATCH --error=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/logs/e4_g5sc_loop4_resume180825_eval_fp32_online_gt8_st4_step50_20260910_v1/slurm-%j.err
set -euo pipefail
STAGE=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/e4_g5sc_loop4_resume180825_eval_fp32_online_gt8_st4_step50_20260910_v1/online_resume
SOURCE_STAGE=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/synthetic96_rw_sample50_stage2_true_rw50_eval_fp32_gpu_shard20_v1
V1_STAGE=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/rw_sample50_gpu_shard_validation1_v1
PROD_STAGE=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/synthetic96_rw_sample50_stage2_true_rw50_eval_fp32_online_20gpu_v1
OUTPUT_ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/evaluation/e4_g5sc_loop4_resume180825_fp32_online_gt8_st4_step50_20260910_v1/E4_G5SC_LOOP4/lineage-177623-180825
JOB_ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/analysis/e4_g5sc_loop4_resume180825_eval_fp32_online_gt8_st4_step50_20260910_v1/E4_G5SC_LOOP4/lineage-177623-180825/job-${SLURM_JOB_ID}
LOG_ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/logs/e4_g5sc_loop4_resume180825_eval_fp32_online_gt8_st4_step50_20260910_v1/E4_G5SC_LOOP4/lineage-177623-180825/job-${SLURM_JOB_ID}
test "${SLURM_JOB_NAME}" = e4g5sc4r8gt1
test "${SLURM_NNODES}" -eq 2
test "${SLURM_NTASKS}" -eq 8
test "${SLURM_JOB_QOS}" = gtqos
(cd "${SOURCE_STAGE}" && sha256sum -c source.sha256)
(cd "${V1_STAGE}" && sha256sum -c source.sha256)
(cd "${PROD_STAGE}" && sha256sum -c source.sha256)
(cd "${STAGE}" && sha256sum -c source.sha256)
mkdir -p "${OUTPUT_ROOT}" "${JOB_ROOT}" "${LOG_ROOT}"
python3 "${STAGE}/contract.py"
exec 9>"${OUTPUT_ROOT}/.evaluator-${SLURM_JOB_NAME}.lock"
flock -n 9 || { echo "evaluator lock held" >&2; exit 2; }
srun --gpu-bind=single:1 --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=4   --kill-on-bad-exit=1 --wait=0   --output="${LOG_ROOT}/group-task-%t-${SLURM_JOB_ID}.out"   --error="${LOG_ROOT}/group-task-%t-${SLURM_JOB_ID}.err"   bash "${STAGE}/run_group_worker.sh"
for task in $(seq 0 7); do test -s "${JOB_ROOT}/task-${task}/task.complete"; done
python3 - "${OUTPUT_ROOT}" <<'PY'
import csv, math, sys, time
from pathlib import Path
root = Path(sys.argv[1])
expected = [root / f"step-{step}" / "talent_detailed.txt" for step in range(5150, 25001, 50)]
def scan():
    signatures = {}
    for panel in expected:
        with panel.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        names = [row["dataset"] for row in rows]
        values = [float(row["accuracy"]) for row in rows]
        if len(rows) != 178 or len(set(names)) != 178 or not all(map(math.isfinite, values)):
            raise SystemExit(f"not strict Exact178: {panel}")
        stat = panel.stat(); signatures[str(panel)] = (stat.st_size, stat.st_mtime_ns)
    return signatures
first = scan(); time.sleep(12); second = scan()
if first != second: raise SystemExit("strict panels changed across stability scan")
print("strict_exact178_panels=398 stable_scan_sec=12 explicit_fp32=1 amp=0 fa3=0 complete=1 protocol_validation=1")
PY
printf 'job_id=%s training_lineage=177623,180825 groups=2 shards=4 gpus=8 strict=398 explicit_fp32=1 complete=1
'   "${SLURM_JOB_ID}" >"${JOB_ROOT}/job.complete"


