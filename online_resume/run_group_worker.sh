#!/usr/bin/env bash
set -euo pipefail
R=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1
STAGE=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/e4_g5sc_loop3_resume181407_eval_fp32_online_24gpu_gt_step50_20260910_v1/online_resume
SOURCE_STAGE=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/synthetic96_rw_sample50_stage2_true_rw50_eval_fp32_gpu_shard20_v1
V1_STAGE=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/rw_sample50_gpu_shard_validation1_v1
PROD_STAGE=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/synthetic96_rw_sample50_stage2_true_rw50_eval_fp32_online_20gpu_v1
EVAL_PROJECT=/vast/users/guangyi.chen/causal_group/zijian.li/tabicl_causal/tabicl-main-paper2602-dataset
TRAIN_ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/e4_g5_support_condition_alpha_loops_base_20260907_v1/source
DATA_ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/tabicl_causal/data178
CACHE_ROOT=${EVAL_PROJECT}/evaluation_results/official_tabiclv2_data178/_dataset_cache
CHECKPOINT_ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/checkpoints/e4_g5_support_condition_alpha_loops_20260907_v1/g36-g5scalpha-loop3-histe4-25k-v1/e4g5sc3lr1-178786
RESUME_CHECKPOINT_ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/checkpoints/e4_g5_support_condition_alpha_loops_20260907_v1/g36-g5scalpha-loop3-histe4-25k-v1/e4g5sc3lr2-181407
OUTPUT_ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/evaluation/e4_g5sc_loop3_resume181407_fp32_online_24gpu_gt_step50_20260910_v1/E4_G5SC_LOOP3/lineage-178786-181407
JOB_ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/analysis/e4_g5sc_loop3_resume181407_eval_fp32_online_24gpu_gt_step50_20260910_v1/E4_G5SC_LOOP3/lineage-178786-181407/job-${SLURM_JOB_ID}
TASK=${SLURM_PROCID:?single-GPU task index is required}
test "${SLURM_JOB_NAME}" = e4g5sc3r24gt1
test "${SLURM_NNODES}" -eq 4
test "${SLURM_NTASKS}" -eq 24
test "${TASK}" -ge 0
test "${TASK}" -lt 24
source "${EVAL_PROJECT}/scripts/activate_local_conda.sh"
export TABICL_EVAL_DISABLE_LOCAL_SRC=1 MODEL_SOURCE_ROOT="${TRAIN_ROOT}"
export CLS11_ACTIVE_SLOT_ENABLED=False REP6_ENABLED=False PRI2_PAIRED_EVIDENCE_ENABLED=False DGP9_SPLIT_POLICY_ENABLED=False
export PYTHONPATH="${TRAIN_ROOT}/src:${PROD_STAGE}/evaluator_fp32_v1:${PROD_STAGE}:${EVAL_PROJECT}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TORCH_SHOW_CPP_STACKTRACES=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python3 - "${JOB_ROOT}" "${TASK}" "${TRAIN_ROOT}" "${CHECKPOINT_ROOT}" "${RESUME_CHECKPOINT_ROOT}" <<'PY'
import json, os, sys
from pathlib import Path
import torch
root, task, train_root = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
old_root, new_root = Path(sys.argv[4]), Path(sys.argv[5])
resume = json.loads((new_root / "resume_record.json").read_text())
assert resume["old_job"] == 178786 and resume["step"] == 14500
assert resume["environment"]["G5_LOOP_PASSES"] == "3"
assert resume["source_checkpoint"] == str(old_root / "step-14500.ckpt")
assert resume["all_checkpoint_states_verified_equal"] is True
count = torch.cuda.device_count()
if not torch.cuda.is_available() or count != 1:
    raise RuntimeError(f"expected exactly one bound ROCm GPU; available={torch.cuda.is_available()} count={count} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r} ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES', '')!r} HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '')!r}")
payload = {
    "task": task,
    "checkpoint_roots": [str(old_root), str(new_root)],
    "step_range": [8850, 25000, 50],
    "resume_boundary": 14500,
    "loop_passes": 3,
    "explicit_fp32": True,
    "clf_use_amp": False,
    "clf_use_fa3": False,
    "cuda_available": True,
    "device_count": count,
    "device_name": torch.cuda.get_device_name(0),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
    "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
    "model_source_root": train_root,
    "single_gpu_binding_verified": True,
}
path = root / f"task-{task}" / "preflight.json"
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
python3 "${STAGE}/gpu_shard_group_scheduler_stage1.py"   --task-index "${TASK}" --group-count 6   --checkpoint-root "${CHECKPOINT_ROOT}" --resume-checkpoint-root "${RESUME_CHECKPOINT_ROOT}" --output-root "${OUTPUT_ROOT}"   --job-root "${JOB_ROOT}" --claims-root "${OUTPUT_ROOT}/.claims-v1"   --lock-path "${OUTPUT_ROOT}/.claim.lock"   --gpu-shard-worker "${STAGE}/gpu_shard_worker_deterministic.py"   --merge-script "${SOURCE_STAGE}/merge_gpu_shards_production.py"   --shard-policy "${V1_STAGE}/gpu_shard_policy_lpt4.json"   --data-root "${DATA_ROOT}" --cache-root "${CACHE_ROOT}"   --evaluator-dir "${PROD_STAGE}/evaluator_fp32_v1"   --inner-batch-wrapper "${PROD_STAGE}/talent_eval_ckpt_per_gpu_inner_batch.py"   --inner-batch-policy "${PROD_STAGE}/walking_activity_safe_policy.json"   --checkpoint-stable-sec 60 --poll-sec 5 --cpu-threads 4

