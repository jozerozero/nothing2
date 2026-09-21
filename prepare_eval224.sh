#!/usr/bin/env bash
set -euo pipefail
FT_ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1/stage/reg_loop3_step22175_finetune50_20260921_v1
ALL_ROOT=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1
MITRA_DATA_ROOT="$ALL_ROOT/stage/mitra_dual_standard681_bg1_20260915_v1"
EVAL_PY=/vast/users/guangyi.chen/anaconda3/envs/tabicl/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"$EVAL_PY" "$FT_ROOT/repo/prepare_eval224.py" \
  --source-plan "$ALL_ROOT/stage/table6_remaining19_standard_hpo_bg8_20260912_v1/plan.json" \
  --checkpoint-dir "$FT_ROOT/run50" \
  --finetune-contract "$FT_ROOT/run50/contract.json" \
  --finetune-complete "$FT_ROOT/run50/complete.json" \
  --source-checkpoint "$FT_ROOT/input/step-22175.ckpt" \
  --eval-data "$MITRA_DATA_ROOT/eval_data.py" \
  --vendor-dir "$MITRA_DATA_ROOT/vendor" \
  --vendor-manifest "$MITRA_DATA_ROOT/vendor_sources.json" \
  --native-source "$FT_ROOT/repo/native_source/src" \
  --inference-protocol "$FT_ROOT/repo/inference_protocol.json" \
  --output "$FT_ROOT/eval224/manifest.json"
"$EVAL_PY" "$FT_ROOT/repo/eval_dispatch.py" plan \
  --first "$FT_ROOT/eval224/capacity_a.json" \
  --second "$FT_ROOT/eval224/capacity_b.json" \
  --plan "$FT_ROOT/eval224/resource_plan.json"
