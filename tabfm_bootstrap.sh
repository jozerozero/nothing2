#!/bin/bash
# Isolated research dependency/weight setup. Existing environments are untouched.
set -euo pipefail
BASE=/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1
TF_STAGE=$BASE/stage/tabfm_defaults_standard681_20260922_v1
TF_BASEPY=/vast/users/guangyi.chen/causal_group/zijian.li/tabicl_causal/tabicl-main-paper2602-dataset/.conda_env/bin/python3
TF_COMMIT=fbb665569425fd2f490c6576b3af967876fe11ff
mkdir -p "$TF_STAGE"
if [ ! -d "$TF_STAGE/official/.git" ]; then
  git clone --depth=1 https://github.com/google-research/tabfm.git "$TF_STAGE/official"
fi
git -C "$TF_STAGE/official" checkout --detach "$TF_COMMIT"
test "$(git -C "$TF_STAGE/official" rev-parse HEAD)" = "$TF_COMMIT"
if [ ! -x "$TF_STAGE/venv/bin/python" ]; then
  "$TF_BASEPY" -m venv --system-site-packages "$TF_STAGE/venv"
fi
"$TF_STAGE/venv/bin/python" -m pip install 'absl-py' 'jaxtyping<0.3' 'typeguard<3'
"$TF_STAGE/venv/bin/python" -m pip install --no-deps -e "$TF_STAGE/official"
"$TF_STAGE/venv/bin/python" -c 'from huggingface_hub import snapshot_download; import sys; print(snapshot_download(repo_id="google/tabfm-1.0.0-pytorch", revision="77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83", local_dir=sys.argv[1], allow_patterns=["*.json","*.md","LICENSE","classification/*","regression/*"]))' "$TF_STAGE/weights"
"$TF_STAGE/venv/bin/python" -c 'import torch,sklearn,tabfm; from sklearn.utils.validation import validate_data; from tabfm import TabFMClassifier,TabFMRegressor,tabfm_v1_0_0_pytorch; print({"torch":torch.__version__,"hip":torch.version.hip,"sklearn":sklearn.__version__,"tabfm":tabfm.__version__,"source":tabfm.__file__}); assert torch.version.hip'
