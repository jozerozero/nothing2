"""Generate the two new T25 launchers without changing any baseline allocation."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from prepare_source import HERE, prepare, replace

ROOT = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
BASE = ROOT/'stage/tabicl_regression_adapted_e4_20260822_v1'
STAGE = ROOT/'stage/t25_g5sc_loop34_bg64_20260912_v1'
PYTHON = '/vast/users/guangyi.chen/causal_group/zijian.li/tabicl_causal/new_tab/nothing3_clean_sp_lr01_1753412/.conda_env/bin/python'


def main():
    assert not STAGE.exists(), f'refuse overwrite existing stage {STAGE}'
    STAGE.mkdir()
    prepare(BASE/'source', STAGE/'source')
    for name in ['regression_safe_tail_length_runtime_patch.py','sitecustomize.py','test_smoke.py','validate_launch.py']:
        shutil.copyfile(HERE/name, STAGE/name)
    baseline = (HERE/'run_tabiclv2_reg_fixed4096_8node_v1.sh').read_text()
    runtime = replace(baseline, 'STAGE=${ROOT}/stage/tabiclv2_reg_fixed4096_matrix_20260824_v1',
                      f'STAGE={STAGE}')
    runtime = replace(runtime, 'SOURCE_ROOT=${BASE_STAGE}/source', 'SOURCE_ROOT=${STAGE}/source')
    runtime = replace(runtime, 'ARM=${REGRESSION_EXPERIMENT_ARM:?REGRESSION_EXPERIMENT_ARM is required}',
                      'ARM=${REGRESSION_EXPERIMENT_ARM:?REGRESSION_EXPERIMENT_ARM is required}\n'
                      'PASSES=${T25_LOOP_PASSES:?T25_LOOP_PASSES is required}\n'
                      'test "${ARM}" = T25\n[[ "${PASSES}" = 3 || "${PASSES}" = 4 ]]')
    runtime = replace(runtime, 'EXPECTED_JOB=rgt025k', 'EXPECTED_JOB=rgt25sc${PASSES}l1')
    runtime = runtime.replace('tabiclv2_reg_t25_4096_8node_20260824_v1', 't25_g5sc_loop${PASSES}_bg64_20260912_v1')
    runtime = replace(runtime, 'validate_fixed4096_contract.py', 'validate_launch.py')
    runtime = replace(runtime, '      --icl_num_blocks 12 \\',
                      '      --shared_depth_icl_num_passes "\'"${PASSES}"\'" \\\n      --icl_num_blocks 12 \\')
    runtime = replace(runtime, 'import hashlib, json, pathlib, sys', 'import hashlib, json, pathlib, sys, os')
    runtime = replace(runtime, '    "only_pair_difference": "TL length/support curriculum" if arm == "TL" else "none",',
        '    "only_pair_difference": "shared_depth_icl_num_passes=3 versus 4",\n'
        '    "baseline_training": 151162,\n'
        '    "loop": {"passes": int(os.environ["T25_LOOP_PASSES"]), "icl_blocks": 12,\n'
        '             "gate": "tanh(a+0.1*(2*sigmoid(w@support_stats51)-1))",\n'
        '             "initialization": "a=0,w=zeros(51); original tensors unchanged",\n'
        '             "extra_pass_activation_recomputation": True},')
    runtime = replace(runtime, '    stage / "sitecustomize.py",',
        '    stage / "sitecustomize.py",\n'
        '    source / "src/tabicl/_model/g5sc_regression_loop.py",\n'
        '    source / "src/tabicl/_model/g5sc_support_stats.py",\n'
        '    source / "src/tabicl/_model/tabicl.py",\n'
        '    source / "src/tabicl/_model/learning.py",\n'
        '    source / "src/tabicl/train/_muon.py",')
    (STAGE/'run_training.sh').write_text(runtime)
    subprocess.run(['bash','-n',str(STAGE/'run_training.sh')],check=True)
    for passes in [3,4]:
        name = f'rgt25sc{passes}l1'
        logs = ROOT/f'logs/t25_g5sc_loop{passes}_bg64_20260912_v1'
        logs.mkdir(parents=True,exist_ok=True)
        slurm = (HERE/'slurm_tabiclv2_reg_t25_4096_8node_v1.sh').read_text()
        slurm = replace(slurm, '--job-name=rgt025k', f'--job-name={name}')
        slurm = slurm.replace('tabiclv2_reg_t25_4096_8node_20260824_v1', f't25_g5sc_loop{passes}_bg64_20260912_v1')
        slurm = replace(slurm, '#SBATCH --no-requeue', '#SBATCH --no-requeue\n#SBATCH --nice=0\n#SBATCH --chdir='+str(STAGE))
        slurm = replace(slurm, '185-192,203,207,215,225,227,233,243-244,246,248,261-262,276,279-281,286,290-292,297-298,311,315',
                               '185-193,195,203,207,215-216,225,227-228,233,243-244,246,248,261-262,276,279-281,286-287,290-292,296-298,311,315')
        slurm = replace(slurm, 'export REGRESSION_EXPERIMENT_ARM=T25',
                        f'export REGRESSION_EXPERIMENT_ARM=T25\nexport T25_LOOP_PASSES={passes}')
        slurm = replace(slurm, str(ROOT/'stage/tabiclv2_reg_fixed4096_matrix_20260824_v1/run_tabiclv2_reg_fixed4096_8node_v1.sh'),
                        str(STAGE/'run_training.sh'))
        p = STAGE/f'loop{passes}.slurm'
        p.write_text(slurm)
        subprocess.run(['bash','-n',str(p)],check=True)
    for p in STAGE.glob('*.sh'):
        p.chmod(0o755)
    print(json.dumps({'prepared':str(STAGE),'submitted':False}))


if __name__=='__main__':
    main()
