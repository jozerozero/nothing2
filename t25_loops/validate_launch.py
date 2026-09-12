"""Read-only validation; never write into the original T25/base stage."""
import argparse
import ast
import json
import os
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--stage',type=Path,required=True)
p.add_argument('--base-stage',type=Path,required=True)
a=p.parse_args()
assert os.environ['REGRESSION_EXPERIMENT_ARM']=='T25'
assert os.environ['REGRESSION_SAFE_TAIL_ENABLED']=='true'
assert os.environ['REGRESSION_TL_LENGTH_CURRICULUM_ENABLED']=='false'
assert os.environ['REGRESSION_CROSS_TABLE_E4_ENABLED']=='false'
passes=int(os.environ['T25_LOOP_PASSES'])
assert passes in (3,4)
assert (a.stage/'source_contract.json').is_file()
source=a.stage/'source/src/tabicl'
config=(source/'train/_run.py').read_text()
assert '"shared_depth_icl_num_passes": self.config.shared_depth_icl_num_passes' in config
profile=json.loads((a.base_stage/'gt_regression_official_train_only_quantile_profile_v2.json').read_text())
assert isinstance(profile,dict)
for receipt in ['cpu_smoke.json']:
    r=json.loads((a.stage/receipt).read_text())
    assert r['status']=='PASS_T25_G5SC_LOOP34_SMOKE',r
print(json.dumps({'status':'PASS_LAUNCH_CONTRACT','passes':passes,'baseline':151162}))
