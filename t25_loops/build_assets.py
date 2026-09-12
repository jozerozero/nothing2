"""Copy frozen local templates into the isolated, versioned deployment payload."""
import ast
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
BASE = WORK/'outputs/tabiclv2_reg_fixed4096_matrix_20260824_v1'
G5 = WORK/'outputs/e4_g5_support_condition_alpha_loops_bgqos_20260907_v1/base_stage/source/src/tabicl/_model/schema_expert.py'
assert BASE.is_dir() and G5.is_file()
for name in ['regression_safe_tail_length_runtime_patch.py', 'sitecustomize.py',
             'run_tabiclv2_reg_fixed4096_8node_v1.sh', 'slurm_tabiclv2_reg_t25_4096_8node_v1.sh']:
    shutil.copyfile(BASE/name, HERE/name)
text = G5.read_text()
lines = text.splitlines(keepends=True)
pieces = ['from __future__ import annotations\nimport math\nfrom typing import Optional\nimport torch\nfrom torch import Tensor, nn\n\n']
for node in ast.parse(text).body:
    if isinstance(node, ast.ClassDef) and node.name in ['SupportSchemaEncoder', 'SupportSchemaStatistics']:
        pieces.append(''.join(lines[node.lineno-1:node.end_lineno])+'\n\n')
assert len(pieces) == 3
(HERE/'support_stats.py').write_text(''.join(pieces))
print(json.dumps({'copied_original_T25_templates':True, 'copied_exact_G5SC_support_stats':True}))
