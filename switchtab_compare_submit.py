"""Submit only unfinished SwitchTab work; TabFM's completed681 is reused."""
import datetime,hashlib,json,os,subprocess
from pathlib import Path

ROOT=Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
STAGE=Path(__file__).resolve().parent
REPO=ROOT/'stage/reg_loop3_step22175_finetune50_20260921_v1/repo'
BASE=ROOT/'stage/table6_remaining19_standard_hpo_bg8_20260912_v1'
LEGACY=ROOT/'stage/table6_nonfoundation_all_benchmarks_20260823_v1'
OUT=ROOT/'evaluation/switchtab_tabfm_comparison_20260923_v1'
PY='/vast/users/guangyi.chen/anaconda3/envs/tabicl/bin/python'
PLAN=ROOT/'evaluation/table6_missing190_standard_hpo_20260922_v1/plan.json'
EXCLUDE='auh7-1b-gpu-[193,195,207,216,228,239,274,287,292,296]'

def run(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=60)
 if p.returncode:raise RuntimeError((args,p.returncode,p.stdout,p.stderr))
 return p.stdout.strip()

def write_new(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2)

def script(kind):
 n=4 if kind=='classification' else 1
 cpus=16 if kind=='classification' else 8
 memory=256 if kind=='classification' else 64
 limit='02:00:00' if kind=='classification' else '01:00:00'
 name='swtab23gap' if kind=='classification' else 'swtab23bng'
 common=f'''#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --partition=faculty
#SBATCH --account=faculty-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks={n}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus-per-task=1
#SBATCH --mem={memory}G
#SBATCH --time={limit}
#SBATCH --signal=USR1@90
#SBATCH --no-requeue
#SBATCH --exclude={EXCLUDE}
#SBATCH --chdir={STAGE}
#SBATCH --output={OUT}/logs/{kind}-%j.out
#SBATCH --error={OUT}/logs/{kind}-%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
'''
 if kind=='classification':
  common+=f'''export PYTHONPATH={BASE}:{BASE}/TALENT:{LEGACY}/python_packages
export T6_BASE_STAGE={BASE} T6_MISSING190_PLAN={PLAN}
BUDGET_EXPORTS="$({PY} -B {REPO}/table6_restart_deadline.py --job-id "$SLURM_JOB_ID" --format exports)"
eval "$BUDGET_EXPORTS"
{PY} -B {BASE}/manage.py check-inputs
srun --exact --exclusive --input=none --nodes=1 --ntasks=4 --cpus-per-task=16 --gpus-per-task=1 --cpu-bind=cores --gpu-bind=single:1 --kill-on-bad-exit=1 {PY} {BASE}/rocm_gpu_entry.py {PY} -B {REPO}/table6_missing190_worker.py preflight --plan {PLAN} --mode gpu
{PY} -B {REPO}/table6_missing190_worker.py gate --plan {PLAN} --ranks 4
srun --exact --exclusive --input=none --nodes=1 --ntasks=4 --cpus-per-task=16 --gpus-per-task=1 --cpu-bind=cores --gpu-bind=single:1 --kill-on-bad-exit=1 {PY} {BASE}/rocm_gpu_entry.py {PY} -B {STAGE}/switchtab_only_missing190.py --repo {REPO} --plan {PLAN}
'''
 else:
  common+=f'''export PYTHONPATH={LEGACY}:{LEGACY}/TALENT:{LEGACY}/python_packages
srun --exact --exclusive --input=none --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --cpu-bind=cores --gpu-bind=single:1 --kill-on-bad-exit=1 {PY} {BASE}/rocm_gpu_entry.py {PY} -B {STAGE}/switchtab_bng_coverage_extension.py --legacy-stage {LEGACY} --canonical-manifest {ROOT}/stage/reg_loop3_step22175_finetune50_20260921_v1/eval224/manifest.json --reference-result {ROOT}/evaluation/tabfm_defaults_standard681_20260922_v1/results/regression/row-009.json --output {OUT}/switchtab_bng_mv.json
'''
 return common

def submit(kind):
 receipt=OUT/(kind+'_submission.json')
 if receipt.exists():return json.loads(receipt.read_text())
 name='swtab23gap' if kind=='classification' else 'swtab23bng'
 queue=run(['squeue','-u','guangyi.chen','-h','-o','%i|%j|%T'])
 assert not any('|'+name+'|' in line for line in queue.splitlines()),'Matching job already exists'
 p=OUT/(kind+'.sh')
 with p.open('x') as f:f.write(script(kind))
 run(['bash','-n',str(p)])
 attempt={'state':'submitting','kind':kind,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_commit':run(['git','-C',str(STAGE),'rev-parse','HEAD'])}
 write_new(OUT/(kind+'_attempt.json'),attempt)
 job=run(['sbatch','--hold','--parsable',str(p)]).split(';')[0]
 assert job.isdigit(),job
 held=run(['scontrol','show','job','-o',job])
 write_new(receipt,{**attempt,'state':'held','job_id':job,'control':held})
 assert 'JobState=PENDING' in held and 'Reason=JobHeldUser' in held and 'JobName='+name+' ' in held
 assert 'NumCPUs='+('64' if kind=='classification' else '8')+' ' in held
 assert 'gres/gpu='+('4' if kind=='classification' else '1') in held
 run(['scontrol','release',job])
 observed=run(['scontrol','show','job','-o',job])
 release={**attempt,'state':'released','job_id':job,'control':observed}
 write_new(OUT/(kind+'_release.json'),release)
 return release

if __name__=='__main__':
 assert STAGE.name=='switchtab_tabfm_compare_20260923_v1',STAGE
 (OUT/'logs').mkdir(parents=True,exist_ok=True)
 print(json.dumps({'classification':submit('classification'),'regression':submit('regression')},indent=2),flush=True)
