"""Freeze two scientifically independent TabSwift protocols on the existing681 splits."""
import json
import subprocess
import time
from pathlib import Path
from eval_one import object_digest, publish_new, require, verify_file
from tabfm_prepare import identity, verify_manifest
from tabswift_bootstrap import BASE, STAGE, COMMIT, REVISION, WEIGHT_SHA

FT = BASE/'stage/reg_loop3_step22175_finetune50_20260921_v1'
VARIANTS = ('official16','budget32x8')
SOURCES = ('tabswift_bootstrap.py','tabswift_prepare.py','tabswift_cpu_check.py','tabswift_one.py','tabswift_ensemble.py',
           'tabswift_dispatch.py','tabswift_submit.py','tabswift_slurm.sh',
           'eval_one.py','tabfm_prepare.py','tabfm_default_one.py','tabfm_default_dispatch.py',
           'tabfm_local_tmp.py','classification32_dispatch.py','classification32_campaign.py',
           'classification32_submit.py','pfn_mitra_one.py')

def read(path): return json.loads(Path(path).read_text())

def prepare():
    require(not (STAGE/'plan.json').exists(),'Plan already exists; never overwrite')
    repo = Path(__file__).resolve().parent
    boot = read(STAGE/'bootstrap_receipt.json')
    require(boot['source_commit']==COMMIT and boot['weight_sha256']==WEIGHT_SHA,'Bootstrap identity mismatch')
    parent_path = BASE/'evaluation/tabfm_defaults_standard681_20260922_v1/manifest.json'
    parent = read(parent_path); verify_manifest(parent)
    require(parent['manifest_id']=='0d34153c706a459621e31a42ce3fd9ce69ddcf1d1781f9cbf6440c9541fc65bd',
            'Unexpected standard split provenance')
    data = {}
    for kind,count in [('classification',457),('regression',224)]:
        verify_file(parent[kind+'_manifest'])
        data[kind] = read(parent[kind+'_manifest']['path']); verify_manifest(data[kind])
        require(len(data[kind]['rows'])==count and data[kind]['membership_count']==count,'Scope mismatch')
    official = STAGE/'official'
    require(subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()==COMMIT,
            'Official commit changed')
    require(not subprocess.check_output(['git','-C',str(official),'status','--porcelain'],text=True).strip(),
            'Official source was modified')
    official_files = sorted((official/'TALENT').rglob('*.py'))
    weight = identity(STAGE/'weights/swift.ckpt'); require(weight['sha256']==WEIGHT_SHA,'Weight changed')
    sources = [identity(repo/name) for name in SOURCES]
    cl = data['classification']['rows']; reg = data['regression']['rows']
    size = lambda r: (r['train_rows']+r['test_rows'])*r['features']
    binary = min((r for r in cl if r['classes']<=10),key=size)
    hierarchy = min((r for r in cl if 10<r['classes']<=100),key=size)
    lowdim = next((r for r in reg if r['dataset'].lower().endswith('quake')),None)
    require(lowdim is not None,'Low-dimensional regression smoke missing')
    # Shape receipts choose a smoke task only, never provide labels or model scores.
    highdim = []
    for row in reg:
        p = FT/'eval224/results/step-22175'/f"row-{row['dataset_index']:03d}.json"
        if p.exists():
            receipt = read(p)
            shape = receipt.get('data_audit',{})
            f = shape.get('features',shape.get('n_features',receipt.get('features',0)))
            if isinstance(f,int) and f>100:
                highdim.append(row)
    if highdim:
        high = min(highdim,key=lambda r:r['work_size'])
    else:
        high = next(r for r in reg if r['dataset']=='talent__us_crime')
    smoke = [{'task_kind':'classification','dataset_index':r['dataset_index']} for r in [binary,hierarchy]] + [
        {'task_kind':'regression','dataset_index':r['dataset_index']} for r in [lowdim,high]]
    manifests=[]
    for variant in VARIANTS:
        out=BASE/'evaluation'/f'tabswift_{variant}_standard681_20260922_v1'
        require(not (out/'manifest.json').exists(),'Variant already frozen')
        counts={'classification':16,'regression':16} if variant=='official16' else {'classification':32,'regression':8}
        man={'schema':1,'name':out.name,'created_epoch':time.time(),'output_root':str(out),
             'classification_manifest':parent['classification_manifest'],
             'regression_manifest':parent['regression_manifest'],
             'classification_raw_inputs':parent['classification_raw_inputs'],
             'raw_source_manifest':parent['raw_source_manifest'],
             'standard_split_parent_manifest':identity(parent_path),
             'official_source':{'path':str(official),'commit':COMMIT,'files':[identity(p) for p in official_files]},
             'weights':{'shared':weight},'weight_repository':'LAMDA-Tabular/TabSwift','weight_revision':REVISION,
             'worker_python':boot['worker_python'],'worker_script':identity(repo/'tabswift_one.py'),
             'worker_sources':sources,'versions':boot['versions'],
             'membership_count':681,'classification_count':457,'regression_count':224,
             'classification_suite_counts':data['classification'].get('suite_counts'),
             'regression_suite_counts':data['regression'].get('suite_counts'),
             'smoke_tasks':smoke,'hierarchical_classification_indices':[r['dataset_index'] for r in cl if r['classes']>10],
             'protocol':{'variant':variant,'n_estimators':counts,'strict_actual_count':variant=='budget32x8',
                'batch_size':16,'random_state':42,'class_shift':{'classification':True,'regression':False},
                'norm_methods':['none','power'],'feat_shuffle_method':'latin','outlier_threshold':4,
                'softmax_temperature':0.9,'average_logits':True,'use_hierarchical':True,'use_amp':True,
                'enable_dim_reduction':True,'pca_dim':100,'no_early_exit':True,
                'outer_preprocessing':'pinned TALENT mean numerical imputation, new categorical token, ordinal indices; support-only fit',
                'regression_targets':'official support mean/std then inverse to original units',
                'test_split':'unchanged frozen standard457/224 full support and full test',
                'official_scope_note':'Official model recipe on our frozen standard splits and seed42, not exact 5-seed paper replication',
                'strict_budget_note':'Actual forwards32/8; low-dimensional duplicate configurations explicitly audited, not claimed unique',
                'finetuning':False},
             'per_task_rss_limit_bytes':48*(1<<30),'per_task_timeout_seconds':7200,
             'no_automatic_retry':True,'training_or_old_results_modified':False}
        man['manifest_id']=object_digest(man);publish_new(out/'manifest.json',man)
        manifests.append(identity(out/'manifest.json'))
    plan={'schema':1,'name':'tabswift_standard681_dual','created_epoch':time.time(),'output_root':str(STAGE),
          'membership_count':1362,'protocol_variants':list(VARIANTS),'campaign_manifests':manifests,
          'worker_python':boot['worker_python'],'source_records':sources,
          'allocation':{'nodes':1,'tasks':4,'gpus':4,'cpus_per_task':4,'memory_gib':256,'hours':24},
          'dispatch_policy':'four physical GPUs alternate variants, atomic isolated queues; failed smoke blocks bulk',
          'bootstrap_receipt':identity(STAGE/'bootstrap_receipt.json')}
    plan['plan_id']=object_digest(plan);publish_new(STAGE/'plan.json',plan)
    print(json.dumps({'plan':str(STAGE/'plan.json'),'plan_id':plan['plan_id'],'manifests':manifests,'smoke':smoke}))

if __name__=='__main__': prepare()
