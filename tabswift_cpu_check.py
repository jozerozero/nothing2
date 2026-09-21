"""Small frozen-input/native-fit validation only; not GPU smoke or scored evaluation."""
import gc
import json
import os
import random
import time
from pathlib import Path
from eval_one import publish_new, require
from tabswift_bootstrap import STAGE

def main():
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    import numpy as np
    import pandas as pd
    import torch
    import tabswift_one as worker
    from tabswift_ensemble import configure,configuration_records
    from tabswift_dispatch import load_plan
    torch.set_num_threads(4)
    plan,campaigns=load_plan(STAGE/'plan.json')
    records=[]
    for path,man,tasks in campaigns:
        man=worker.load_campaign(path)
        for task in man['smoke_tasks']:
            kind,index=task['task_kind'],task['dataset_index']
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            loader=worker.data_helper.load_classification if kind=='classification' else worker.data_helper.load_regression
            data,row,tx,ys,vx,yt,data_audit,raw_files=loader(man,index,np,pd)
            estimator,processing,settings=worker.import_native(man,kind)
            estimator.device='cpu'
            train,fit_y,test,info,enc,pre=worker.official_preprocess(tx,ys,vx,kind,processing)
            with worker.checkpoint_load_guard(torch,man['weights']['shared']):
                handle=configure(estimator,kind,estimator.n_estimators,
                                 strict_actual_count=man['protocol']['strict_actual_count'])
                try:
                    estimator.fit(train,fit_y)
                    configs=configuration_records(estimator.ensemble_generator_,kind)
                    require(not estimator.model_.training,'Model unexpectedly in training mode')
                    if man['protocol']['strict_actual_count']:
                        require(len(configs)==man['protocol']['n_estimators'][kind],'Strict config count failed')
                    records.append({'variant':man['protocol']['variant'],'dataset':row['dataset'],
                         'task_kind':kind,'index':index,'support_rows':len(train),'test_rows':len(test),
                         'features':train.shape[1],'configured_members':len(configs),
                         'native_model_loaded':True,'preprocessing_finite':True,
                         'native_fit_passed':True,'inference_executed':False})
                finally:handle.close()
            del estimator,train,fit_y,test,tx,vx,ys,yt
            gc.collect()
    receipt={'plan_id':plan['plan_id'],'epoch':time.time(),'cpu_only':True,'gpu_smoke_passed':False,
             'full_evaluation_started':False,'checks':records}
    publish_new(STAGE/'cpu_preparation_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
