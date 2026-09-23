"""Native CPU forwards for the FP32 repair; never publish scored GPU results."""
import argparse
import gc
import json
import os
import random
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                      NUMEXPR_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import numpy as np
    import pandas as pd
    import torch
    import tabswift_one_v2 as worker
    from tabswift_dispatch import load_plan
    from tabswift_ensemble import configure
    from eval_one import publish_new, require
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_available(), 'CPU-only probe unexpectedly sees a GPU')
    plan, campaigns = load_plan(args.plan)
    checks = []
    for path, man, _ in campaigns:
        man = worker.load_campaign(path)
        # Binary classification exercises the formerly unconditional AMP path;
        # low-dimensional regression exercises native/strict ensemble handling.
        for task in (man['smoke_tasks'][0], man['smoke_tasks'][2]):
            kind, index = task['task_kind'], task['dataset_index']
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            loader = worker.data_helper.load_classification if kind == 'classification' else worker.data_helper.load_regression
            _, row, tx, ys, vx, yt, _, _ = loader(man, index, np, pd)
            estimator, processing, settings = worker.import_native(man, kind)
            estimator.device = 'cpu'
            train, fit_y, test, _, _, _ = worker.official_preprocess(tx, ys, vx, kind, processing)
            started = time.monotonic()
            with torch.inference_mode(), worker.checkpoint_load_guard(torch, man['weights']['shared']):
                prediction, audit = worker.predict_native(estimator, train, fit_y, test, kind,
                    man['protocol']['strict_actual_count'], configure, torch)
            record = {'variant':man['protocol']['variant'], 'task_kind':kind,
                      'dataset_index':index, 'dataset':row['dataset'],
                      'support_rows':len(train), 'test_rows':len(test), 'predictions':len(prediction),
                      'elapsed_seconds':time.monotonic()-started, 'ensemble_audit':audit,
                      'cpu_only':True, 'gpu_smoke_passed':False, 'formal_result':False}
            checks.append(record)
            print(json.dumps(record), flush=True)
            del estimator, tx, ys, vx, yt, train, fit_y, test, prediction
            gc.collect()
    receipt = {'plan_id':plan['plan_id'], 'epoch':time.time(), 'checks':checks,
               'cpu_native_forward_passed':True, 'gpu_smoke_passed':False,
               'formal_results_published':0, 'new_allocation':False}
    publish_new(args.plan.parent/'cpu_native_validation.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
