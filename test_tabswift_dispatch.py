import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import tabswift_dispatch as dispatch

class QueueTest(unittest.TestCase):
    def test_protocols_alternate_without_losing_tasks(self):
        tasks=[{'task_kind':'classification' if i<457 else 'regression',
                'dataset_index':i if i<457 else i-457,'dataset':f'dataset-{i}','row':{'work_size':i+1}}
               for i in range(681)]
        campaigns=[(None,None,list(reversed(tasks))),(None,None,tasks)]
        order=list(dispatch.pending_order(campaigns))
        self.assertEqual(len(order),1362)
        self.assertEqual([v for v,_ in order],[0,1]*681)
        for a,b in zip(order[::2],order[1::2]):self.assertEqual(a[1],b[1])
        self.assertEqual(len({(v,t['task_kind'],t['dataset_index']) for v,t in order}),1362)

    def test_budget_shortfall_is_rejected_but_official_native_is_allowed(self):
        task={'task_kind':'regression'}
        strict={'protocol':{'variant':'budget32x8','strict_actual_count':True,'n_estimators':{'regression':8}}}
        official={'protocol':{'strict_actual_count':False}}
        with patch.object(dispatch,'NATIVE_VALID',return_value={'actual_ensemble_count':6}):
            with self.assertRaisesRegex(RuntimeError,'Strict actual'):dispatch.validated_result(None,strict,task)
            self.assertEqual(dispatch.validated_result(None,official,task)['actual_ensemble_count'],6)
        with patch.object(dispatch,'NATIVE_VALID',return_value={'actual_ensemble_count':8}):
            self.assertEqual(dispatch.validated_result(None,strict,task)['actual_ensemble_count'],8)

    def test_plan_digest_failure_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json';p.write_text(json.dumps({'plan_id':'bad'}))
            with self.assertRaisesRegex(RuntimeError,'Plan digest'):dispatch.load_plan(p)

if __name__=='__main__':unittest.main()
