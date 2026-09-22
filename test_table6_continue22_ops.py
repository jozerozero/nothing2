import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch
import table6_continue22_ops as ops

class Contracts(TestCase):
    def test_fields_refuse_changed_resources_or_sources(self):
        m=SimpleNamespace(STAGE=Path('/s'),LOGS=Path('/logs'))
        item={'kind':'r19','job_id':'1','name':'t6r22u01'}
        d={'JobId':'1','JobName':'t6r22u01','Partition':'faculty','Account':'faculty-acc','QOS':'bgqos',
           'NumTasks':'8','NumCPUs':'64','CPUs/Task':'8','MinMemoryNode':'512G','TimeLimit':'02:00:00',
           'Nice':'0','Requeue':'0','Command':'/s/run.sh','WorkDir':'/s','StdOut':'/logs/slurm-1.out',
           'StdErr':'/logs/slurm-1.err','NumNodes':'1','ReqTRES':'cpu=64,gres/gpu=8',
           'NtasksPerN:B:S:C':'8:0:*:*','Dependency':'(null)','JobState':'PENDING','Reason':'JobHeldUser'}
        with patch.object(ops,'modules',return_value=(m,m)):
            self.assertIs(ops.validate_fields(d,item),m)
            for key in ('NumCPUs','CPUs/Task','TimeLimit','Command','QOS','Nice','ReqTRES','Dependency','Reason'):
                with self.assertRaises(RuntimeError):ops.validate_fields({**d,key:'wrong'},item)

    def test_existing_ledger_blocks_submit_before_any_remote_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'submission.json';p.write_text('{}')
            with patch.object(ops,'LEDGER',p),patch.object(ops,'modules',return_value=(None,None)),patch.object(ops,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'no retry'):ops.main('submit')
                run.assert_not_called()

    def test_guard_rejects_active_old_or_new_campaign(self):
        m=SimpleNamespace(STAGE=Path('/new'))
        for name in ('t6r22g01','t6r22u01','t6g22r2'):
            with patch.object(ops,'modules',return_value=(m,m)),patch.object(ops,'run',return_value=f'123|{name}|RUNNING|/other'):
                with self.assertRaisesRegex(RuntimeError,'active old/new'):ops.guard()

if __name__=='__main__':main()
