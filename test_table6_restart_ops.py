import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import table6_restart_ops as ops


class ContractTests(unittest.TestCase):
    def test_script_resources_and_error_propagation(self):
        script = ops.script()
        for item in ('--time=02:00:00', '--nodes=1', '--ntasks=8', '--gpus-per-task=1',
                     '--gpu-bind=single:1', '--no-requeue', '--qos=bgqos', '--nice=0',
                     '--mem=512G', '--cpus-per-task=8', 'set -euo pipefail'):
            self.assertIn(item, script)
        self.assertIn('BUDGET_EXPORTS="$(' , script)
        self.assertNotIn('eval "$(' , script)
        self.assertIn('JOB_BUDGET_END_MONOTONIC', script)

    def test_never_resubmit_existing_or_uncertain_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp)/'submission.json'
            ledger.write_text('{"state":"submitting"}')
            with patch.object(ops, 'LEDGER', ledger), patch.object(ops, 'run') as command:
                with self.assertRaises(AssertionError):
                    ops.main('submit')
                command.assert_not_called()

    def test_bounded_job_count(self):
        for count in (0, 11, 100):
            with self.assertRaises(AssertionError):
                ops.main('submit', count)

    def test_exact_control_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            with patch.object(ops,'STAGE',stage):
                (stage/'spool-123.sh').write_text(ops.script())
                fields = dict(Partition='faculty',Account='faculty-acc',QOS='bgqos',
                    NumNodes='1-1',NumTasks='8',NumCPUs='64',TimeLimit='02:00:00',
                    Nice='0',Requeue='0',Dependency='(null)',WorkDir=str(stage),
                    Command=str(stage/'run.sh'),MinMemoryNode='512G',
                    StdOut=str(ops.LOGS/'slurm-123.out'),StdErr=str(ops.LOGS/'slurm-123.err'),
                    ReqTRES='cpu=64,mem=512G,node=1,gres/gpu=8',TresPerTask='cpu=8,gres/gpu=1',
                    JobId='123',JobName='t6r22g01',JobState='PENDING',Reason='JobHeldUser',
                    ExcNodeList=ops.EXCLUDE)
                fields.update({'CPUs/Task':'8','NtasksPerN:B:S:C':'8:0:*:*'})
                def fake(args, **kwargs):
                    if args[1:3] == ['show','job']:
                        return ' '.join(f'{k}={v}' for k,v in fields.items())
                    if args[1:3] == ['show','hostnames']:
                        return 'auh7-1b-gpu-193\nauh7-1b-gpu-195'
                    raise AssertionError(args)
                with patch.object(ops,'run',side_effect=fake):
                    ops.control('123','t6r22g01',held=True)
                    for key,bad in [('ReqTRES','cpu=640,gres/gpu=80'),('JobName','t6r22g02'),
                                    ('NumNodes','2'),('Reason','Priority'),('CPUs/Task','16')]:
                        before=fields[key]
                        fields[key]=bad
                        with self.assertRaises(AssertionError):
                            ops.control('123','t6r22g01',held=True)
                        fields[key]=before


if __name__ == '__main__':
    unittest.main()
