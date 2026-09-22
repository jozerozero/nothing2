import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import allocated_gpu_uuid as b


def devices():
    return [{'uuid': f'{i+1:016x}', 'pci': f'0000:{i+5:02x}:00.0'} for i in range(8)]


def mapping():
    return {'job': '999', 'node': 'node', 'gpus': devices(), 'mapping_id': 'id',
            'deadline': {'environment': {'JOB_BUDGET_JOB_ID': '999', 'JOB_BUDGET_END_MONOTONIC': '100'}}}


class BindingTests(unittest.TestCase):
    def test_partial_or_wrong_allocation_refused(self):
        raw = 'JobId=999 JobState=RUNNING NumNodes=1 NodeList=node UserId=user(2012) AllocTRES=cpu=64,mem=256G,gres/gpu=8'
        b.allocation(raw, job='999', node='node', cpus=64, mem_gib=256, uid=2012)
        b.allocation(raw.replace('256G', '262144M'), job='999', node='node', cpus=64, mem_gib=256, uid=2012)
        for bad in (raw.replace('gpu=8', 'gpu=1'), raw.replace('NumNodes=1', 'NumNodes=2'),
                    raw.replace('user(2012)', 'user(2)'), raw.replace('mem=256G', 'mem=64G')):
            with self.assertRaises(RuntimeError):
                b.allocation(bad, job='999', node='node', cpus=64, mem_gib=256, uid=2012)

    def test_real_eight_unique_gpus_required(self):
        self.assertEqual(len(b.validate_devices(devices())), 8)
        for bad in (devices()[:4], devices()[:-1]+[devices()[0]],
                    [{**g, 'uuid': '0000000000000000'} if i == 0 else g for i, g in enumerate(devices())]):
            with self.assertRaises(RuntimeError): b.validate_devices(bad)

    def test_all_eight_ranks_single_uuid_no_double_filter_aliases(self):
        found = []
        for rank in range(8):
            env = {'SLURM_JOB_ID': '999', 'SLURM_NTASKS': '8', 'SLURM_PROCID': str(rank),
                   'SLURM_LOCALID': str(rank), 'SLURM_STEP_ID': '3', 'ROCR_VISIBLE_DEVICES': str(rank),
                   'CUDA_VISIBLE_DEVICES': '0', 'HIP_VISIBLE_DEVICES': str(rank),
                   'GPU_DEVICE_ORDINAL': '0,1,2,3,4,5,6,7', 'OTHER_SETTING': 'preserved'}
            actual = b.rank_environment(mapping(), env, hostname='node')
            self.assertEqual(actual['ROCR_VISIBLE_DEVICES'], 'GPU-'+devices()[rank]['uuid'])
            self.assertTrue(all(key not in actual for key in b.MASKS[1:]))
            self.assertEqual(actual['OTHER_SETTING'], 'preserved')
            self.assertEqual(actual['SLURM_PROCID'], str(rank))
            self.assertEqual(actual['JOB_BUDGET_END_MONOTONIC'], '100')
            found.append(actual['EXPECTED_GPU_UUID'])
        self.assertEqual(len(set(found)), 8)

    def test_rank_cannot_reuse_other_job_node_or_partial_rank_environment(self):
        env = dict(SLURM_JOB_ID='999', SLURM_NTASKS='8', SLURM_PROCID='0', SLURM_LOCALID='0', SLURM_STEP_ID='1')
        for changed in ({'SLURM_JOB_ID':'998'}, {'SLURM_NTASKS':'4'}, {'SLURM_LOCALID':'1'}, {'SLURM_STEP_ID':'batch'}):
            with self.assertRaises(RuntimeError): b.rank_environment(mapping(), {**env, **changed}, hostname='node')
        with self.assertRaises(RuntimeError): b.rank_environment(mapping(), env, hostname='other')

    def test_mapping_content_or_source_change_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'mapping.json'
            data = {**mapping(), 'schema': 'allocated_gpu_uuid_v1', 'all_eight_physical_gpus_verified': True,
                    'source_sha256': hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest()}
            del data['mapping_id']; data['mapping_id'] = b.digest(data)
            path.write_text(json.dumps(data)); self.assertEqual(b.load_mapping(path), data)
            data['job'] = 'other'; path.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, 'digest'): b.load_mapping(path)

    def test_bootstrap_requires_real_one_task_step(self):
        with mock.patch.dict(b.os.environ, {'SLURM_NTASKS': '8', 'SLURM_PROCID': '0', 'SLURM_LOCALID': '0'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'one genuine'): b.bootstrap('/not-written')


if __name__ == '__main__': unittest.main()
