"""CPU-only tests: no scheduler, network, fit, or real result writes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import shared_eval_plan as p


def observation(kind):
    t = p.TARGETS[kind]
    parent, node = t['parent_job_id'], t['node']
    ids = [t['gpu']['uuid']] + [f'{i:016x}' for i in range(1, 8)]
    cards = [dict(uuid=value, pci=t['gpu']['pci'] if i == 0 else f'0000:{i+32:02x}:00.0',
                  hardware_idle=True, busy_percent=0, vram_used_bytes=0, foreign_fd_owner_pids=[])
             for i, value in enumerate(ids)]
    processes = []

    def add(pid, cmd, step='batch', parent_pid=0, rocr='0,1,2,3,4,5,6,7', probe=False):
        env = {'SLURM_JOB_ID': parent, 'ROCR_VISIBLE_DEVICES': rocr}
        if step != 'batch':
            env['SLURM_STEP_ID'] = step
        record = dict(pid=pid, parent_pid=parent_pid, start_ticks=pid*100,
                      cmdline=cmd, selected_environment=env, rss_bytes=100,
                      cgroup=['0::/system.slice/slurmstepd.scope/job_'+parent+'/step_'+step+'/user/task_0'],
                      gpu_fds=[], is_probe=probe)
        processes.append(record)
        return record

    add(1, ['/bin/sh', '/var/spool/slurmd/job'+parent+'/slurm_script'])
    add(2, ['bash', '-lc', 'echo ['+node+'] holding allocation across nodes: '+node+' ; trap : TERM INT; sleep infinity'], parent_pid=1)
    add(3, ['sleep', 'infinity'], parent_pid=2)
    if kind == 'gap190':
        add(4, [p.SUPERVISOR_PY, p.VIDEO+'/node_supervisor.py', '--cards', '1,2,3,4,5,6,7'], '662')
    for i, card in enumerate(t['worker_cards']):
        worker = add(10+card, [p.VIDEO_PY, '-u', p.VIDEO+'/campaign.py', 'worker', '--worker',
                              'node'+node.rsplit('-', 1)[-1]+'_gpu'+str(card), '--memory-mode', 'stream'],
                     '662' if kind == 'gap190' else str(card+50), 4 if kind == 'gap190' else 777,
                     rocr='GPU-'+ids[i+1])
        worker['selected_environment']['GPU_DEVICE_ORDINAL'] = '0,1,2,3,4,5,6,7'
        worker['gpu_fds'] = [dict(uuid=g['uuid'], pci=g['pci']) for g in cards]
    add(99, ['/python', '-B', str(p.HERE/'shared_gpu_capacity_probe.py'), '--node', '--parent', parent,
             '--expected-node', node], '999', rocr='-1', probe=True)
    return dict(parent=parent, node=node, complete=True, source_sha256='probe', epoch=1200,
                job_fields={'JobId': parent, 'JobState': 'RUNNING', 'NumNodes': '1', 'NodeList': node,
                            'UserId': 'guangyi.chen(1000)', 'AllocTRES': 'cpu=64,mem=64G,gres/gpu=8',
                            'TimeLimit': '2-00:00:00', 'RunTime': '01:00:00'},
                own_process_inspection_complete=True, process_errors=[], foreign_fd_inspection_unknown=[],
                owned_processes=processes, probe_pid=99, same_uid_rss_bytes=sum(x['rss_bytes'] for x in processes),
                gpus=cards, node_memory_available_bytes=100*p.GIB,
                parent_cpu_proof_complete=True, idle_cpu_threshold_percent_exclusive=25,
                parent_batch_cpu_ids=list(range(64)), cpu_percent_samples=[[0.0]*128, [0.0]*128],
                idle_parent_cpu_ids_twice=list(range(64)))


class PlanTests(unittest.TestCase):
    def test_both_reviewed_exact_rosters_and_enumeration_fds(self):
        for kind in p.TARGETS:
            r = p.reviewed_processes(observation(kind), p.TARGETS[kind])
            self.assertEqual(len(r['worker_uuid_masks']), len(p.TARGETS[kind]['worker_cards']))
            self.assertNotIn('GPU-'+p.TARGETS[kind]['gpu']['uuid'], r['worker_uuid_masks'])

    def test_hold_command_no_step_environment_is_real_batch(self):
        r = observation('gap190')
        self.assertNotIn('SLURM_STEP_ID', r['owned_processes'][0]['selected_environment'])
        p.reviewed_processes(r, p.TARGETS['gap190'])

    def test_unknown_command_rejected(self):
        r = observation('gap190'); r['owned_processes'][4]['cmdline'] = ['python', 'future_training.py']
        with self.assertRaisesRegex(RuntimeError, 'Unreviewed existing command'):
            p.reviewed_processes(r, p.TARGETS['gap190'])

    def test_target_uuid_and_broad_worker_mask_rejected(self):
        for mask in ('GPU-'+p.TARGETS['gap190']['gpu']['uuid'], '0,1,2,3,4,5,6,7'):
            r = observation('gap190'); r['owned_processes'][4]['selected_environment']['ROCR_VISIBLE_DEVICES'] = mask
            with self.assertRaisesRegex(RuntimeError, 'Worker GPU mask'):
                p.reviewed_processes(r, p.TARGETS['gap190'])

    def test_future_supervisor_extra_card_rejected(self):
        r = observation('gap190'); r['owned_processes'][3]['cmdline'][-1] = '0,1,2,3,4,5,6,7'
        with self.assertRaises(RuntimeError):
            p.reviewed_processes(r, p.TARGETS['gap190'])

    def test_privileged_uncertainty_explicit_nonroot_rejected(self):
        r = observation('swiftdual')
        r['foreign_fd_inspection_unknown'] = [{'pid': 888, 'uid': 0, 'reason': 'foreign_fd_permission_denied'}]
        self.assertFalse(p.reviewed_processes(r, p.TARGETS['swiftdual'])['all_uid_ownership_proven'])
        r['foreign_fd_inspection_unknown'][0]['uid'] = 1234
        with self.assertRaisesRegex(RuntimeError, 'foreign non-root'):
            p.reviewed_processes(r, p.TARGETS['swiftdual'])

    def sample_pair(self, kind='gap190'):
        a = dict(schema='shared_gpu_capacity_sample_v1', source_sha256='probe', started_epoch=960,
                 epoch=970, parents=[observation(kind)])
        b = copy.deepcopy(a); b.update(started_epoch=985, epoch=990)
        b['parents'][0]['epoch'] += 20
        return a, b

    def test_known_os_and_login_services_are_not_model_workers(self):
        item={'pid':777,'uid':1001,'username':'ubuntu','comm':'bash',
              'reason':'foreign_fd_permission_denied',
              'cgroup':['0::/user.slice/user-1001.slice/session-1598.scope']}
        self.assertTrue(p.reviewed_foreign_service(item))
        self.assertFalse(p.reviewed_foreign_service(dict(item,comm='python')))
        self.assertFalse(p.reviewed_foreign_service(dict(item,uid=2013)))
        self.assertFalse(p.reviewed_foreign_service(dict(item,cgroup=['0::/job_555/step_0'])))

    def proof(self, a, b, kind='gap190', now=1000):
        with mock.patch.object(p, 'read_pinned', side_effect=[({'path': '/a'}, a), ({'path': '/b'}, b)]), \
             mock.patch.object(p, 'identity', return_value={'sha256': 'probe'}):
            return p.proof(['/a', '/b'], p.TARGETS[kind], now=now)

    def test_proof_login_not_node_clock(self):
        a, b = self.sample_pair()
        proof, review = self.proof(a, b)
        self.assertEqual(proof['controller_finished_epoch'], 990)
        self.assertTrue(review['approved'])
        self.assertEqual(len(proof['idle_parent_cpu_ids_both_samples']), 64)

    def test_stale_close_changed_ownership_and_ram_fail_closed(self):
        a, b = self.sample_pair()
        mutations = [lambda x: x.update(started_epoch=984),
                     lambda x: x.update(epoch=600),
                     lambda x: x['parents'][0]['owned_processes'][4].update(start_ticks=9),
                     lambda x: x['parents'][0].update(same_uid_rss_bytes=23*p.GIB),
                     lambda x: x['parents'][0]['gpus'][0].update(foreign_fd_owner_pids=[123])]
        for change in mutations:
            changed = copy.deepcopy(b); change(changed)
            with self.assertRaises(RuntimeError): self.proof(a, changed)

    def test_insufficient_idle_cpu_intersection(self):
        a, b = self.sample_pair()
        for sample, allowed in [(a, range(0, 16)), (b, range(16, 32))]:
            r = sample['parents'][0]
            r['idle_parent_cpu_ids_twice'] = list(allowed)
            r['cpu_percent_samples'] = [[0 if c in allowed else 99 for c in range(128)]]*2
        with self.assertRaisesRegex(RuntimeError, 'Insufficient CPUs'):
            self.proof(a, b)

    def test_independent_failure_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(p.launch, 'STAGE', Path(tmp)):
            with mock.patch.object(p, 'build', side_effect=[RuntimeError('RAM'), {'sidecar_id': 'try-swiftdual', 'plan_id': 'abc'}]):
                result = p.build_both(['/a', '/b'], 'try', '/g', '/o', '/b')
            self.assertEqual(result['gap190']['state'], 'not_created')
            self.assertEqual(result['swiftdual']['state'], 'plan_created_not_launched')
            path = Path(result['swiftdual']['path']); original = path.read_bytes()
            with mock.patch.object(p, 'build', return_value={'sidecar_id': 'try-swiftdual', 'plan_id': 'changed'}):
                again = p.build_both(['/a', '/b'], 'try', '/g', '/o', '/b')
            self.assertEqual(again['swiftdual']['state'], 'not_created')
            self.assertEqual(path.read_bytes(), original)

    def test_identity_rejects_symlink_and_pins_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'source'; path.write_text('content')
            rec = p.identity(path); self.assertEqual(p.q.verify_file(rec), path.resolve())
            link = Path(tmp)/'link'; link.symlink_to(path)
            with self.assertRaisesRegex(RuntimeError, 'symlink'): p.identity(link)
            path.write_text('changed')
            with self.assertRaises(RuntimeError): p.q.verify_file(rec)


if __name__ == '__main__':
    unittest.main()
