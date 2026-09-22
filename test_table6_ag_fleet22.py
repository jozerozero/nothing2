"""No remote/Slurm/model side effects: fleet resource and preservation tests."""
import copy
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import table6_ag_fleet22 as f


class FleetTests(unittest.TestCase):
    def raw(self,parent='206117',memory='2T',cpus=128,**changes):
        fields={'JobId':parent,'JobState':'RUNNING','NumNodes':'1','NodeList':f.ALLOWED[parent],
                'UserId':f'user({os.getuid()})','NumCPUs':str(cpus),'AllocTRES':f'cpu={cpus},mem={memory},node=1',
                'TimeLimit':'3-00:00:00','RunTime':'1-00:00:00','EndTime':'2026-09-24T20:00:00'}
        fields.update(changes)
        return ' '.join(k+'='+v for k,v in fields.items())

    def proof(self,parent='206117',memory=2048,used=1045,cpus=128):
        return {'allow_cpu_sidecar':True,'allow_cpu_contention':True,'parent_job_id':parent,'node':f.ALLOWED[parent],
                'observations':[{'parent_job_id':parent,'node':f.ALLOWED[parent],'observed_epoch':t,
                                 'parent_memory_source':'cgroup_v1','parent_memory_current_bytes':used*f.GIB,
                                 'parent_memory_limit_bytes':memory*f.GIB,'available_memory_bytes':1800*f.GIB,
                                 'parent_cpu_ids':list(range(cpus))} for t in (100,115)]}

    def plan(self):
        return f.resource_plan(self.raw(),self.proof(),'206117',f.ALLOWED['206117'],now=120)

    def test_306_exact_complement_preserves_existing_step_resources(self):
        r=self.plan()
        self.assertEqual(r['cpus_per_rank'],64)
        self.assertEqual(r['memory_bytes'],128*f.GIB)
        self.assertEqual(r['rss_limit_bytes'],120*f.GIB)
        self.assertEqual(r['protected_memory_bytes'],256*f.GIB)
        self.assertEqual(r['parent_margin_bytes'],64*f.GIB)
        self.assertFalse(set(r['selected_cpu_ids']) & set(f.PROTECTED_CPUS))
        self.assertEqual(sorted(r['selected_cpu_ids']+f.PROTECTED_CPUS),list(range(128)))
        self.assertEqual(hex(sum(1<<cpu for cpu in r['selected_cpu_ids'])),'0xffff0000ffff0000ffff0000ffff0000')
        self.assertFalse(r['idle_cpu_check'])
        self.assertEqual(r['file_cache_credit_bytes'],0)

    def test_small64g_video_parents_are_blocked_even_if_node_has2t_free(self):
        for parent in ('196092','196093','200798','200797','204828','204827','204826','194259','194181','194180'):
            with self.subTest(parent=parent),self.assertRaisesRegex(RuntimeError,'actual parent memory'):
                f.resource_plan(self.raw(parent,'64G',64),self.proof(parent,64,58,64),parent,f.ALLOWED[parent],now=120)

    def test_insufficient_memory_does_not_credit_idle_or_mapped_cache(self):
        p=self.proof(used=1700)
        for row in p['observations']:
            row['idle_cpu_ids']=list(range(128));row['reclaimable_cache_bytes']=1000*f.GIB
        with self.assertRaisesRegex(RuntimeError,'actual parent memory'):
            f.resource_plan(self.raw(),p,'206117',f.ALLOWED['206117'],now=120)

    def test_stale_wrong_parent_or_unauthorized_contention_rejected(self):
        for mutate in (lambda p:p.update(allow_cpu_contention=False),
                       lambda p:p.update(parent_job_id='206116'),
                       lambda p:p['observations'][0].update(observed_epoch=-300),
                       lambda p:p['observations'][1].update(observed_epoch=110)):
            p=self.proof();mutate(p)
            with self.assertRaises(RuntimeError):
                f.resource_plan(self.raw(),p,'206117',f.ALLOWED['206117'],now=120)

    def test_wrong_owner_multinode_nonrunning_or_unlisted_parent_block(self):
        for changes in ({'JobState':'PENDING'},{'NumNodes':'8'},{'UserId':'stranger(99999)'},{'NodeList':'wrong'}):
            with self.assertRaises(RuntimeError):f.parent_fields(self.raw(**changes),'206117',f.ALLOWED['206117'])
        with self.assertRaises(RuntimeError):f.parent_fields(self.raw(),'206116','auh7-1b-gpu-308')

    def test_srun_is_one_cpu_only_masked_overlap_step_not_new_parent(self):
        command=f.srun_command('206117',f.ALLOWED['206117'],self.plan(),Path('/new/launch'))
        for value in ('--ntasks=1','--cpus-per-task=64','--mem=128G','--gpus=0','--gpus-per-task=0',
                      '--gres=none','--overlap','--exact','--time=02:00:00',
                      '--cpu-bind=mask_cpu:0xffff0000ffff0000ffff0000ffff0000'):
            self.assertIn(value,command)
        self.assertNotIn('sbatch',command);self.assertNotIn('scancel',command)
        env=f.clean_environment(64)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            self.assertEqual(env[key],'64')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')

    def test_short_parent_cap_shrinks_and_almost_expired_parent_blocks(self):
        raw=self.raw(parent='208823',memory='512G',RunTime='00:45:00',TimeLimit='01:00:00')
        with self.assertRaisesRegex(RuntimeError,'insufficient bounded'):
            f.resource_plan(raw,self.proof('208823',512,50),'208823',f.ALLOWED['208823'],now=120)
        raw=self.raw(parent='208823',memory='512G',RunTime='00:44:00',TimeLimit='01:00:00')
        r=f.resource_plan(raw,self.proof('208823',512,50),'208823',f.ALLOWED['208823'],now=120)
        self.assertEqual(r['step_time_limit_seconds'],660)
        command=f.srun_command('208823',f.ALLOWED['208823'],r,Path('/new'))
        self.assertIn('--time=00:11:00',command)

    def test_runtime_parent_guard_reserves_v5_peak_no_scientific_error(self):
        r=self.plan()
        snapshot={'parent_memory_current_bytes':1045*f.GIB,'parent_memory_limit_bytes':2048*f.GIB,
                  'available_memory_bytes':1800*f.GIB}
        f.check_memory(snapshot,r,startup=True)
        snapshot['parent_memory_current_bytes']=1740*f.GIB
        with self.assertRaisesRegex(RuntimeError,'whole_parent_memory_guard'):f.check_memory(snapshot,r)

    def test_scientific_time_budget_none_and_source_pins(self):
        pairs=[{'config':{'time_limit':None}}]*457
        with patch.object(f,'ORIGINAL_LOAD',return_value=('original','plan',pairs)):
            self.assertEqual(f.load_frozen(),('original','plan',pairs))
        with patch.object(f,'ORIGINAL_LOAD',return_value=('original','plan',[{'config':{'time_limit':3600}}])):
            with self.assertRaises(RuntimeError):f.load_frozen()
        self.assertEqual({name:f.sources()[name] for name in f.PINS},f.PINS)
        self.assertEqual(f.ag.SEEDS,list(range(15)))

    def test_rank_refuses_any_binding_mismatch_before_fitting(self):
        plan={'parent':'206117','node':f.ALLOWED['206117'],'resources':self.plan()}
        env=dict(f.ag.CPU_ENV,SLURM_JOB_ID='206117',SLURM_STEP_ID='400',SLURM_PROCID='0',
                 SLURM_NTASKS='1',SLURM_CPUS_PER_TASK='64',SLURM_NNODES='1',GPU_DEVICE_ORDINAL='-1')
        with patch.object(f,'verify_plan',return_value=plan),patch.dict(os.environ,env),\
                patch.object(f.socket,'gethostname',return_value=plan['node']),\
                patch.object(f.os,'sched_getaffinity',return_value=set(range(64)),create=True):
            with self.assertRaisesRegex(RuntimeError,'never widen'):
                f.rank_entry('/new')

    def test_private_frozen_work_dispatches_native_guards_and_keeps_old_module(self):
        plan={'parent':'206117','node':f.ALLOWED['206117'],'resources':self.plan(),
              'ranks':1,'launch_id':'unique','allocated_memory_bytes':2048*f.GIB}
        env={'SLURM_JOB_ID':'206117','SLURM_STEP_ID':'400'}
        attrs=('CAMPAIGN','terminal_evidence','preflight','run_seed','publish')
        saved={name:getattr(f.ag,name) for name in attrs}
        old_stage=f.v3.STAGE
        try:
            with patch.dict(f.PRIVATE,verify_plan=lambda directory:plan,
                            dynamic_preflight=lambda *args:('new_preflight',args[-1])),\
                    patch.dict(os.environ,env),patch.object(f.socket,'gethostname',return_value=plan['node']),\
                    patch.object(f.v3.EnvironmentBudget,'from_environment',return_value='budget'),\
                    patch.object(f.ag,'main',return_value=0):
                self.assertEqual(f.PRIVATE['work_entry']('/new'),0)
                self.assertEqual(f.ag.preflight('original','budget','audit','stop'),('new_preflight',1))
                self.assertIs(f.ag.terminal_evidence,f.exact_step_evidence)
                self.assertEqual(f.ag.CAMPAIGN,f.CAMPAIGN)
                self.assertIsNot(f.ag.publish,f.publish)
                self.assertEqual(f.v3.STAGE,old_stage)
        finally:
            for name,value in saved.items():setattr(f.ag,name,value)

    def test_memory_guard_before_fit_does_not_spawn_or_write_model_error(self):
        import sys
        plan={'parent':'206117','node':f.ALLOWED['206117'],'resources':self.plan(),'launch_id':'one'}
        native=[sys.executable,'-B',str(f.v3.REPO/'table6_restart_ag.py'),'seed','--key','a'*24,'--seed','0']
        with patch.object(f.v3,'memory_snapshot',return_value={'parent_memory_current_bytes':2000*f.GIB,
                'parent_memory_limit_bytes':2048*f.GIB,'available_memory_bytes':1800*f.GIB}),\
                patch.object(f.subprocess,'Popen') as spawn:
            result=f.guarded_run_seed(native,'/never-opened',3,types.SimpleNamespace(remaining=lambda:500),
                                      types.SimpleNamespace(reason=None),plan)
            self.assertIsNone(result[0]);self.assertIn('memory_guard',result[1]);spawn.assert_not_called()


if __name__=='__main__':unittest.main()
