"""Pinned Loop4 multi-QOS deployment contract; never discover a replacement lineage."""
import json, os
from pathlib import Path

ROOT=Path("/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1")
TAG="e4_g5sc_loop4_resume180825_eval_fp32_online_gt8_st4_step50_20260910_v1"
STAGE=ROOT/"stage"/TAG/"online_resume"
OLD=ROOT/"checkpoints/e4_g5_support_condition_alpha_loops_20260907_v1/g36-g5scalpha-loop4-histe4-25k-v1/e4g5sc4l25v1-177623"
NEW=ROOT/"checkpoints/e4_g5_support_condition_alpha_loops_20260907_v1/g36-g5scalpha-loop4-histe4-25k-v1/e4g5sc4lr1-180825"
OUT=ROOT/"evaluation/e4_g5sc_loop4_resume180825_fp32_online_gt8_st4_step50_20260910_v1/E4_G5SC_LOOP4/lineage-177623-180825"
LEGACY=ROOT/"evaluation/e4_g5sc_loop4_177623_fp32_online_12gpu_gt_step50_20260908_v1/E4_G5SC_LOOP4/train-177623"
BASE=ROOT/"evaluation/cross_table_e4_data_only_4096_fp32_online_4gpu_st_step50_20260818_v2/RWSAMPLE50_CROSS_TABLE_E4/train-142620"
LOG=ROOT/"logs"/TAG
RECEIPT=STAGE/"submission_state.json"
SPECS={
 "gtqos":{"qos":"gtqos","job_name":"e4g5sc4r8gt1","account":"faculty-acc","nodes":2,"gpus":8,"group_count":2,"time_limit":"3-00:00:00","slurm_file":"slurm_gt.sh"},
 "stqos":{"qos":"stqos","job_name":"e4g5sc4r4st1","account":"test-acc","nodes":1,"gpus":4,"group_count":1,"time_limit":"1-00:00:00","slurm_file":"slurm_st.sh"},
}

def registered_jobs():
    receipt=json.loads(RECEIPT.read_text())
    assert receipt["registration_complete"] is True
    assert set(receipt["jobs"])==set(SPECS)
    jobs={}
    for qos, spec in SPECS.items():
        job=receipt["jobs"][qos]
        for key, value in spec.items(): assert job[key]==value,(key,job,value)
        job_id=str(job["evaluation_job"])
        assert job_id.isdigit() and job_id not in jobs and job_id not in ("181417","180467","178889","180825")
        jobs[job_id]=job
    return jobs

def expected_job():
    qos=os.environ["SLURM_JOB_QOS"]
    spec=SPECS[qos]
    assert os.environ["SLURM_JOB_NAME"]==spec["job_name"]
    return spec

def runtime_check():
    spec=expected_job()
    assert int(os.environ["SLURM_NNODES"])==spec["nodes"]
    assert int(os.environ["SLURM_NTASKS"])==spec["gpus"]
    assert registered_jobs()[os.environ["SLURM_JOB_ID"]]["qos"]==spec["qos"]
    assert Path(__file__).resolve().parent==STAGE
    print("registered_loop4_evaluator=1 qos="+spec["qos"]+" job="+os.environ["SLURM_JOB_ID"],flush=True)

if __name__=="__main__": runtime_check()

