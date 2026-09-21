"""Install TabSwift in a new private environment, leaving all active environments alone."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

BASE = Path('/vast/users/guangyi.chen/causal_group/zijian.li/codex/all178_crossfit_20260722_v1')
STAGE = BASE / 'stage/tabswift_standard681_20260922_v1'
COMMIT = '8edf8f0b4225bc03e1f5db011912619cd92b798d'
REVISION = 'b829456edb7c41ad93a2851a8df245db362e1c83'
WEIGHT_SHA = '16e324177be2ab9e2bac15e5edf7867329e6595e134a5c4c9b6d79d3b657b363'
WEIGHT_SIZE = 32947079

def run(args):
    return subprocess.check_output([str(x) for x in args], text=True).strip()

def main():
    from eval_one import publish_new, require
    require(not (STAGE/'bootstrap_receipt.json').exists(), 'Bootstrap already complete; do not reinstall')
    STAGE.mkdir(parents=True, exist_ok=True)
    official = STAGE/'official'
    if not official.exists():
        subprocess.run(['git','clone','https://github.com/LAMDA-Tabular/TabSwift.git',str(official)],check=True)
    require(not run(['git','-C',official,'status','--porcelain']), 'Official checkout is dirty')
    subprocess.run(['git','-C',str(official),'checkout','--detach',COMMIT],check=True)
    require(run(['git','-C',official,'rev-parse','HEAD']) == COMMIT,'Source commit mismatch')
    environment = STAGE/'venv'
    if not environment.exists():
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(environment)],check=True)
    python = environment/'bin/python'
    subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','--no-deps',
                    'scikit-learn==1.6.1','category-encoders==2.8.1'],check=True)
    weights = STAGE/'weights'; weights.mkdir(exist_ok=True)
    target = weights/'swift.ckpt'
    if not target.exists():
        temporary = weights/f'.swift-{os.getpid()}.download'
        url = f'https://huggingface.co/LAMDA-Tabular/TabSwift/resolve/{REVISION}/swift.ckpt'
        with urllib.request.urlopen(url,timeout=180) as response, temporary.open('xb') as output:
            while block := response.read(1<<20):
                output.write(block)
            output.flush(); os.fsync(output.fileno())
        require(temporary.stat().st_size == WEIGHT_SIZE and
                hashlib.sha256(temporary.read_bytes()).hexdigest() == WEIGHT_SHA,'Downloaded weight identity mismatch')
        os.link(temporary,target); temporary.unlink()
    require(target.stat().st_size == WEIGHT_SIZE and
            hashlib.sha256(target.read_bytes()).hexdigest() == WEIGHT_SHA,'Existing weight identity mismatch')
    check = '''import importlib.metadata as m,json,sys
sys.path[:0]=[sys.argv[1],sys.argv[1]+'/TALENT/model/lib']
import torch,category_encoders,sklearn
from TALENT.model.lib.data import data_nan_process,data_enc_process,data_label_process
from tabswift import TabSwiftClassifier
from tabswift.regressor import TabSwiftRegressor
assert sklearn.__version__=='1.6.1'
assert hasattr(TabSwiftClassifier,'_validate_data')
print(json.dumps({p:m.version(p) for p in ['torch','numpy','scipy','scikit-learn','pandas','category-encoders','huggingface-hub','psutil']}))
'''
    versions = json.loads(run([python,'-c',check,official]).splitlines()[-1])
    receipt = {'created_epoch':time.time(),'source_commit':COMMIT,'weight_revision':REVISION,
               'weight_sha256':WEIGHT_SHA,'weight_bytes':WEIGHT_SIZE,'versions':versions,
               'worker_python':str(python),'source':str(official),
               'existing_environments_modified':False,'gpu_smoke_passed':False}
    publish_new(STAGE/'bootstrap_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)

if __name__ == '__main__': main()
