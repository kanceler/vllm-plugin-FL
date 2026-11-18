"""Serial CPU -> GPU regression/operator -> micro gates. No model jobs here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

PYTHON='/workspace/tianshu-mixed-env-20261109/bin/python'


def free_resources(root,stage):
    gpu=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
    (root/(stage+'-gpu-before.txt')).write_text(gpu)
    (root/(stage+'-processes-before.txt')).write_text(subprocess.check_output(['ps','-eo','pid,etime,args'],text=True))
    if 'No running processes found' not in gpu:raise RuntimeError('GPU occupied: '+stage)
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',9031))==0:raise RuntimeError('9031 listener')
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('0.0.0.0',9031))


def environment(root,stage,static):
    env=dict(os.environ)
    for name in list(env):
        if name.startswith(('VLLM_FL_','MIXED_SMALL_','GEMM_DIAG_')) or name in ('PYTHONPATH','FLAGGEMS_DB_URL'):
            del env[name]
    env.update(PYTHONPATH=str(root/'source'),VLLM_PLUGINS='fl',PYTHONUNBUFFERED='1',
        PATH='/usr/local/corex/bin:/usr/local/bin:/usr/bin:/bin',
        LD_LIBRARY_PATH='/usr/local/corex/lib64:/usr/local/openmpi/lib:/usr/local/lib',
        VLLM_FL_K_STATIC_FOLD='1' if static else '0',
        VLLM_FL_INT8_KV_REPAIR='1' if static else '0',
        VLLM_FL_K_TOP4='0',VLLM_FL_K_TOP4_JOINT='0',VLLM_FL_V_ONLY='0',
        VLLM_FL_MIXED_ATTN_SPLIT='1',VLLM_FL_BIV150_DECODE_DIRECT_INDEX='1',
        VLLM_FL_BIV150_ROW_PARALLEL_SAMPLING='1',TORCHINDUCTOR_COMPILE_THREADS='1')
    for group in ('DOWN','GATE_UP','QKV','O'):env['VLLM_FL_TIANSHU_W8A16_'+group]='0'
    db=root/(stage+'-flaggems.db');shutil.copy2(root/'initial-flaggems.db',db)
    env['FLAGGEMS_DB_URL']='sqlite:///'+str(db)
    return env


def stage(root,name,command,static):
    free_resources(root,name)
    env=environment(root,name,static)
    record=dict(command=command,env={k:v for k,v in env.items() if k.startswith(('VLLM','PYTHONPATH','FLAGGEMS')) or k in ('PATH','LD_LIBRARY_PATH')})
    (root/(name+'-command.json')).write_text(json.dumps(record,indent=2))
    with (root/(name+'.log')).open('x') as log:
        process=subprocess.Popen(command,cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
        (root/'gate-state.json').write_text(json.dumps(dict(stage=name,pid=process.pid,status='running'),indent=2))
        code=process.wait()
    (root/(name+'.exit')).write_text(str(code))
    for _ in range(20):
        gpu=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
        if 'No running processes found' in gpu:break
        time.sleep(1)
    (root/(name+'-gpu-after.txt')).write_text(gpu)
    if code:raise RuntimeError(name+' failed: '+str(code))
    if 'No running processes found' not in gpu:raise RuntimeError('GPU not released')


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--micro',action='store_true')
    p.add_argument('--operators-only',action='store_true')
    p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();r=a.root
    assert a.attempt>=1
    assert (r/'CPU_COMPLETE').exists()
    try:
        if a.micro:
            assert (r/'GPU_COMPLETE').exists(), 'GPU gates must pass before micro'
            stage(r,'micro-01',[PYTHON,'benchmarks/diagnostics/validate_static_kv_013_compat.py',str(r/'micro-01'),'--micro'],True)
            assert (r/'micro-01/COMPLETE').exists()
            (r/'MICRO_COMPLETE').write_text('Paired full calls completed; review raw timings before model gate\n')
        else:
            if a.operators_only:
                assert (r/'legacy-regression.exit').read_text()=='0'
            else:
                stage(r,'legacy-regression',[PYTHON,'-m','pytest','-q',
                    'tests/unit_tests/attention/test_mixed_policy.py',
                    'tests/unit_tests/attention/test_decode_index_policy.py',
                    'tests/unit_tests/patches/test_iluvatar_sampling.py'],False)
            name=f'operators-{a.attempt:02d}'
            stage(r,name,[PYTHON,'benchmarks/diagnostics/validate_static_kv_013_compat.py',str(r/name)],True)
            assert (r/name/'COMPLETE').exists()
            (r/'GPU_COMPLETE').write_text('Original BF16 regressions + static KV byte/numeric/replay gates passed\n')
        (r/'gate-state.json').write_text(json.dumps(dict(status='complete',micro=a.micro),indent=2))
    except BaseException as e:
        (r/'gate-state.json').write_text(json.dumps(dict(status='failed',error=repr(e)),indent=2));raise


if __name__=='__main__':main()
